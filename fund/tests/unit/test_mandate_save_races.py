"""Mandate saves must not silently undo each other.

A save is read-modify-write: the Mandate view builds the whole mandate from the copy it
polled, so a change another writer landed since (a remembered directive, the CLI's
``mandate set``) was overwritten by the stale fields. The save path now refuses a save
whose base no longer matches the stored mandate, takes the row lock before it writes,
and the TUI lets only one save run at a time.

SQLite cannot show row locking, so the lock itself is asserted on the statement the
repository builds; the compare-and-refuse and single-flight logic run for real.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

import pytest
from _fund_fakes import (
    PORTFOLIO_ID,
    make_mandate,
    threaded_sqlite_engine,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import PinnedInstrument
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from textual.widgets import Input

from fund import observe
from fund.actions import apply_action
from fund.audit import HumanAuditLogRepository, MandateRepository
from fund.tui.app import ShellApp
from fund.tui.views import MandateView


@pytest.fixture
def env(tmp_path):
    """A file-backed SQLite engine whose sessions can run on several threads."""
    engine = threaded_sqlite_engine(tmp_path / "mandate.db")
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def get_session():
        """Yield a session on the shared engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    yield SimpleNamespace(
        get_session=get_session,
        persistence=SimpleNamespace(
            saver=MemorySaver(), store=InMemoryStore(), pool=None
        ),
    )
    engine.dispose()


def _seed(env, **changes):
    mandate = make_mandate().model_copy(update=changes)
    with env.get_session() as session:
        MandateRepository(session).upsert(mandate)
        session.commit()
    return mandate


def _stored(env):
    with env.get_session() as session:
        stored = observe.get_mandate(session, PORTFOLIO_ID)
    assert stored is not None
    return stored


def _edit_rows(env):
    with env.get_session() as session:
        return HumanAuditLogRepository(session).list_recent(action="mandate_edit")


def test_a_save_based_on_a_stale_read_is_refused(env) -> None:
    """Another writer's change survives; the stale save is not applied or audited."""
    base = _seed(env)
    _seed(env, remembered_directives=("no EM",))
    stale_save = base.model_copy(update={"capital": Decimal("5000")})

    with env.get_session() as session, pytest.raises(ValueError, match="changed since"):
        apply_action(
            "mandate_edit",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=stale_save,
            prev_value=base.model_dump(mode="json"),
        )

    stored = _stored(env)
    assert stored.remembered_directives == ("no EM",)
    assert stored.capital == base.capital
    assert _edit_rows(env) == []


def test_the_refusal_is_a_typed_conflict(env) -> None:
    """Callers can tell a conflict from a validation error."""
    from fund.audit.mandate_repository import MandateConflictError

    base = _seed(env)
    _seed(env, drift_l1_threshold=0.3)

    with env.get_session() as session, pytest.raises(MandateConflictError):
        MandateRepository(session).upsert(
            base.model_copy(update={"capital": Decimal("1")}),
            expected=base.model_dump(mode="json"),
        )


def test_a_save_based_on_the_current_mandate_goes_through(env) -> None:
    """The compare must not turn a normal save into a conflict."""
    base = _seed(env)
    with env.get_session() as session:
        current = observe.get_mandate(session, PORTFOLIO_ID)
        apply_action(
            "mandate_edit",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=base.model_copy(update={"capital": Decimal("5000")}),
            prev_value=current.model_dump(mode="json"),
        )
        session.commit()

    assert _stored(env).capital == Decimal("5000")


def test_a_first_save_has_nothing_to_conflict_with(env) -> None:
    """A mandate that does not exist yet is created even when a base is passed."""
    mandate = make_mandate()
    with env.get_session() as session:
        apply_action(
            "mandate_edit",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=mandate,
            prev_value=mandate.model_dump(mode="json"),
        )
        session.commit()

    assert _stored(env).capital == mandate.capital


def test_a_save_without_a_recorded_base_is_not_compared(env) -> None:
    """Callers that pass no base (or a non-mapping audit note) keep working."""
    _seed(env)
    changed = make_mandate().model_copy(update={"capital": Decimal("7")})
    with env.get_session() as session:
        apply_action(
            "mandate_edit",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=changed,
        )
        session.commit()

    assert _stored(env).capital == Decimal("7")


def test_upsert_reads_the_row_with_a_lock(env) -> None:
    """The read-modify-write takes the mandate row lock, queueing saves on Postgres."""
    _seed(env)
    statements = []
    with env.get_session() as session:
        real_execute = session.execute

        def spy(statement, *args, **kwargs):
            """Record every statement before running it."""
            statements.append(statement)
            return real_execute(statement, *args, **kwargs)

        with mock.patch.object(session, "execute", side_effect=spy):
            MandateRepository(session).upsert(make_mandate())

    compiled = [
        str(s.compile(dialect=postgresql.dialect()))
        for s in statements
        if hasattr(s, "compile")
    ]
    assert any(
        "portfolio_mandates" in sql and "FOR UPDATE" in sql for sql in compiled
    ), compiled


def test_a_pin_insert_lost_to_a_concurrent_save_is_retried(env, monkeypatch) -> None:
    """The save that loses the race for a pin row re-syncs instead of failing."""
    from portopt_db.repositories.universe import pinned_instrument_repository as pins

    real = pins.PinnedInstrumentRepository.replace_pins
    calls = []

    def flaky(self, source, tickers):
        """Fail the first call the way a lost insert race does."""
        calls.append(source)
        if len(calls) == 1:
            raise IntegrityError("INSERT INTO pinned_instruments", {}, Exception("dup"))
        return real(self, source, tickers)

    monkeypatch.setattr(pins.PinnedInstrumentRepository, "replace_pins", flaky)
    mandate = make_mandate().model_copy(update={"eligible_tickers": ("XYZ",)})
    with env.get_session() as session:
        apply_action(
            "mandate_edit",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=mandate,
        )
        session.commit()

    assert len(calls) == 2
    with env.get_session() as session:
        tickers = session.scalars(select(PinnedInstrument.ticker)).all()
    assert tickers == ["XYZ"]


def _shell(env) -> ShellApp:
    return ShellApp(
        PORTFOLIO_ID,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=lambda: None,
        poll_interval=30.0,
    )


async def _open_mandate(app, pilot) -> MandateView:
    from textual.widgets import ContentSwitcher

    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-mandate"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-mandate", MandateView)


@pytest.mark.asyncio
async def test_tui_save_does_not_overwrite_a_change_landed_since_the_poll(env) -> None:
    """A directive or CLI edit committed after the view's last poll is not clobbered."""
    _seed(env)
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-drift", Input).value = "0.25"
        # Lands between the poll that fed the view and the adviser's ctrl+s.
        _seed(env, remembered_directives=("no EM",))

        app.action_save_mandate()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()

        assert "changed since" in app.last_status
    stored = _stored(env)
    assert stored.remembered_directives == ("no EM",)
    assert stored.drift_l1_threshold == 0.1


@pytest.mark.asyncio
async def test_tui_runs_one_mandate_save_at_a_time(env, monkeypatch) -> None:
    """A second ctrl+s while a save is in flight is refused, not run beside it."""
    import fund.actions as actions_mod

    _seed(env)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_apply(action, **kwargs):
        """Hold the save open until the test releases it."""
        calls.append(action)
        entered.set()
        release.wait(timeout=2.0)
        return SimpleNamespace(message="mandate updated")

    monkeypatch.setattr(actions_mod, "apply_action", slow_apply)
    app = _shell(env)
    mandate = make_mandate()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        first = app._mandate_worker(PORTFOLIO_ID, mandate, None)
        deadline = time.monotonic() + 3
        while not entered.is_set():
            assert time.monotonic() < deadline
            await pilot.pause(0.02)

        second = app._mandate_worker(PORTFOLIO_ID, mandate, None)
        await second.wait()
        refused = app.last_status
        release.set()
        await first.wait()
        await pilot.pause()

    assert calls == ["mandate_edit"]
    assert "in progress" in refused
