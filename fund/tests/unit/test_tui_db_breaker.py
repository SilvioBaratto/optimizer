"""The cockpit greys out its write actions while the database breaker is open.

Validates ``fund.tui._mixins.dbgate.DbGateMixin`` as mixed into ``ShellApp``
(``check_action``, ``_sync_db_gate``), the poll tick that drives it, the Ask
decision route (``AskMixin._confirm_ask_action`` and ``_dispatch_ask_action``),
the attention feed's log (``AttentionMixin._log_attention_failure``), and
``fund.tui.app.main`` handing the shell the breaker of ``fund.database``'s
manager. The breaker is a real ``DbCircuitBreaker`` on a fake clock, opened by
recording connection failures or by reads through a ``DatabaseManager`` whose
engine cannot connect, so no test waits for time to pass.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import suppress
from types import SimpleNamespace

import pytest
from _fund_fakes import PORTFOLIO_ID, make_mandate
from _gate_env import make_gate_env
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager
from portopt_db.resilience import BreakerConfig, BreakerState, DbCircuitBreaker
from sqlalchemy.exc import DisconnectionError

from fund.audit import MandateRepository
from fund.tui import app as tui_app
from fund.tui._mixins.dbgate import DB_WRITE_ACTIONS
from fund.tui.app import ConfirmScreen, RationaleConfirmScreen, ShellApp
from fund.tui.ask import ResolvedAction
from fund.tui.barge_in import BargeInResult, BargeInScreen

_THRESHOLD = 2
_TIMEOUT_S = 30.0
_READ_ONLY_ACTIONS = ("help", "audit_cycle_actor", "export_audit", "quit")


class _Clock:
    """Monotonic fake clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds


@pytest.fixture
def env(tmp_path):
    """A run paused at its gate, with the portfolio's mandate persisted."""
    harness = make_gate_env(tmp_path / "breaker.db", max_orders=2)
    with harness.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    yield harness
    harness.dispose()


@pytest.fixture
def clock() -> _Clock:
    """A fresh fake clock per test."""
    return _Clock()


@pytest.fixture
def breaker(clock: _Clock) -> DbCircuitBreaker:
    """A closed breaker that opens after two connection failures."""
    return DbCircuitBreaker(
        BreakerConfig(failure_threshold=_THRESHOLD, reset_timeout_s=_TIMEOUT_S),
        clock=clock,
    )


def _shell(env, breaker: DbCircuitBreaker | None) -> ShellApp:
    app = ShellApp(
        PORTFOLIO_ID,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=env.model_factory,
        poll_interval=30.0,
    )
    app.watch_db_breaker(breaker)
    return app


def _open(breaker: DbCircuitBreaker) -> None:
    for _ in range(_THRESHOLD):
        breaker.record_failure(DisconnectionError("server closed the connection"))


@pytest.mark.asyncio
async def test_write_actions_are_disabled_and_explained_while_open(
    env, breaker: DbCircuitBreaker
) -> None:
    """Every write binding greys out and the status bar says why; reads stay on."""
    app = _shell(env, breaker)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _open(breaker)

        app._sync_db_gate()
        await pilot.pause()

        assert {app.check_action(a, ()) for a in DB_WRITE_ACTIONS} == {None}
        assert {app.check_action(a, ()) for a in _READ_ONLY_ACTIONS} == {True}
        assert app.last_status.startswith("database unavailable: ")
        assert "write actions are disabled" in app.last_status


@pytest.mark.asyncio
async def test_a_disabled_write_action_does_not_run(
    env, breaker: DbCircuitBreaker, monkeypatch
) -> None:
    """A key that starts a write is swallowed: nothing reaches ``apply_action``."""
    import fund.actions as actions_mod

    calls: list[str] = []
    monkeypatch.setattr(
        actions_mod, "apply_action", lambda action, **_kw: calls.append(action)
    )
    app = _shell(env, breaker)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _open(breaker)
        app._sync_db_gate()

        ran = await app.run_action("run_portfolio")
        await pilot.pause()

        assert ran is False
        assert calls == []
        assert app.last_worker is None


@pytest.mark.asyncio
async def test_the_poll_tick_notices_the_breaker_opening(
    env, breaker: DbCircuitBreaker
) -> None:
    """No caller has to ask: the regular poll greys the actions out."""
    app = _shell(env, breaker)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _open(breaker)

        app._poll()
        await pilot.pause()

        assert app.check_action("decide_approve", ()) is None
        assert app.last_status.startswith("database unavailable: ")


@pytest.mark.asyncio
async def test_a_breaker_waiting_for_its_probe_still_disables_writes(
    env, breaker: DbCircuitBreaker, clock: _Clock
) -> None:
    """A passed timeout is not a recovery: only a successful probe re-enables."""
    app = _shell(env, breaker)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _open(breaker)
        app._sync_db_gate()

        clock.advance(_TIMEOUT_S)
        app._sync_db_gate()
        await pilot.pause()

        assert app.check_action("save_mandate", ()) is None
        assert "database unavailable" in (app._db_refusal_text() or "")


@pytest.mark.asyncio
async def test_writes_come_back_once_a_probe_reaches_the_database(
    env, breaker: DbCircuitBreaker, clock: _Clock
) -> None:
    """A successful probe closes the breaker and the next sync re-enables writes."""
    app = _shell(env, breaker)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _open(breaker)
        app._sync_db_gate()
        clock.advance(_TIMEOUT_S)
        ticket = breaker.before_call()
        breaker.record_success(ticket)

        app._sync_db_gate()
        await pilot.pause()

        assert {app.check_action(a, ()) for a in DB_WRITE_ACTIONS} == {True}
        assert "reachable again" in app.last_status


@pytest.mark.asyncio
async def test_without_a_breaker_nothing_is_gated(env) -> None:
    """A shell built without a breaker keeps every action and the status bar."""
    app = _shell(env, None)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._set_status("ready")

        app._sync_db_gate()

        assert {app.check_action(a, ()) for a in DB_WRITE_ACTIONS} == {True}
        assert app.last_status == "ready"


class _FakePool:
    """Counts ``close`` calls on the persistence pool ``main`` builds."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        """Record that the pool was closed."""
        self.closed += 1


def test_main_hands_the_shell_its_database_breaker(monkeypatch) -> None:
    """The cockpit watches the breaker of the manager its sessions go through."""
    import fund.audit as audit_mod
    import fund.database as db_mod

    persistence = SimpleNamespace(
        saver=MemorySaver(), store=InMemoryStore(), pool=_FakePool()
    )
    seen: list[object] = []
    monkeypatch.setattr(audit_mod, "setup_langgraph", lambda *a, **k: persistence)
    monkeypatch.setattr(ShellApp, "run", lambda self: seen.append(self._db_breaker))
    monkeypatch.setattr(sys, "argv", ["fund-tui"])

    with suppress(SystemExit):
        tui_app.main()

    assert seen == [db_mod.database_manager.breaker]


class _SqliteConfig(DbConfig):
    """SQLite config: the psycopg2 connect args would be rejected."""

    def connect_args(self) -> dict:
        """Return no driver arguments, since sqlite3 accepts none of psycopg2's."""
        return {}


def _manager_on(path, breaker: DbCircuitBreaker) -> DatabaseManager:
    return DatabaseManager(_SqliteConfig(url=f"sqlite:///{path}"), breaker=breaker)


def _shell_on(env, manager: DatabaseManager) -> ShellApp:
    app = ShellApp(
        PORTFOLIO_ID,
        session_factory=manager.get_session,
        persistence=env.persistence,
        model_factory=env.model_factory,
        poll_interval=30.0,
    )
    app.watch_db_breaker(manager.breaker)
    return app


@pytest.mark.asyncio
async def test_a_failing_engine_greys_out_writes_until_a_probe_reaches_it(
    env, tmp_path, breaker: DbCircuitBreaker, clock: _Clock
) -> None:
    """Poll reads through a dead engine open the breaker; a probe that connects
    to the engine once it is back brings the writes back."""
    folder = tmp_path / "appears-later"
    manager = _manager_on(folder / "fund.db", breaker)
    app = _shell_on(env, manager)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._poll()
            app._poll()
            await pilot.pause()

            assert breaker.state is BreakerState.OPEN
            assert app.check_action("decide_approve", ()) is None
            assert app.last_status.startswith("database unavailable: ")

            folder.mkdir()
            clock.advance(_TIMEOUT_S)
            app._poll()
            app._sync_db_gate()
            await pilot.pause()

            assert breaker.state is BreakerState.CLOSED
            assert {app.check_action(a, ()) for a in DB_WRITE_ACTIONS} == {True}
            assert "reachable again" in app.last_status
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_an_ask_action_is_refused_before_its_confirm_while_unavailable(
    env, breaker: DbCircuitBreaker
) -> None:
    """Ask's decision verbs meet the same gate as the keys: no confirm is offered."""
    log: list[str] = []
    view = SimpleNamespace(append_log=lambda _who, text: log.append(text))
    app = _shell(env, breaker)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _open(breaker)
        app._sync_db_gate()

        app._confirm_ask_action(
            view, ResolvedAction("approve", uuid.uuid4(), PORTFOLIO_ID, None)
        )
        await pilot.pause()

        assert not isinstance(app.screen, ConfirmScreen)
        assert log and log[-1].startswith("database unavailable: ")
        assert app.last_status == log[-1]


@pytest.mark.asyncio
async def test_a_confirmed_ask_approve_is_refused_while_unavailable(
    env, breaker: DbCircuitBreaker, monkeypatch
) -> None:
    """A confirm answered after the breaker opened sends nothing to ``apply_action``."""
    import fund.actions as actions_mod

    calls: list[str] = []
    monkeypatch.setattr(
        actions_mod, "apply_action", lambda action, **_kw: calls.append(action)
    )
    log: list[str] = []
    view = SimpleNamespace(append_log=lambda _who, text: log.append(text))
    app = _shell(env, breaker)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _open(breaker)
        app._sync_db_gate()

        worker = app._dispatch_ask_action(
            view, ResolvedAction("approve", uuid.uuid4(), PORTFOLIO_ID, None)
        )
        await pilot.pause()

        assert worker is None
        assert calls == []
        assert log[-1].startswith("database unavailable: ")


def test_a_refused_attention_read_logs_one_line_per_outage(
    env, tmp_path, breaker: DbCircuitBreaker, caplog: pytest.LogCaptureFixture
) -> None:
    """Each poll tick meets the open breaker; the log gets one line, no stack."""
    manager = _manager_on(tmp_path / "fund.db", breaker)
    _open(breaker)
    app = _shell_on(env, manager)

    with caplog.at_level(logging.WARNING, logger="fund.tui.app"):
        reads = [app._read_attention() for _ in range(3)]

    records = [r for r in caplog.records if r.name == "fund.tui.app"]
    assert len(records) == 1
    assert "database unavailable" in records[0].getMessage()
    assert records[0].exc_info is None
    assert all(read.degraded for read in reads)


@pytest.mark.asyncio
async def test_a_fund_rearm_rationale_answered_after_the_breaker_opened_sends_nothing(
    env, breaker: DbCircuitBreaker, monkeypatch
) -> None:
    """The re-arm ceremony dispatches from its own overlay, so it checks the gate."""
    started: list[tuple] = []
    log: list[str] = []
    view = SimpleNamespace(append_log=lambda _who, text: log.append(text))
    app = _shell(env, breaker)
    monkeypatch.setattr(app, "_rearm_worker", lambda *args: started.append(args))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._confirm_ask_action(view, ResolvedAction("rearm", None, None, "fund"))
        await pilot.pause()
        assert isinstance(app.screen, RationaleConfirmScreen)
        _open(breaker)
        app._sync_db_gate()

        app.screen.dismiss("the outage is over")
        await pilot.pause()

        assert started == []
        assert log[-1].startswith("database unavailable: ")
        assert app.last_status == log[-1]


@pytest.mark.asyncio
async def test_a_steer_submitted_after_the_breaker_opened_sends_nothing(
    env, breaker: DbCircuitBreaker, monkeypatch
) -> None:
    """The steer overlay dispatches from its own callback, so it checks the gate too."""
    started: list[tuple] = []
    log: list[str] = []
    view = SimpleNamespace(append_log=lambda _who, text: log.append(text))
    app = _shell(env, breaker)
    monkeypatch.setattr(app, "_barge_in_worker", lambda *args: started.append(args))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._confirm_ask_action(
            view, ResolvedAction("respond", uuid.uuid4(), PORTFOLIO_ID, None)
        )
        await pilot.pause()
        assert isinstance(app.screen, BargeInScreen)
        _open(breaker)
        app._sync_db_gate()

        app.screen.dismiss(BargeInResult(text="hold cash", remember=False))
        await pilot.pause()

        assert started == []
        assert log[-1].startswith("database unavailable: ")
