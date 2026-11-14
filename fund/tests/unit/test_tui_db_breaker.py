"""The cockpit greys out its write actions while the database breaker is open.

Validates ``fund.tui._mixins.dbgate.DbGateMixin`` as mixed into ``ShellApp``
(``check_action``, ``_sync_db_gate``), the poll tick that drives it, and
``fund.tui.app.main`` handing the shell the breaker of ``fund.database``'s
manager. The breaker is a real ``DbCircuitBreaker`` on a fake clock, opened by
recording connection failures, so no test waits for time to pass.
"""

from __future__ import annotations

import sys
from contextlib import suppress
from types import SimpleNamespace

import pytest
from _fund_fakes import PORTFOLIO_ID, make_mandate
from _gate_env import make_gate_env
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.resilience import BreakerConfig, DbCircuitBreaker
from sqlalchemy.exc import DisconnectionError

from fund.audit import MandateRepository
from fund.tui import app as tui_app
from fund.tui._mixins.dbgate import DB_WRITE_ACTIONS
from fund.tui.app import ShellApp

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
        breaker.before_call()
        breaker.record_success()

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
