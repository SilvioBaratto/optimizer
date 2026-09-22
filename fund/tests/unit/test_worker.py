"""T7 — fund daemon entrypoint (``fund.worker``): boot, drain, startup reconcile.

The wiring counterpart to Task 6's ``fund.scheduler`` (the steps). ``worker.py``
builds the lifetime runtime once (model + fallback + LangGraph pool), installs it
via ``configure_runtime``, reconciles orphan slots on startup, starts the
scheduler, then blocks until SIGTERM/SIGINT and drains within a time bound.

Contracts under test (SPEC §Phase-9D / plan Task 7):

* ``_reconcile_orphans`` fails lease-expired fund slots on startup (heartbeat
  lease — NULL/stale ``last_heartbeat_at``), spares fresh ones, and is
  **non-fatal** (a DB hiccup logs and returns rather than aborting the boot).
* ``_install_signal_handlers`` registers SIGTERM **and** SIGINT so either sets the
  module-level ``_shutdown`` event.
* ``_drain_and_shutdown`` pauses the scheduler, then ``shutdown(wait=True)`` inside
  a helper thread bounded by the drain timeout — surviving a ``pause``/``shutdown``
  hiccup, and reporting ``False`` when the deadline elapses first.
* ``main`` boots the full sequence with mocks (no real scheduler/pool/DB/block),
  and its drain path calls ``scheduler.shutdown`` + ``pool.close`` + ``close_db``.

A subprocess guard (fresh interpreter) proves ``import fund.worker`` drags in none
of the agent stack (deepagents/langchain/langgraph) nor the ingestion daemon
(``app``) — the agent stack lives lazily inside ``_build_runtime``.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from portopt_db.models import BackgroundJob, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from fund import worker

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def job_factory() -> Generator[object, None, None]:
    """A fresh in-memory DB per test + a session-context factory.

    Mirrors the T2 ``job_session`` rationale: ``_reconcile_orphans`` commits, so
    its test needs a private engine (a StaticPool ``:memory:`` DB shares one
    connection across sessions on the same engine — fine single-threaded).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    @contextmanager
    def _factory() -> Generator[Session, None, None]:
        session = Session(bind=engine)
        try:
            yield session
        finally:
            session.close()

    try:
        yield _factory
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def _clear_shutdown() -> Generator[None, None, None]:
    """Never let one test's ``_shutdown`` state leak into the next."""
    worker._shutdown.clear()
    yield
    worker._shutdown.clear()


def _insert_slot(factory, *, job_type: str, heartbeat: str) -> None:
    """Seed one active fund slot with a ``fresh`` or ``stale`` heartbeat."""
    hb = (
        datetime.now(UTC)
        if heartbeat == "fresh"
        else datetime.now(UTC) - timedelta(seconds=10_000)
    )
    with factory() as session:
        session.add(
            BackgroundJob(
                job_type=job_type,
                status="running",
                started_at=datetime.now(UTC),
                last_heartbeat_at=hb,
            )
        )
        session.commit()


# --- _reconcile_orphans -----------------------------------------------------


def test_reconcile_orphans_reaps_stale_fund_slot(job_factory) -> None:
    _insert_slot(job_factory, job_type="fund_rebalance_sweep", heartbeat="stale")

    worker._reconcile_orphans(session_factory=job_factory)

    with job_factory() as session:
        rows = session.query(BackgroundJob).all()
    assert [r.status for r in rows] == ["failed"]


def test_reconcile_orphans_spares_fresh_slot(job_factory) -> None:
    _insert_slot(job_factory, job_type="fund_drift_monitor", heartbeat="fresh")

    worker._reconcile_orphans(session_factory=job_factory)

    with job_factory() as session:
        row = session.query(BackgroundJob).one()
    assert row.status == "running"


def test_reconcile_orphans_is_non_fatal_on_db_error() -> None:
    def _boom_factory():
        raise RuntimeError("db down")

    # Must not raise — a startup reconcile failure cannot abort the boot.
    worker._reconcile_orphans(session_factory=_boom_factory)


# --- _install_signal_handlers -----------------------------------------------


def test_sigterm_handler_sets_shutdown(monkeypatch) -> None:
    captured: dict[int, Callable[[int, object], None]] = {}
    monkeypatch.setattr(
        worker.signal, "signal", lambda sig, handler: captured.__setitem__(sig, handler)
    )

    worker._install_signal_handlers()

    assert set(captured) == {signal.SIGTERM, signal.SIGINT}
    assert not worker._shutdown.is_set()
    captured[signal.SIGTERM](signal.SIGTERM, None)
    assert worker._shutdown.is_set()


def test_sigint_handler_sets_shutdown(monkeypatch) -> None:
    captured: dict[int, Callable[[int, object], None]] = {}
    monkeypatch.setattr(
        worker.signal, "signal", lambda sig, handler: captured.__setitem__(sig, handler)
    )

    worker._install_signal_handlers()
    captured[signal.SIGINT](signal.SIGINT, None)

    assert worker._shutdown.is_set()


# --- _drain_and_shutdown ----------------------------------------------------


def test_drain_pauses_then_shuts_down() -> None:
    scheduler = MagicMock()

    drained = worker._drain_and_shutdown(scheduler, drain_timeout_seconds=5)

    assert drained is True
    scheduler.pause.assert_called_once()
    scheduler.shutdown.assert_called_once_with(wait=True)


def test_drain_survives_pause_error() -> None:
    scheduler = MagicMock()
    scheduler.pause.side_effect = RuntimeError("already paused")

    drained = worker._drain_and_shutdown(scheduler, drain_timeout_seconds=5)

    assert drained is True
    scheduler.shutdown.assert_called_once_with(wait=True)


def test_drain_survives_shutdown_error() -> None:
    scheduler = MagicMock()
    scheduler.shutdown.side_effect = RuntimeError("shutdown boom")

    # The helper thread's finally still sets ``drained`` even when shutdown raises.
    drained = worker._drain_and_shutdown(scheduler, drain_timeout_seconds=5)

    assert drained is True


def test_drain_reports_timeout_when_shutdown_overruns() -> None:
    scheduler = MagicMock()
    block = threading.Event()
    scheduler.shutdown.side_effect = lambda wait: block.wait(5)

    drained = worker._drain_and_shutdown(scheduler, drain_timeout_seconds=0.05)
    block.set()  # release the daemon helper thread

    assert drained is False


# --- main -------------------------------------------------------------------


def _install_main_stubs(monkeypatch, *, healthy: bool):
    """Patch every side-effecting boundary so ``main`` runs headless, fast.

    Returns a namespace with the mock scheduler, the fake runtime, and the list
    of objects handed to ``configure_runtime`` (install-then-clear).
    """
    scheduler = MagicMock()
    runtime = SimpleNamespace(persistence=SimpleNamespace(pool=MagicMock()))
    installed: list[object] = []
    calls: list[str] = []

    dm = MagicMock()
    dm.health_check.return_value = healthy

    monkeypatch.setattr(worker, "database_manager", dm)
    monkeypatch.setattr(worker, "init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr(worker, "_build_runtime", lambda: runtime)
    monkeypatch.setattr(worker, "configure_runtime", installed.append)
    monkeypatch.setattr(worker, "_reconcile_orphans", lambda: calls.append("reconcile"))
    monkeypatch.setattr(worker, "create_scheduler", lambda: scheduler)
    monkeypatch.setattr(worker, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(worker._shutdown, "wait", lambda *a, **k: True)
    monkeypatch.setattr(worker, "close_db", lambda: calls.append("close_db"))

    return SimpleNamespace(
        scheduler=scheduler, runtime=runtime, installed=installed, calls=calls
    )


def test_main_boots_starts_and_drains(monkeypatch) -> None:
    stubs = _install_main_stubs(monkeypatch, healthy=True)

    worker.main()

    stubs.scheduler.start.assert_called_once()
    stubs.scheduler.shutdown.assert_called_once_with(wait=True)  # drain ran
    stubs.runtime.persistence.pool.close.assert_called_once()
    assert "init_db" in stubs.calls
    assert "reconcile" in stubs.calls
    assert "close_db" in stubs.calls
    # Runtime installed for the run, then cleared on shutdown.
    assert stubs.installed[0] is stubs.runtime
    assert stubs.installed[-1] is None


def test_main_warns_but_continues_on_failed_health_check(monkeypatch, caplog) -> None:
    stubs = _install_main_stubs(monkeypatch, healthy=False)

    with caplog.at_level(logging.WARNING, logger=worker.logger.name):
        worker.main()

    assert any("health check" in r.message.lower() for r in caplog.records)
    stubs.scheduler.start.assert_called_once()  # boot continued despite the warning


# --- import hygiene ---------------------------------------------------------


def test_import_fund_worker_is_agent_stack_free() -> None:
    # The agent stack + model builders live lazily inside ``_build_runtime``; a
    # bare ``import fund.worker`` must drag none of it (nor the ingestion daemon)
    # into a fresh interpreter's sys.modules. Fresh process ⇒ order-independent.
    code = textwrap.dedent(
        """
        import sys
        import fund.worker  # noqa: F401
        forbidden = {
            "deepagents",
            "langchain",
            "langchain_core",
            "langchain_ollama",
            "langgraph",
            "app",
        }
        leaked = sorted({m.split(".")[0] for m in sys.modules} & forbidden)
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
