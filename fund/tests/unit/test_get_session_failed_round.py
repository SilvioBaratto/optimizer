"""A round that fails under ``get_session`` still leaves its failed run on record.

The CLI and the TUI drive rounds on one shared session and let the error escape the
``with get_session()`` block, so the manager rolls the round back and takes the run row,
its decisions and its error with it. ``get_session`` records the failure afterwards,
in a session of its own, for any exception that carries a failed-round snapshot.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _fund_fakes import ASOF
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_round_failure_persistence import _failing_runtime, _runs
from test_scheduler_steps import PID_A

import fund.database as fund_database
from fund.audit.autonomy_repository import AutonomyRepository
from fund.scheduler import _drive_rebalance


@pytest.fixture
def manager_sessions(tmp_path, monkeypatch):
    """Sessions on a private file-based SQLite engine, served as the shared manager.

    The manager's contract is mirrored: roll back when the block raises, always close.
    Real commits across several connections need a file, not the SAVEPOINT
    ``db_session``; a writer waits one second, not thirty, for the lock.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'round.db'}",
        connect_args={"check_same_thread": False, "timeout": 1},
    )
    Base.metadata.create_all(bind=engine)

    @contextmanager
    def _sessions():
        session = Session(bind=engine)
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def _get_session():
        with _sessions() as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(
        fund_database, "database_manager", SimpleNamespace(get_session=_get_session)
    )
    yield _sessions
    engine.dispose()


def _drive_failing_round(runtime) -> None:
    with fund_database.get_session() as session:
        _drive_rebalance(
            PID_A,
            ASOF,
            session=session,
            persistence=runtime.persistence,
            model=runtime.model,
            fallback=None,
        )
        session.commit()


def _fail_in_the_block() -> None:
    with fund_database.get_session():
        raise ValueError("not a round")


def test_a_failed_round_is_recorded_after_get_session_rolls_it_back(manager_sessions):
    """The run row, its status and its error survive the manager's rollback."""
    runtime, _ = _failing_runtime(manager_sessions)

    with pytest.raises(Exception, match="timed out"):
        _drive_failing_round(runtime)

    (run,) = _runs(manager_sessions)
    assert run.status == "failed"
    assert "TimeoutError" in run.error


def test_a_failed_round_hands_back_the_first_round_gate(manager_sessions):
    """The one-time gate the failed round spent is armed again for the next round."""
    runtime, _ = _failing_runtime(manager_sessions)

    with pytest.raises(Exception, match="timed out"):
        _drive_failing_round(runtime)

    with manager_sessions() as session:
        assert AutonomyRepository(session).portfolio_state(PID_A).first_round_gated


def test_an_unrelated_error_records_nothing(manager_sessions):
    """An exception without a failed-round snapshot is re-raised and writes no run."""
    with pytest.raises(ValueError, match="not a round"):
        _fail_in_the_block()

    assert _runs(manager_sessions) == []
