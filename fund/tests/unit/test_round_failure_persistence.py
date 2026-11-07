"""A round's bookkeeping must not sit in, or be lost with, the round's transaction.

The first-round gate is a row in ``fund_autonomy_state``: spending it inside the round's
transaction holds that row for the whole LLM round, so an operator's freeze waits on it
(and fails after the lock timeout) during the round meant to have the most oversight.
A round that fails is rolled back by its caller; its failure and its orphan checkpoint
thread must outlive that.

These tests need real commits and several connections, so they use a private
file-based SQLite engine rather than the SAVEPOINT ``db_session``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _fund_fakes import (
    ASOF,
    UNIVERSE,
    AllocatorTimesOutFundModel,
    make_constraint_set,
    seed_panel,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision, AgentRun, Base, FundAutonomyState
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from test_scheduler_steps import PID_A, _seed_mandate

from fund.agents import graph
from fund.audit.autonomy_repository import AutonomyRepository
from fund.audit.mifid_repository import MifidProfileRepository, put_constraint_set
from fund.config import settings
from fund.scheduler import (
    SchedulerRuntime,
    _autonomy_blocks_round,
    _in_discard_cooldown,
    run_drift_monitor,
    run_rebalance_sweep,
)

_PID = uuid.UUID("dddddddd-eeee-4fff-8000-111111111111")


@pytest.fixture
def session_factory(tmp_path):
    """File-based SQLite sessions; a writer waits a second, not 30, for the lock.

    SQLite admits one writer at a time, so a bookkeeping write made while the round's
    transaction is still open waits for it; PostgreSQL takes row locks and does not.
    """
    eng = create_engine(
        f"sqlite:///{tmp_path / 'sched.db'}",
        connect_args={"check_same_thread": False, "timeout": 1},
    )
    Base.metadata.create_all(bind=eng)

    @contextmanager
    def _factory():
        session = Session(bind=eng)
        try:
            yield session
        finally:
            session.close()

    yield _factory
    eng.dispose()


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(
        f"sqlite:///{tmp_path / 'round.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    eng.dispose()


def _gated(engine, pid: uuid.UUID) -> bool:
    """Whether the first-round gate is armed, read through a fresh connection."""
    with Session(bind=engine) as reader:
        return AutonomyRepository(reader).portfolio_state(pid).first_round_gated


def test_consuming_the_first_round_gate_commits_on_its_own(engine):
    with Session(bind=engine) as round_session:
        assert graph._consume_first_round(round_session, _PID) is True

        assert _gated(engine, _PID) is False  # visible before the round commits
        round_session.rollback()

    assert _gated(engine, _PID) is False
    with Session(bind=engine) as later:
        assert graph._consume_first_round(later, _PID) is False


def test_restoring_the_first_round_gate_commits_on_its_own(engine):
    with Session(bind=engine) as round_session:
        graph._consume_first_round(round_session, _PID)

        graph._restore_first_round(round_session, _PID)
        round_session.rollback()

    assert _gated(engine, _PID) is True


def test_the_autonomy_check_before_a_round_creates_no_rows(session_factory):
    with session_factory() as session:
        assert _autonomy_blocks_round(session, PID_A) is False
        # Counted in the same transaction: a row the check inserted would sit in it
        # (holding the unique index) until the whole round ends.
        assert (
            session.execute(
                select(func.count()).select_from(FundAutonomyState)
            ).scalar_one()
            == 0
        )


def test_the_autonomy_check_still_sees_a_freeze(session_factory):
    from fund.audit.autonomy_repository import AutonomyScope, AutonomyState

    with session_factory() as session:
        AutonomyRepository(session).set_state(
            AutonomyScope.PORTFOLIO, PID_A, AutonomyState.FROZEN, "test"
        )
        session.commit()
        assert _autonomy_blocks_round(session, PID_A) is True


# --- a failed round survives its caller's rollback -------------------------------


def _failing_runtime(session_factory) -> tuple[SchedulerRuntime, MemorySaver]:
    _seed_mandate(session_factory, PID_A, cron=True, drift=True, threshold=0.1)
    with session_factory() as session:
        seed_panel(session)
        MifidProfileRepository(session).add_version(
            portfolio_id=PID_A,
            questionnaire={},
            constraint_set=make_constraint_set(PID_A).model_dump(mode="json"),
            suitability={},
            store_key=settings.constraint_set_store_key,
        )
        session.commit()
    store = InMemoryStore()
    put_constraint_set(
        store, make_constraint_set(PID_A), store_key=settings.constraint_set_store_key
    )
    saver = MemorySaver()
    model = AllocatorTimesOutFundModel(universe=UNIVERSE, weights={"AAA": 1.0})
    runtime = SchedulerRuntime(
        model=model,
        fallback=None,
        persistence=SimpleNamespace(store=store, saver=saver),
    )
    return runtime, saver


def _runs(factory) -> list[AgentRun]:
    with factory() as session:
        return list(session.query(AgentRun).filter_by(portfolio_id=PID_A).all())


def test_a_failed_scheduled_round_is_recorded_after_the_rollback(session_factory):
    runtime, _ = _failing_runtime(session_factory)

    run_rebalance_sweep(session_factory=session_factory, runtime=runtime, asof=ASOF)

    (run,) = _runs(session_factory)
    assert run.status == "failed"
    assert "TimeoutError" in run.error
    assert run.finished_at is not None
    with session_factory() as session:
        steps = [
            d.hitl_decision
            for d in session.query(AgentDecision).filter_by(run_id=run.id)
            if d.hitl_decision
        ]
    assert {"decision": "failed", "reason": "TimeoutError"} in steps


def test_a_failed_scheduled_round_starts_the_drift_cooldown(session_factory):
    # The cooldown counts `failed` among the outcomes nobody acted on, which only works
    # when the failed run is still there after the rollback.
    runtime, _ = _failing_runtime(session_factory)
    run_rebalance_sweep(session_factory=session_factory, runtime=runtime, asof=ASOF)

    with session_factory() as session:
        assert _in_discard_cooldown(session, PID_A, cooldown_seconds=86400) is True


def test_a_failed_scheduled_round_hands_back_the_first_round_gate(session_factory):
    runtime, _ = _failing_runtime(session_factory)

    run_rebalance_sweep(session_factory=session_factory, runtime=runtime, asof=ASOF)

    with session_factory() as session:
        assert AutonomyRepository(session).portfolio_state(PID_A).first_round_gated


def test_a_failed_round_does_not_leave_its_checkpoint_thread_behind(session_factory):
    runtime, saver = _failing_runtime(session_factory)

    run_rebalance_sweep(session_factory=session_factory, runtime=runtime, asof=ASOF)

    (run,) = _runs(session_factory)
    assert saver.get_tuple({"configurable": {"thread_id": str(run.id)}}) is None


def test_a_failed_drift_round_is_recorded_too(session_factory, monkeypatch):
    import fund.observe as observe

    runtime, _ = _failing_runtime(session_factory)
    monkeypatch.setattr(
        observe,
        "portfolio_state",
        lambda s, pid: SimpleNamespace(
            target_approved=True, drift_l1=0.9, portfolio_id=pid
        ),
    )
    monkeypatch.setattr(
        observe, "reprofile_status", lambda *a, **k: SimpleNamespace(due=False)
    )

    run_drift_monitor(session_factory=session_factory, runtime=runtime, asof=ASOF)

    (run,) = _runs(session_factory)
    assert run.status == "failed"
