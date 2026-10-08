"""``fund.actions.apply_action`` — the unified confirm→apply→audit path.

The single entry point every shell mutation routes through. These tests prove the
Phase-1 slice (``approve`` / ``reject``): each resumes the paused run via
``resume_run`` *and* appends one ``human_audit_log`` row with run/portfolio
provenance, derives the portfolio from the run when the caller omits it, opens no
session of its own (never commits — the caller owns the transaction), and rejects an
unknown action before touching the DB. Driven over the same network-free stack the
cockpit tests use: an isolated in-memory SQLite engine + ``MemorySaver`` +
``InMemoryStore`` + a ``ScriptedFundModel``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
from _fund_fakes import (
    PORTFOLIO_ID,
    UNIVERSE,
    ScriptedFundModel,
    seed_paused_fund_run,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fund.actions import ActionResult, apply_action
from fund.audit import AgentRunRepository, HumanAuditLogRepository


@pytest.fixture
def paused_env():
    """An isolated SQLite engine + in-memory LangGraph handles with a paused run."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one connection => one in-memory DB across sessions
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    saver = MemorySaver()
    store = InMemoryStore()

    @contextmanager
    def get_session():
        """Yield a session over the shared in-memory engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    run_id, expected = seed_paused_fund_run(get_session, saver, store)

    yield SimpleNamespace(
        get_session=get_session,
        saver=saver,
        store=store,
        run_id=run_id,
        expected=expected,
        model_factory=lambda: ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )
    engine.dispose()


def test_approve_resumes_run_and_logs_audit(paused_env) -> None:
    """Approve finalises the run completed and appends a provenance-stamped row."""
    env = paused_env
    with env.get_session() as session:
        result = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()

    assert isinstance(result, ActionResult)
    assert result.action == "approve"
    assert result.run_id == env.run_id
    assert result.portfolio_id == PORTFOLIO_ID
    assert "approved" in result.message

    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "completed"
        assert run.weights == env.expected
        rows = HumanAuditLogRepository(session).list_recent(action="approve")
        assert len(rows) == 1
        assert rows[0].id == result.audit_id
        assert rows[0].run_id == env.run_id
        assert rows[0].portfolio_id == PORTFOLIO_ID
        assert rows[0].actor == "human"
        assert rows[0].override_flag is False
        assert rows[0].suitability_flag is False


def test_reject_resumes_run_and_logs_audit(paused_env) -> None:
    """Reject finalises the run rejected (no weights) and logs a reject row."""
    env = paused_env
    with env.get_session() as session:
        result = apply_action(
            "reject",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()

    assert result.action == "reject"
    assert "rejected" in result.message

    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "rejected"
        assert run.weights == {}
        rows = HumanAuditLogRepository(session).list_recent(action="reject")
        assert len(rows) == 1
        assert rows[0].run_id == env.run_id


def test_killswitch_resumes_as_reject_and_logs_distinct_action(paused_env) -> None:
    """Killswitch finalises the run rejected but logs a distinct killswitch row."""
    env = paused_env
    with env.get_session() as session:
        result = apply_action(
            "killswitch",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()

    assert result.action == "killswitch"
    assert "killswitch" in result.message

    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "rejected"  # finalised through the reject resume path
        assert run.weights == {}
        # Logged distinctly: a killswitch row, never a plain reject row.
        assert HumanAuditLogRepository(session).list_recent(action="reject") == []
        rows = HumanAuditLogRepository(session).list_recent(action="killswitch")
        assert len(rows) == 1
        assert rows[0].id == result.audit_id
        assert rows[0].run_id == env.run_id
        assert rows[0].portfolio_id == PORTFOLIO_ID


def test_portfolio_id_derived_from_run_when_omitted(paused_env) -> None:
    """Omitting ``portfolio_id`` recovers it from the run for the result + audit row."""
    env = paused_env
    with env.get_session() as session:
        result = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()

    assert result.portfolio_id == PORTFOLIO_ID
    with env.get_session() as session:
        rows = HumanAuditLogRepository(session).list_recent()
        assert rows[0].portfolio_id == PORTFOLIO_ID


def test_apply_action_does_not_commit_the_caller_owns_the_transaction(
    paused_env,
) -> None:
    """``apply_action`` never commits — the injected session owns the boundary (D1)."""
    env = paused_env
    with env.get_session() as session, mock.patch.object(session, "commit") as spy:
        apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        spy.assert_not_called()
        session.rollback()


def test_unknown_action_raises_before_touching_db(db_session) -> None:
    """An unrecognised action kind raises ``ValueError`` and logs nothing."""
    with pytest.raises(ValueError, match="unknown action"):
        apply_action("frobnicate", session=db_session, run_id=uuid.uuid4())
    assert HumanAuditLogRepository(db_session).list_recent() == []


def test_resume_action_requires_a_run_id(db_session) -> None:
    """Approve/reject without a ``run_id`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="requires a run_id"):
        apply_action("approve", session=db_session)
