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
    make_mandate,
    seed_paused_fund_run,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fund import observe
from fund.actions import ActionResult, apply_action, is_within_suitability_band
from fund.audit import AgentRunRepository, HumanAuditLogRepository, MandateRepository


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


@pytest.fixture
def clean_db():
    """A private function-scoped SQLite session where committed rows don't leak.

    The shared ``db_session`` fixture is a SAVEPOINT over a session-scoped engine, so a
    ``commit`` inside a test bleeds into sibling tests. The mandate/guardrail actions
    commit, so they open their own throwaway engine instead.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = session_local()
    try:
        yield session
    finally:
        session.close()
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


# --- T5: mandate_edit (run-policy, persists) + guardrail_override (audit-only) ---


def test_mandate_edit_persists_via_upsert_and_logs(clean_db) -> None:
    """A run-policy edit upserts the mandate (reload shows it) and logs, unflagged."""
    original = make_mandate()
    proposed = original.model_copy(update={"drift_l1_threshold": 0.25})

    result = apply_action(
        "mandate_edit",
        session=clean_db,
        portfolio_id=PORTFOLIO_ID,
        mandate=proposed,
        prev_value=original.model_dump(mode="json"),
    )
    clean_db.commit()

    assert result.action == "mandate_edit"
    assert result.portfolio_id == PORTFOLIO_ID

    reloaded = observe.get_mandate(clean_db, PORTFOLIO_ID)
    assert reloaded is not None
    assert reloaded.drift_l1_threshold == 0.25  # upsert persisted the new value

    rows = HumanAuditLogRepository(clean_db).list_recent(action="mandate_edit")
    assert len(rows) == 1
    assert rows[0].id == result.audit_id
    assert rows[0].portfolio_id == PORTFOLIO_ID
    assert rows[0].override_flag is False
    assert rows[0].suitability_flag is False
    assert rows[0].new_value["drift_l1_threshold"] == 0.25


def test_mandate_edit_requires_a_mandate(clean_db) -> None:
    """``mandate_edit`` without a mandate payload raises before touching the DB."""
    with pytest.raises(ValueError, match="requires a mandate"):
        apply_action("mandate_edit", session=clean_db, portfolio_id=PORTFOLIO_ID)
    assert MandateRepository(clean_db).get(PORTFOLIO_ID) is None
    assert HumanAuditLogRepository(clean_db).list_recent() == []


def test_guardrail_override_logs_audit_only_with_flags(clean_db) -> None:
    """An out-of-band guardrail edit logs a flagged row and persists no mandate."""
    result = apply_action(
        "guardrail_override",
        session=clean_db,
        portfolio_id=PORTFOLIO_ID,
        target="risk_band",
        prev_value="balanced",
        new_value="aggressive",
        override_flag=True,
        suitability_flag=True,
        source="mifid_profile",
    )
    clean_db.commit()

    assert result.action == "guardrail_override"
    rows = HumanAuditLogRepository(clean_db).list_recent(action="guardrail_override")
    assert len(rows) == 1
    assert rows[0].target == "risk_band"
    assert rows[0].prev_value == "balanced"
    assert rows[0].new_value == "aggressive"
    assert rows[0].override_flag is True
    assert rows[0].suitability_flag is True
    # audit-only this phase: nothing is persisted to the mandate.
    assert MandateRepository(clean_db).get(PORTFOLIO_ID) is None


def test_guardrail_override_within_band_logs_unflagged(clean_db) -> None:
    """An in-band guardrail edit is still audited, but carries no override flags."""
    apply_action(
        "guardrail_override",
        session=clean_db,
        portfolio_id=PORTFOLIO_ID,
        target="risk_band",
        prev_value="balanced",
        new_value="conservative",
    )
    clean_db.commit()

    rows = HumanAuditLogRepository(clean_db).list_recent(action="guardrail_override")
    assert len(rows) == 1
    assert rows[0].override_flag is False
    assert rows[0].suitability_flag is False


# --- T7: drive flows (new_portfolio persists; profile/run are invocation-only) ---


def test_new_portfolio_upserts_initial_mandate_and_logs(clean_db) -> None:
    """``new_portfolio`` creates the portfolio by upserting its mandate + logs it."""
    mandate = make_mandate()

    result = apply_action(
        "new_portfolio",
        session=clean_db,
        portfolio_id=PORTFOLIO_ID,
        mandate=mandate,
    )
    clean_db.commit()

    assert result.action == "new_portfolio"
    assert result.portfolio_id == PORTFOLIO_ID
    # The portfolio now exists: its mandate round-trips, so it joins the roster.
    assert observe.get_mandate(clean_db, PORTFOLIO_ID) is not None
    rows = HumanAuditLogRepository(clean_db).list_recent(action="new_portfolio")
    assert len(rows) == 1
    assert rows[0].id == result.audit_id
    assert rows[0].portfolio_id == PORTFOLIO_ID
    assert rows[0].target == "portfolio"


def test_new_portfolio_requires_a_mandate(clean_db) -> None:
    """``new_portfolio`` without a mandate raises before touching the DB."""
    with pytest.raises(ValueError, match="requires a mandate"):
        apply_action("new_portfolio", session=clean_db, portfolio_id=PORTFOLIO_ID)
    assert HumanAuditLogRepository(clean_db).list_recent() == []


def test_profile_requires_a_portfolio_id(db_session) -> None:
    """``profile`` without a ``portfolio_id`` raises before building the agent."""
    with pytest.raises(ValueError, match="requires a portfolio_id"):
        apply_action("profile", session=db_session)


def test_run_requires_a_portfolio_id_and_a_mandate(db_session) -> None:
    """``run`` guards both its required args before invoking the fund pipeline."""
    with pytest.raises(ValueError, match="requires a portfolio_id"):
        apply_action("run", session=db_session, mandate=make_mandate())
    with pytest.raises(ValueError, match="requires a mandate"):
        apply_action("run", session=db_session, portfolio_id=PORTFOLIO_ID)


def test_is_within_suitability_band_treats_the_assessment_band_as_a_ceiling() -> None:
    """A proposal no riskier than the suitability ceiling is in band; riskier is not."""
    assert is_within_suitability_band("balanced", "balanced") is True
    # de-risking below the ceiling is always suitable; going riskier is out of band.
    assert is_within_suitability_band("conservative", "balanced") is True
    assert is_within_suitability_band("growth", "balanced") is False
    assert is_within_suitability_band("aggressive", "defensive") is False


def test_is_within_suitability_band_flags_unknown_bands() -> None:
    """An unrecognised band resolves toward flagging (the compliance-safe default)."""
    # Unknown ceiling → most-restrictive: only the most-conservative proposal stays in.
    assert is_within_suitability_band("defensive", "garbled") is True
    assert is_within_suitability_band("conservative", "garbled") is False
    # Unknown proposal → most-risky: flags unless the ceiling is already the maximum.
    assert is_within_suitability_band("garbled", "growth") is False
    assert is_within_suitability_band("garbled", "aggressive") is True


def test_apply_action_is_idempotent_on_repeated_key(paused_env) -> None:
    """A repeated keyed action replays the first outcome without re-resuming the run."""
    env = paused_env
    with env.get_session() as session:
        first = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            idempotency_key="run-approve-1",
        )
        session.commit()

    with env.get_session() as session:
        second = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            idempotency_key="run-approve-1",
        )
        session.commit()
        rows = HumanAuditLogRepository(session).list_recent(action="approve")

    # The replay returns the first row verbatim and writes no second audit row — proof
    # the pre-check short-circuited before resume_run ran a second time (a real second
    # resume on the now-completed run would have errored or re-finalised it).
    assert second.audit_id == first.audit_id
    assert "already applied" in second.message
    assert len(rows) == 1
