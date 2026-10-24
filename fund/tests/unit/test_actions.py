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

import contextlib
import datetime as dt
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from _fund_fakes import (
    ASOF,
    N_DAYS,
    PORTFOLIO_ID,
    START,
    UNIVERSE,
    ScriptedFundModel,
    _count_tool_calls,
    classify_instruments,
    make_constraint_set,
    make_energy_exclusion_constraint_set,
    make_mandate,
    seed_paused_fund_run,
)
from _profiler_fakes import (
    ScriptedProfilerModel,
    final_reply,
    make_answers,
)
from _profiler_fakes import make_model as make_profiler_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import Base, FundAutonomyState, PaperOrder
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from pydantic import Field
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fund import observe
from fund.actions import ActionResult, apply_action, is_within_suitability_band
from fund.audit import (
    AgentRunRepository,
    AutonomyRepository,
    AutonomyScope,
    AutonomyState,
    HumanAuditLogRepository,
    MandateRepository,
    MifidProfileRepository,
    PositionRepository,
    put_constraint_set,
)
from fund.audit.orders_repository import weights_hash
from fund.config import settings
from fund.schemas.constraint_set import Bounds, UniverseFilters
from fund.schemas.enums import GicsSector, LossReaction


class _StopRunError(Exception):
    """Raised by a stubbed ``run_fund`` once it has recorded the decision bar."""


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


def _drop_bars_after_asof(env) -> None:
    """Delete every priced bar past the run's decision bar: no fill bar remains."""
    from _fund_fakes import ASOF
    from portopt_db.models.market_data.yfinance_data import PriceHistory

    with env.get_session() as session:
        session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
        session.commit()


@pytest.fixture
def fill_bar_vanishes_after_the_check(monkeypatch):
    """Let the approve's fill-bar dry run pass although ``place_orders`` will fail.

    The dry run is a read taken before the resume, so a bar that goes missing between
    it and the placement (or any other refusal of the placement) still has to end the
    run as ``incomplete`` rather than claim a rebalance.
    """
    from fund.agents import graph as graph_mod
    from fund.tools import orders

    monkeypatch.setattr(graph_mod, "_require_fill_bars", lambda *_a, **_k: None)
    # ``edit`` runs the same dry run before it records anything.
    monkeypatch.setattr(orders, "fill_shortfalls", lambda *_a, **_k: [])


@pytest.mark.parametrize(
    ("action", "error", "message"),
    [
        ("approve", RuntimeError, "paper ticket"),
        # An edit is refused even earlier, before it appends its allocator row.
        ("edit", ValueError, "cannot be ticketed"),
    ],
)
def test_approve_without_a_fill_bar_is_refused_and_leaves_the_gate_open(
    paused_env, action, error, message
) -> None:
    """No close after ``asof`` yet: the approve is refused before the graph resumes,
    so the run stays at its live gate and the same action can be retried."""
    env = paused_env
    _drop_bars_after_asof(env)
    edited = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    extra = {"new_value": edited} if action == "edit" else {}

    with (
        env.get_session() as session,
        pytest.raises(error, match=message),
    ):
        apply_action(
            action,
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            **extra,
        )

    with env.get_session() as session:
        row = AgentRunRepository(session).get_run(env.run_id)
        assert row.status == "paused"
        assert observe.interrupt_for(env.saver, row) is not None
        assert session.query(PaperOrder).count() == 0


@pytest.mark.parametrize("action", ["approve", "edit"])
def test_approve_without_a_ticket_does_not_report_plain_success(
    paused_env, action, fill_bar_vanishes_after_the_check
) -> None:
    """A gate resumed while ``place_orders`` finds no fill bar leaves the run
    incomplete with no ticket; the status line must say so instead of claiming the
    run was approved, while the human's action is still audited."""
    env = paused_env
    _drop_bars_after_asof(env)
    edited = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    extra = {"new_value": edited} if action == "edit" else {}

    with env.get_session() as session:
        result = apply_action(
            action,
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            **extra,
        )
        session.commit()

    assert "no paper ticket" in result.message
    assert "incomplete" in result.message
    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "incomplete"
        assert session.query(PaperOrder).count() == 0
        assert PositionRepository(session).get_holdings(PORTFOLIO_ID) == []
        assert len(HumanAuditLogRepository(session).list_recent(action=action)) == 1


def test_approve_that_regates_on_a_responded_run_is_not_reported_as_plain_success(
    paused_env, fill_bar_vanishes_after_the_check
) -> None:
    """A run an earlier barge-in left ``responded`` whose approve ticketed nothing
    (no fill bar) and re-gated stays ``responded``; the status line must still say no
    paper ticket was placed rather than a bare "approved"."""
    env = paused_env

    def regating() -> ScriptedFundModel:
        return ScriptedFundModel(universe=UNIVERSE, weights=env.expected, max_orders=3)

    with env.get_session() as session:
        apply_action(
            "respond",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=regating(),
            pm_directive="Trim AAA before committing.",
        )
        session.commit()
    _drop_bars_after_asof(env)

    with env.get_session() as session:
        result = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=regating(),
        )
        session.commit()

    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "responded"
        assert session.query(PaperOrder).count() == 0
    assert "no paper ticket" in result.message
    assert "responded" in result.message


def test_approve_refused_by_the_mandate_reports_that_nothing_was_booked(
    paused_env,
) -> None:
    """An approve the gated tool refuses (re-profiled mandate) is logged as the human's
    decision, but the status line and the run must not claim a booked allocation."""
    env = paused_env
    with env.get_session() as session:
        classify_instruments(
            session, {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
        )
        session.commit()
    put_constraint_set(
        env.store,
        make_energy_exclusion_constraint_set(),
        store_key=settings.constraint_set_store_key,
    )

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

    assert "no paper ticket" in result.message
    assert "incomplete" in result.message
    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "incomplete"
        assert session.query(PaperOrder).count() == 0
        assert PositionRepository(session).get_holdings(PORTFOLIO_ID) == []
        assert len(HumanAuditLogRepository(session).list_recent(action="approve")) == 1


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


def test_respond_barge_in_regates_and_logs_human_audit_row(paused_env) -> None:
    """Respond threads the directive to the PM, keeps the run alive as ``responded``,
    and — unlike a bare ``resume_run`` call — appends a ``human_audit_log`` row.

    The barge-in is only auditable because it funnels through ``apply_action``; a
    ``respond`` driven directly against the graph writes only the ``agent_runs``
    decision, so this is the path that keeps the human trail complete.
    """
    env = paused_env
    directive = "Trim AAA before committing."
    with env.get_session() as session:
        result = apply_action(
            "respond",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            # A re-gate model drives a second place_orders call after the respond.
            model=ScriptedFundModel(
                universe=UNIVERSE, weights=env.expected, max_orders=2
            ),
            pm_directive=directive,
        )
        session.commit()

    assert result.action == "respond"
    assert "responded" in result.message

    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "responded"  # alive, still awaiting approval
        rows = HumanAuditLogRepository(session).list_recent(action="respond")
        assert len(rows) == 1
        assert rows[0].id == result.audit_id
        assert rows[0].run_id == env.run_id
        assert rows[0].portfolio_id == PORTFOLIO_ID


def test_respond_that_finalises_the_run_incomplete_says_so(paused_env) -> None:
    """A PM that does not re-gate after a respond leaves nothing to approve.

    The status line must report the finalised run, not imply it is back at a gate.
    """
    env = paused_env
    with env.get_session() as session:
        result = apply_action(
            "respond",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            # The default model orders once and never re-gates after the respond.
            model=ScriptedFundModel(universe=UNIVERSE, weights=env.expected),
            pm_directive="Looks fine, proceed.",
        )
        session.commit()
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "incomplete"

    assert "incomplete" in result.message
    assert "responded" not in result.message
    assert "awaiting approval" not in result.message


def test_respond_requires_a_pm_directive(paused_env) -> None:
    """Respond without a directive is rejected before the agent stack is touched."""
    env = paused_env
    with (
        env.get_session() as session,
        pytest.raises(ValueError, match="pm_directive"),
    ):
        apply_action(
            "respond",
            session=session,
            run_id=env.run_id,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )


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


# --- T4: edit (edited weights committed through the approve path) ----------------


def test_edit_commits_edited_weights_through_the_approve_path(paused_env) -> None:
    """A valid edit resumes via approve; the ticket, weights_hash, finalised run,
    and positions snapshot all reflect the EDITED weights, not the original."""
    env = paused_env
    edited = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    assert edited != env.expected  # the edit must actually change the allocation

    with env.get_session() as session:
        result = apply_action(
            "edit",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            new_value=edited,
        )
        session.commit()

    assert result.action == "edit"
    assert result.run_id == env.run_id

    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "completed"
        assert run.weights == edited  # agent_runs.weights derives from the edit

        orders = (
            session.query(PaperOrder)
            .filter(PaperOrder.portfolio_id == PORTFOLIO_ID)
            .all()
        )
        assert len(orders) == 1
        assert orders[0].weights == edited
        assert orders[0].weights_hash == weights_hash(edited)  # +weights_hash

        positions = {
            p.ticker: p.weight
            for p in PositionRepository(session).get_holdings(PORTFOLIO_ID)
        }
        assert positions == edited

        rows = HumanAuditLogRepository(session).list_recent(action="edit")
        assert len(rows) == 1
        assert rows[0].id == result.audit_id
        assert rows[0].override_flag is True  # ⚑ a human override of the allocation
        assert rows[0].new_value == edited


def test_edit_rejects_math_invalid_weights_and_commits_nothing(paused_env) -> None:
    """An edit that breaks the long-only / sum-to-one invariants is rejected with
    its violations and writes nothing — the run stays paused, no ticket is placed."""
    env = paused_env
    invalid = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.1}  # sums to 0.9 — budget breach

    with env.get_session() as session:
        with pytest.raises(ValueError, match="violat"):
            apply_action(
                "edit",
                session=session,
                run_id=env.run_id,
                portfolio_id=PORTFOLIO_ID,
                checkpointer=env.saver,
                store=env.store,
                model=env.model_factory(),
                new_value=invalid,
            )
        session.rollback()

    with env.get_session() as session:
        assert HumanAuditLogRepository(session).list_recent(action="edit") == []
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "paused"  # still at the gate, untouched
        assert session.query(PaperOrder).count() == 0


def _assert_edit_refused_and_run_still_paused(env, edited) -> None:
    """The edit is rejected up front: nothing audited or booked, the gate still live."""
    with env.get_session() as session:
        with pytest.raises(ValueError, match="violat"):
            apply_action(
                "edit",
                session=session,
                run_id=env.run_id,
                portfolio_id=PORTFOLIO_ID,
                checkpointer=env.saver,
                store=env.store,
                model=env.model_factory(),
                new_value=edited,
            )
        session.rollback()

    with env.get_session() as session:
        assert HumanAuditLogRepository(session).list_recent(action="edit") == []
        assert AgentRunRepository(session).get_run(env.run_id).status == "paused"
        assert session.query(PaperOrder).count() == 0
        assert PositionRepository(session).get_holdings(PORTFOLIO_ID) == []


def test_edit_over_the_position_cap_is_rejected_and_commits_nothing(paused_env) -> None:
    """The adviser's edit is held to the same K&E position cap the live gate applies;
    weights that pass the default bounds but exceed the cap never reach the gate."""
    env = paused_env
    capped = make_constraint_set().model_copy(
        update={"universe_filters": UniverseFilters(max_position_cap=0.4)}
    )
    put_constraint_set(env.store, capped, store_key=settings.constraint_set_store_key)

    _assert_edit_refused_and_run_still_paused(env, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})


def test_edit_into_an_esg_excluded_name_is_rejected_and_commits_nothing(
    paused_env,
) -> None:
    """An edit that holds a name in an ESG-excluded sector is refused before the
    resume, exactly as the live ``risk_check`` would fail it."""
    env = paused_env
    with env.get_session() as session:
        classify_instruments(
            session, {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
        )
        session.commit()
    put_constraint_set(
        env.store,
        make_energy_exclusion_constraint_set(),
        store_key=settings.constraint_set_store_key,
    )

    _assert_edit_refused_and_run_still_paused(env, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})


def test_edit_requires_a_run_id(db_session) -> None:
    """Edit without a ``run_id`` raises before touching the DB."""
    with pytest.raises(ValueError, match="requires a run_id"):
        apply_action("edit", session=db_session, new_value={"AAA": 1.0})
    assert HumanAuditLogRepository(db_session).list_recent() == []


def test_edit_requires_new_value_weights(paused_env) -> None:
    """Edit without edited weights raises before resuming the run."""
    env = paused_env
    with env.get_session() as session:
        with pytest.raises(ValueError, match="requires"):
            apply_action(
                "edit",
                session=session,
                run_id=env.run_id,
                checkpointer=env.saver,
                store=env.store,
                model=env.model_factory(),
            )
        session.rollback()


def test_edit_is_idempotent_on_repeated_key(paused_env) -> None:
    """A repeated keyed edit replays the first outcome without re-resuming the run."""
    env = paused_env
    edited = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}

    with env.get_session() as session:
        first = apply_action(
            "edit",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            new_value=edited,
            idempotency_key="run-edit-1",
        )
        session.commit()

    with env.get_session() as session:
        second = apply_action(
            "edit",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            new_value=edited,
            idempotency_key="run-edit-1",
        )
        session.commit()
        rows = HumanAuditLogRepository(session).list_recent(action="edit")
        orders = (
            session.query(PaperOrder)
            .filter(PaperOrder.portfolio_id == PORTFOLIO_ID)
            .all()
        )

    assert second.audit_id == first.audit_id
    assert "already applied" in second.message
    assert len(rows) == 1  # no second audit row
    assert len(orders) == 1  # no second ticket


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


def test_reprofile_through_actions_supersedes_the_active_profile(clean_db) -> None:
    """A second ``profile`` + ``approve`` must land as the next version.

    Otherwise the adviser sees "approved" while the old, riskier bounds stay active
    and the re-profile flag never clears.
    """
    pid = uuid.UUID("a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d")
    saver, store = MemorySaver(), InMemoryStore()

    def profile_and_approve(answers) -> None:
        profiled = apply_action(
            "profile",
            session=clean_db,
            portfolio_id=pid,
            questionnaire="questionnaire ...",
            model=make_profiler_model(answers, portfolio_id=str(pid)),
            checkpointer=saver,
            store=store,
        )
        apply_action(
            "approve",
            session=clean_db,
            run_id=profiled.run_id,
            checkpointer=saver,
            store=store,
            model=ScriptedProfilerModel(chat_responses=[final_reply(), final_reply()]),
        )
        clean_db.commit()

    balanced = make_answers()
    defensive = make_answers(
        likert=(2, 2, 2), max_loss=0.05, buffer=12.0, reaction=LossReaction.SELL_ALL
    )
    profile_and_approve(balanced)
    first = MifidProfileRepository(clean_db).get_active(pid)
    profile_and_approve(defensive)

    active = MifidProfileRepository(clean_db).get_active(pid)
    assert first.version == 1
    assert active.version == 2
    assert active.constraint_set["a_gamma"] > first.constraint_set["a_gamma"]
    cached = store.get((str(pid),), settings.constraint_set_store_key)
    assert cached.value["a_gamma"] == active.constraint_set["a_gamma"]


def test_profile_that_never_reaches_the_gate_fails_instead_of_awaiting_approval(
    clean_db,
) -> None:
    """A model that answers in prose never opens the gate; nothing can be approved."""
    pid = uuid.UUID("a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d")
    model = ScriptedProfilerModel(
        structured_outcomes=[make_answers()], chat_responses=[final_reply()]
    )

    with pytest.raises(RuntimeError, match="approval gate"):
        apply_action(
            "profile",
            session=clean_db,
            portfolio_id=pid,
            questionnaire="questionnaire ...",
            model=model,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
        )

    assert HumanAuditLogRepository(clean_db).list_recent(action="profile") == []
    (run,) = AgentRunRepository(clean_db).list_runs_for_portfolio(pid)
    assert run.status == "incomplete"


def test_profile_blocked_by_a_suitability_breach_is_committed_not_rolled_back(
    clean_db,
) -> None:
    """A hard MiFID breach is a recorded outcome, so it must survive the commit.

    Callers commit only when ``apply_action`` returns and roll back on any exception,
    so a breach that propagates erases the blocked run, its breach decision and the
    human audit row - the compliance-relevant block leaves no trace.
    """
    pid = uuid.UUID("a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d")
    model = make_profiler_model(
        make_answers(exclusions=tuple(GicsSector)), portfolio_id=str(pid)
    )
    try:
        result = apply_action(
            "profile",
            session=clean_db,
            portfolio_id=pid,
            questionnaire="questionnaire ...",
            model=model,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
        )
    except ValueError:
        clean_db.rollback()
    else:
        clean_db.commit()

    (run,) = AgentRunRepository(clean_db).list_runs_for_portfolio(pid)
    assert run.status == "blocked"
    assert [d.step for d in run.decisions if d.step == "suitability_breach"]
    (audit,) = HumanAuditLogRepository(clean_db).list_recent(action="profile")
    assert audit.run_id == run.id
    assert audit.suitability_flag is True
    assert result.run_id == run.id
    assert result.audit_id == audit.id
    assert "blocked" in result.message
    assert MifidProfileRepository(clean_db).get_active(pid) is None


def test_blocked_onboarding_keeps_only_the_breach_trail(clean_db) -> None:
    """A suitability breach mid-onboarding must not commit the mandate staged before it.

    The TUI worker stages ``new_portfolio`` and ``profile`` on one session; the
    breach commits the blocked run's trail, but the portfolio that was never
    onboarded must roll back with its audit row. The blocked profile is returned as
    a result, so the worker's own commit then lands only the breach's audit row.
    """
    from _profiler_fakes import make_answers, make_model
    from portopt_db.models import AgentRun

    from fund.schemas.enums import GicsSector

    model = make_model(
        make_answers(exclusions=tuple(GicsSector)),  # empties the universe
        portfolio_id=str(PORTFOLIO_ID),
    )
    apply_action(
        "new_portfolio",
        session=clean_db,
        portfolio_id=PORTFOLIO_ID,
        mandate=make_mandate(),
    )

    result = apply_action(
        "profile",
        session=clean_db,
        portfolio_id=PORTFOLIO_ID,
        questionnaire="Everything but nothing, EUR.",
        model=model,
        checkpointer=MemorySaver(),
        store=InMemoryStore(),
    )
    clean_db.commit()  # the worker commits once the action lands

    runs = clean_db.query(AgentRun).filter(AgentRun.portfolio_id == PORTFOLIO_ID).all()
    assert [r.status for r in runs] == ["blocked"]
    assert result.run_id == runs[0].id
    assert observe.get_mandate(clean_db, PORTFOLIO_ID) is None
    audit_rows = HumanAuditLogRepository(clean_db).list_recent()
    assert [row.action for row in audit_rows] == ["profile"]
    assert audit_rows[0].run_id == runs[0].id


def test_run_requires_a_portfolio_id_and_a_mandate(db_session) -> None:
    """``run`` guards both its required args before invoking the fund pipeline."""
    with pytest.raises(ValueError, match="requires a portfolio_id"):
        apply_action("run", session=db_session, mandate=make_mandate())
    with pytest.raises(ValueError, match="requires a mandate"):
        apply_action("run", session=db_session, portfolio_id=PORTFOLIO_ID)


def test_run_without_asof_drives_at_the_default_decision_bar(
    db_session, monkeypatch
) -> None:
    """An unset ``asof`` resolves to a bar with a stored close after it, not today."""
    fillable_bar = dt.date(2024, 2, 8)
    monkeypatch.setattr(observe, "default_decision_bar", lambda session: fillable_bar)
    driven: list[dt.date] = []

    def _record_then_stop(*_args, asof, **_kwargs):
        driven.append(asof)
        raise _StopRunError

    monkeypatch.setattr("fund.agents.graph.run_fund", _record_then_stop)

    with pytest.raises(_StopRunError):
        apply_action(
            "run",
            session=db_session,
            portfolio_id=PORTFOLIO_ID,
            mandate=make_mandate(),
        )

    assert driven == [fillable_bar]


def test_run_reports_a_gateless_incomplete_round_as_incomplete(paused_env) -> None:
    """A round that ends ``incomplete`` (here: the risk check blocked the ticket) has
    no gate to approve, so its message must not claim it is awaiting approval."""
    model = ScriptedFundModel(
        universe=UNIVERSE, weights=paused_env.expected, risk_passes=False
    )
    with paused_env.get_session() as session:
        result = apply_action(
            "run",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=make_mandate(),
            asof=ASOF,
            model=model,
            checkpointer=paused_env.saver,
            store=paused_env.store,
        )
        status = AgentRunRepository(session).get_run(result.run_id).status

    assert status == "incomplete"
    assert "incomplete" in result.message
    assert "awaiting approval" not in result.message


def test_run_reports_a_paused_round_as_awaiting_approval(paused_env) -> None:
    with paused_env.get_session() as session:
        result = apply_action(
            "run",
            session=session,
            portfolio_id=PORTFOLIO_ID,
            mandate=make_mandate(),
            asof=ASOF,
            model=paused_env.model_factory(),
            checkpointer=paused_env.saver,
            store=paused_env.store,
        )

    assert "paused (awaiting approval)" in result.message


@pytest.mark.parametrize(
    ("status", "awaits_approval"),
    [("paused", True), ("incomplete", False), ("rejected", False)],
)
def test_run_message_claims_awaiting_approval_only_for_a_paused_run(
    db_session, status, awaits_approval
) -> None:
    """A run that ended without a gate (round cap, no order, freeze) has nothing to
    approve, so its status line must not say it is awaiting approval."""
    run_id = uuid.uuid4()

    def fake_run_fund(*_args, **_kwargs):
        return SimpleNamespace(run_id=run_id, status=status)

    with mock.patch("fund.agents.graph.run_fund", fake_run_fund):
        result = apply_action(
            "run",
            session=db_session,
            portfolio_id=PORTFOLIO_ID,
            mandate=make_mandate(),
            asof=dt.date(2026, 1, 2),
        )

    assert status in result.message
    assert ("awaiting approval" in result.message) is awaits_approval


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


# --- T6: autonomy write verbs (freeze / rearm) + killswitch scope ----------------


def test_freeze_portfolio_sets_frozen_state_and_logs(clean_db) -> None:
    """A portfolio-scope freeze flips the autonomy row FROZEN and logs an audit row."""
    result = apply_action(
        "freeze",
        session=clean_db,
        scope="portfolio",
        portfolio_id=PORTFOLIO_ID,
        rationale="drawdown breach",
    )
    clean_db.commit()

    assert result.action == "freeze"
    assert result.portfolio_id == PORTFOLIO_ID

    state = AutonomyRepository(clean_db).portfolio_state(PORTFOLIO_ID)
    assert state.state == "frozen"
    assert state.frozen_reason == "drawdown breach"
    assert state.frozen_at is not None

    rows = HumanAuditLogRepository(clean_db).list_recent(action="freeze")
    assert len(rows) == 1
    assert rows[0].id == result.audit_id
    assert rows[0].target == "portfolio"


def test_freeze_fund_sets_frozen_state_and_logs(clean_db) -> None:
    """A fund-scope freeze flips the single fund autonomy row FROZEN and logs."""
    result = apply_action(
        "freeze",
        session=clean_db,
        scope="fund",
        rationale="halt everything",
    )
    clean_db.commit()

    assert result.action == "freeze"
    assert result.portfolio_id is None

    state = AutonomyRepository(clean_db).fund_state()
    assert state.state == "frozen"
    assert state.frozen_reason == "halt everything"

    rows = HumanAuditLogRepository(clean_db).list_recent(action="freeze")
    assert len(rows) == 1
    assert rows[0].target == "fund"


def test_freeze_with_an_overlong_rationale_fits_the_column_and_audits_it_in_full(
    clean_db,
) -> None:
    """``frozen_reason`` is a bounded column; a long rationale must not block a freeze.

    Postgres rejects an over-length value (SQLite does not), which would abort the
    killswitch itself. The audit row's rationale is unbounded, so the full text lives
    there.
    """
    rationale = "halt: " + "x" * 600
    limit = FundAutonomyState.__table__.c.frozen_reason.type.length

    apply_action("freeze", session=clean_db, scope="fund", rationale=rationale)
    clean_db.commit()

    state = AutonomyRepository(clean_db).fund_state()
    assert state.state == "frozen"
    assert len(state.frozen_reason) <= limit
    assert state.frozen_reason.startswith("halt: xxx")
    (row,) = HumanAuditLogRepository(clean_db).list_recent(action="freeze")
    assert row.rationale == rationale


def test_rearm_portfolio_restores_armed(clean_db) -> None:
    """Re-arming a frozen portfolio restores ARMED and clears the frozen fields."""
    apply_action(
        "freeze",
        session=clean_db,
        scope="portfolio",
        portfolio_id=PORTFOLIO_ID,
        rationale="pause",
    )
    clean_db.commit()

    result = apply_action(
        "rearm",
        session=clean_db,
        scope="portfolio",
        portfolio_id=PORTFOLIO_ID,
    )
    clean_db.commit()

    assert result.action == "rearm"
    state = AutonomyRepository(clean_db).portfolio_state(PORTFOLIO_ID)
    assert state.state == "armed"
    assert state.frozen_reason is None
    assert state.frozen_at is None


def test_portfolio_freeze_and_rearm_messages_name_the_portfolio(clean_db) -> None:
    """The status line says which portfolio changed posture, not just the scope."""
    frozen = apply_action(
        "freeze", session=clean_db, scope="portfolio", portfolio_id=PORTFOLIO_ID
    )
    rearmed = apply_action(
        "rearm", session=clean_db, scope="portfolio", portfolio_id=PORTFOLIO_ID
    )
    fund = apply_action("freeze", session=clean_db, scope="fund")

    assert frozen.message == f"portfolio {PORTFOLIO_ID} frozen"
    assert rearmed.message == f"portfolio {PORTFOLIO_ID} re-armed"
    assert fund.message == "fund frozen"


def test_rearm_fund_requires_a_rationale(clean_db) -> None:
    """A fund-scope re-arm without a rationale is rejected before any write."""
    with pytest.raises(ValueError, match="rationale"):
        apply_action("rearm", session=clean_db, scope="fund")
    assert HumanAuditLogRepository(clean_db).list_recent() == []


def test_rearm_fund_records_the_rationale(clean_db) -> None:
    """A fund-scope re-arm with a rationale restores ARMED and records the reason."""
    apply_action("freeze", session=clean_db, scope="fund", rationale="halt")
    clean_db.commit()

    result = apply_action(
        "rearm",
        session=clean_db,
        scope="fund",
        rationale="risk cleared; resuming",
    )
    clean_db.commit()

    assert AutonomyRepository(clean_db).fund_state().state == "armed"
    rows = HumanAuditLogRepository(clean_db).list_recent(action="rearm")
    assert len(rows) == 1
    assert rows[0].id == result.audit_id
    assert rows[0].rationale == "risk cleared; resuming"


def test_freeze_run_scope_rejects_run_without_autonomy_change(paused_env) -> None:
    """A run-scope freeze keeps killswitch semantics: reject, posture stays ARMED."""
    env = paused_env
    with env.get_session() as session:
        result = apply_action(
            "freeze",
            session=session,
            scope="run",
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()

    assert result.action == "freeze"
    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "rejected"
        # A run-scope freeze flips no autonomy posture: the rows the run seeded stay
        # ARMED (only a portfolio/fund freeze writes FROZEN).
        autonomy = AutonomyRepository(session)
        assert autonomy.fund_state().state == "armed"
        assert autonomy.portfolio_state(PORTFOLIO_ID).state == "armed"


def test_approve_is_refused_when_the_fund_froze_after_the_gate_opened(
    paused_env,
) -> None:
    """A gate that reached the queue after the freeze cannot ship a ticket."""
    env = paused_env
    with env.get_session() as session:
        AutonomyRepository(session).set_state(
            AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="killswitch"
        )
        session.commit()

    with env.get_session() as session:
        with pytest.raises(RuntimeError, match="frozen"):
            apply_action(
                "approve",
                session=session,
                run_id=env.run_id,
                portfolio_id=PORTFOLIO_ID,
                checkpointer=env.saver,
                store=env.store,
                model=env.model_factory(),
            )
        session.rollback()

    with env.get_session() as session:
        assert session.query(PaperOrder).count() == 0
        assert PositionRepository(session).get_holdings(PORTFOLIO_ID) == []
        assert HumanAuditLogRepository(session).list_recent(action="approve") == []


def test_freeze_does_not_rewrite_a_run_an_approve_completed_first(
    paused_env, monkeypatch
) -> None:
    """A portfolio freeze lists the live gates, then rejects each. An approve that
    finalises one of them in between wins: the freeze must leave that run ``completed``
    instead of rewriting it to ``rejected`` over a ticket and a positions snapshot."""
    env = paused_env
    with env.get_session() as session:
        repo = AgentRunRepository(session)
        stale_listing = repo.list_paused_runs()
        assert [run.id for run in stale_listing] == [env.run_id]
        repo.finalize_run(env.run_id, weights=env.expected, status="completed")
        monkeypatch.setattr(
            AgentRunRepository,
            "list_paused_runs",
            lambda self, portfolio_id=None: stale_listing,
        )

        apply_action(
            "freeze",
            session=session,
            scope="portfolio",
            portfolio_id=PORTFOLIO_ID,
            rationale="halt",
        )
        session.commit()

    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "completed"
        assert run.weights == env.expected
        state = AutonomyRepository(session).portfolio_state(PORTFOLIO_ID)
        assert state.state == "frozen"


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


# --- freeze vs. a round in flight -------------------------------------------------


def _freeze_portfolio(session, *, scope: str) -> None:
    """Flip ``scope``'s autonomy row FROZEN without halting any run (the race)."""
    from fund.audit.autonomy_repository import AutonomyScope, AutonomyState

    AutonomyRepository(session).set_state(
        AutonomyScope(scope),
        PORTFOLIO_ID if scope == "portfolio" else None,
        AutonomyState.FROZEN,
        reason="drawdown breach",
    )


@pytest.mark.parametrize("scope", ["portfolio", "fund"])
def test_run_rejects_the_gate_when_its_scope_froze_mid_round(
    clean_db, monkeypatch, scope
) -> None:
    """A freeze that lands while a round runs must not be outrun by that round's gate.

    The freeze rejects committed paused gates only; an in-flight round's run row is not
    visible to it, so the round re-checks autonomy when it reaches the gate.
    """

    def _round_frozen_midway(model, mandate, *, portfolio_id, asof, session, **_kwargs):
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=portfolio_id,
            asof=asof,
            seed=1,
            universe=[],
            optimizer_config={},
        )
        _freeze_portfolio(session, scope=scope)  # the adviser freezes mid-round
        repo.mark_paused(run.id)
        return SimpleNamespace(run_id=run.id, status="paused")

    monkeypatch.setattr("fund.agents.graph.run_fund", _round_frozen_midway)

    result = apply_action(
        "run",
        session=clean_db,
        portfolio_id=PORTFOLIO_ID,
        mandate=make_mandate(),
        asof=ASOF,
    )
    clean_db.commit()

    run = AgentRunRepository(clean_db).get_run(result.run_id)
    assert run.status == "rejected"  # halted, not left as an approvable gate
    assert AgentRunRepository(clean_db).list_paused_runs(PORTFOLIO_ID) == []
    assert "frozen" in result.message


@pytest.mark.parametrize("scope", ["portfolio", "fund"])
def test_approve_refuses_a_run_whose_scope_is_frozen(paused_env, scope) -> None:
    """An approve must never place orders while its fund or portfolio is FROZEN."""
    env = paused_env
    with env.get_session() as session:
        _freeze_portfolio(session, scope=scope)  # a gate that outlived the freeze
        session.commit()

    with env.get_session() as session:
        with pytest.raises(RuntimeError, match="frozen"):
            apply_action(
                "approve",
                session=session,
                run_id=env.run_id,
                portfolio_id=PORTFOLIO_ID,
                checkpointer=env.saver,
                store=env.store,
                model=env.model_factory(),
            )
        session.rollback()

    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "paused"
        assert session.query(PaperOrder).count() == 0


# --- edit must only commit weights that resolve to a placeable, in-universe book ---


def _edit(env, session, new_value) -> ActionResult:
    return apply_action(
        "edit",
        session=session,
        run_id=env.run_id,
        portfolio_id=PORTFOLIO_ID,
        checkpointer=env.saver,
        store=env.store,
        model=env.model_factory(),
        new_value=new_value,
    )


def _seed_off_universe_instrument(session, ticker: str) -> None:
    """A priced instrument the allocator never proposed (fillable, but off-universe)."""
    exchange = Exchange(name=f"EX-{ticker}")
    session.add(exchange)
    session.flush()
    instrument = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange_id=exchange.id,
        instrument_type="EQUITY",
        asset_class="equity",
        yfinance_ticker=ticker,
    )
    session.add(instrument)
    session.flush()
    for day in range(N_DAYS):
        session.add(
            PriceHistory(
                instrument_id=instrument.id,
                date=START + dt.timedelta(days=day),
                close=10.0 + day,
                volume=1000,
            )
        )
    session.commit()


def test_edit_rejects_an_unknown_ticker_before_writing_anything(paused_env) -> None:
    """A typo'd ticker sums to 1 and passes the bounds check, so the edit has to be
    refused up front — before it is logged as the run's newest allocation."""
    env = paused_env
    with env.get_session() as session:
        with pytest.raises(ValueError, match="ZZZZ"):
            _edit(env, session, {"AAA": 0.5, "BBB": 0.3, "ZZZZ": 0.2})
        # No rollback: the rejected edit must not have touched the open transaction.
        assert AgentRunRepository(session).latest_optimizer_weights(
            env.run_id
        ) == pytest.approx(env.expected)
        assert HumanAuditLogRepository(session).list_recent(action="edit") == []


def test_edit_rejects_a_real_instrument_outside_the_runs_universe(paused_env) -> None:
    """An instrument the allocator never considered bypasses the run's universe
    screen, so it is refused even though it is priced and would fill."""
    env = paused_env
    with env.get_session() as session:
        _seed_off_universe_instrument(session, "DDD")
        with pytest.raises(ValueError, match="DDD"):
            _edit(env, session, {"AAA": 0.5, "BBB": 0.3, "DDD": 0.2})
        session.rollback()

    with env.get_session() as session:
        assert session.query(PaperOrder).count() == 0
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "paused"


def test_edit_normalises_ticker_case_and_whitespace(paused_env) -> None:
    """``' aaa '`` is the run's ``AAA``: the ticket, the snapshot and the audit row all
    carry the canonical ticker rather than a phantom ``aaa`` position."""
    env = paused_env
    edited = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    with env.get_session() as session:
        _edit(env, session, {" aaa ": 0.5, "bbb": 0.3, "CCC": 0.2})
        session.commit()

    with env.get_session() as session:
        order = session.query(PaperOrder).one()
        assert order.weights == edited
        positions = {
            p.ticker: p.weight
            for p in PositionRepository(session).get_holdings(PORTFOLIO_ID)
        }
        assert positions == edited
        row = HumanAuditLogRepository(session).list_recent(action="edit")[0]
        assert row.new_value == edited


def test_edit_rejects_two_spellings_of_one_ticker(paused_env) -> None:
    """``aaa`` and ``AAA`` normalise to one name; the later must not silently win."""
    env = paused_env
    with env.get_session() as session:
        with pytest.raises(ValueError, match="twice"):
            _edit(env, session, {"aaa": 0.2, "AAA": 0.3, "BBB": 0.5})
        session.rollback()


def test_edit_applies_the_portfolio_constraint_set_bounds(paused_env) -> None:
    """The edit is checked against the portfolio's own ConstraintSet, as the pipeline's
    risk gate is — not just the default 0..1 long-only box."""
    env = paused_env
    capped = make_constraint_set().model_copy(
        update={"bounds": Bounds(max_weights=0.4)}
    )
    put_constraint_set(env.store, capped, store_key=settings.constraint_set_store_key)

    with env.get_session() as session:
        with pytest.raises(ValueError, match="max_weight"):
            _edit(env, session, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
        session.rollback()

    with env.get_session() as session:
        assert session.query(PaperOrder).count() == 0
        assert HumanAuditLogRepository(session).list_recent(action="edit") == []


def test_edit_absorbs_the_rounding_of_a_four_decimal_weight_set(paused_env) -> None:
    """The shell edits weights at 4 decimals, so a book that is economically fully
    invested (a third each) sums to 0.9999. That rounding must not bounce the edit;
    the book that is ticketed and audited is the one rescaled to the budget."""
    env = paused_env
    with env.get_session() as session:
        _edit(env, session, {"AAA": 0.3333, "BBB": 0.3333, "CCC": 0.3333})
        session.commit()

    with env.get_session() as session:
        order = session.query(PaperOrder).one()
        assert sum(order.weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert order.weights == pytest.approx(
            {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}
        )
        row = HumanAuditLogRepository(session).list_recent(action="edit")[0]
        assert row.new_value == order.weights


def test_edit_absorbs_rounding_without_pushing_a_capped_name_over_its_cap(
    paused_env,
) -> None:
    """A name sitting exactly at the ConstraintSet cap cannot take its share of the
    rounding shortfall: the residual goes to the names with headroom instead of
    scaling the capped one past its bound and bouncing an economically valid book."""
    env = paused_env
    capped = make_constraint_set().model_copy(
        update={"bounds": Bounds(max_weights=0.5)}
    )
    put_constraint_set(env.store, capped, store_key=settings.constraint_set_store_key)

    typed = {"AAA": 0.5, "BBB": 0.25, "CCC": 0.2499}
    with env.get_session() as session:
        _edit(env, session, typed)
        session.commit()

    with env.get_session() as session:
        order = session.query(PaperOrder).one()
        assert order.weights["AAA"] == 0.5
        assert sum(order.weights.values()) == pytest.approx(1.0, abs=1e-12)
        for ticker, weight in typed.items():
            assert abs(order.weights[ticker] - weight) <= 5e-5 + 1e-12


def test_edit_absorbs_rounding_by_at_most_half_a_display_step_per_name(
    paused_env,
) -> None:
    """A big name must not drift by more than the 4-dp rounding the adviser could not
    see: proportional rescaling would move a 0.9 holding by 9e-5."""
    env = paused_env
    typed = {"AAA": 0.9, "BBB": 0.0999, "CCC": 0.0}
    with env.get_session() as session:
        _edit(env, session, typed)
        session.commit()

    with env.get_session() as session:
        order = session.query(PaperOrder).one()
        assert sum(order.weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert order.weights["CCC"] == 0.0
        for ticker, weight in typed.items():
            assert abs(order.weights[ticker] - weight) <= 5e-5 + 1e-12


def test_edit_refuses_a_shortfall_the_bounds_leave_no_headroom_to_absorb(
    paused_env,
) -> None:
    """With the capped name unable to move and one name left to absorb, the shortfall
    exceeds what rounding can explain; the real book is refused, not silently bent."""
    env = paused_env
    capped = make_constraint_set().model_copy(
        update={"bounds": Bounds(max_weights=0.5)}
    )
    put_constraint_set(env.store, capped, store_key=settings.constraint_set_store_key)

    with env.get_session() as session:
        with pytest.raises(ValueError, match="budget"):
            _edit(env, session, {"AAA": 0.5, "BBB": 0.4999})
        session.rollback()

    with env.get_session() as session:
        assert session.query(PaperOrder).count() == 0


def test_edit_still_rejects_a_budget_shortfall_beyond_display_rounding(
    paused_env,
) -> None:
    """Rounding slack is bounded by the 4-dp resolution; a real 0.3% hole is not it."""
    env = paused_env
    with env.get_session() as session:
        with pytest.raises(ValueError, match="budget"):
            _edit(env, session, {"AAA": 0.333, "BBB": 0.333, "CCC": 0.333})
        session.rollback()

    with env.get_session() as session:
        assert session.query(PaperOrder).count() == 0


# --- the idempotency key is claimed before the resume's side effects ---------------


def test_concurrent_submit_of_a_keyed_approve_cannot_resume_the_gate_twice(
    tmp_path,
) -> None:
    """The checkpointer commits on its own connection, so only the audit row's unique
    key can stop a second submit of the same approve from re-driving the gate while
    the first is still inside ``resume_run`` (its audit row not yet committed)."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'actions.db'}",
        connect_args={"timeout": 0},  # a held write lock fails fast instead of waiting
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with session_local() as seed:
        run = AgentRunRepository(seed).create_run(
            portfolio_id=PORTFOLIO_ID,
            asof=ASOF,
            seed=1,
            universe=[],
            optimizer_config={"step": "fund"},
        )
        seed.commit()
        run_id = run.id

    key = f"{run_id}:approve"
    resumes: list[str] = []

    def fake_resume(rid, decision, **_kwargs):
        resumes.append(decision)
        if len(resumes) == 1:
            # The loser of the claim is refused while the winner is mid-resume.
            with session_local() as other, contextlib.suppress(DBAPIError):
                apply_action("approve", session=other, run_id=rid, idempotency_key=key)
        return {}

    with session_local() as first:
        with mock.patch("fund.agents.graph.resume_run", side_effect=fake_resume):
            apply_action("approve", session=first, run_id=run_id, idempotency_key=key)
        first.commit()

    assert resumes == ["approve"]
    with session_local() as check:
        rows = HumanAuditLogRepository(check).list_recent(action="approve")
        assert [r.idempotency_key for r in rows] == [key]
    engine.dispose()


def test_retry_after_a_rolled_back_approve_places_the_lost_ticket(paused_env) -> None:
    """If the approve's transaction is lost after the checkpoint advanced, a retry of
    the same key must not mark the run completed over a ticket that no longer exists:
    it places the ticket again and completes only once one backs the run."""
    env = paused_env
    key = f"{env.run_id}:approve"

    def approve(session) -> ActionResult:
        return apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
            idempotency_key=key,
        )

    with env.get_session() as session:
        approve(session)
        session.rollback()  # the commit never landed; the checkpoint already moved on

    with env.get_session() as session:
        assert session.query(PaperOrder).count() == 0
        approve(session)
        session.commit()

    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "completed"
        (order,) = session.query(PaperOrder).all()
        assert order.weights == env.expected
        assert {
            p.ticker: p.weight
            for p in PositionRepository(session).get_holdings(PORTFOLIO_ID)
        } == env.expected
        rows = HumanAuditLogRepository(session).list_recent(action="approve")
        assert [r.idempotency_key for r in rows] == [key]


# --- a failed narration turn after the ticket landed must not lose the approval ----


class _PostTicketOutageModel(ScriptedFundModel):
    """Scripted PM whose provider goes down on the turn after ``place_orders`` ran."""

    def _pm_turn(self, messages):
        if _count_tool_calls(messages, "place_orders"):
            raise RuntimeError("LLM provider 529 overloaded")
        return super()._pm_turn(messages)


def test_approve_is_finalised_from_the_ticket_when_the_post_ticket_pm_turn_fails(
    paused_env,
) -> None:
    """The gated ``place_orders`` writes the ticket and consumes the checkpoint's
    interrupt before the PM's follow-up model turn. If that turn raises, the approval
    must still land from the placed ticket — not roll back and strand the run."""
    env = paused_env
    with env.get_session() as session:
        result = apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=_PostTicketOutageModel(universe=UNIVERSE, weights=env.expected),
        )
        session.commit()

    assert "approved" in result.message
    with env.get_session() as session:
        run = AgentRunRepository(session).get_run(env.run_id)
        assert run.status == "completed"
        assert run.weights == env.expected
        assert session.query(PaperOrder).count() == 1
        positions = {
            p.ticker: p.weight
            for p in PositionRepository(session).get_holdings(PORTFOLIO_ID)
        }
        assert positions == env.expected
        assert len(HumanAuditLogRepository(session).list_recent(action="approve")) == 1


# --- an approve is bound to the weights it ticketed --------------------------------


def test_approve_audit_row_records_the_weights_that_were_ticketed(paused_env) -> None:
    """The gate's own place_orders args are the PM's proposal; the ticket is placed
    from the audited optimizer weights. The approve row has to name what was traded,
    or the trail cannot show what the adviser signed off."""
    env = paused_env
    with env.get_session() as session:
        apply_action(
            "approve",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()

    with env.get_session() as session:
        row = HumanAuditLogRepository(session).list_recent(action="approve")[0]
        order = session.query(PaperOrder).one()
        assert row.target == "weights"
        assert row.new_value == {
            "weights": pytest.approx(env.expected),
            "weights_hash": order.weights_hash,
        }


def test_reject_audit_row_carries_no_weights(paused_env) -> None:
    """Only an approve commits a ticket, so only it records one."""
    env = paused_env
    with env.get_session() as session:
        apply_action(
            "reject",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=env.model_factory(),
        )
        session.commit()

    with env.get_session() as session:
        row = HumanAuditLogRepository(session).list_recent(action="reject")[0]
        assert row.new_value is None


# --- the run-scope emergency stop needs no model, store or checkpoint --------------


def _paused_run(session, *, step: str = "fund", status: str = "paused"):
    """A run parked at a gate, with no checkpoint behind it (nothing to resume)."""
    run = AgentRunRepository(session).create_run(
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        seed=1,
        universe=[],
        optimizer_config={"step": step},
        status=status,
    )
    session.commit()
    return run.id


@pytest.mark.parametrize(
    ("action", "extra"),
    [("reject", {}), ("killswitch", {}), ("freeze", {"scope": "run"})],
)
def test_run_scope_stop_finalises_rejected_without_a_model_store_or_checkpoint(
    clean_db, action, extra
) -> None:
    """Reject / killswitch / freeze(run) are the human's emergency stop: they must
    work with the provider down or the portfolio's ConstraintSet gone, so they settle
    in the DB instead of rebuilding the PM agent to resume the graph."""
    run_id = _paused_run(clean_db)

    result = apply_action(
        action, session=clean_db, run_id=run_id, portfolio_id=PORTFOLIO_ID, **extra
    )
    clean_db.commit()

    assert result.action == action
    run = AgentRunRepository(clean_db).get_run(run_id)
    assert run.status == "rejected"
    assert run.weights == {}
    assert [d.step for d in run.decisions] == ["place_orders"]
    assert run.decisions[0].hitl_decision == {"decision": "reject"}
    rows = HumanAuditLogRepository(clean_db).list_recent(action=action)
    assert [r.id for r in rows] == [result.audit_id]
    assert rows[0].run_id == run_id


def test_reject_of_a_profiler_gate_records_the_profiler_decision(clean_db) -> None:
    """The profiler's ``save_profile`` gate is settled the same model-free way."""
    run_id = _paused_run(clean_db, step="profiler")

    apply_action("reject", session=clean_db, run_id=run_id)
    clean_db.commit()

    run = AgentRunRepository(clean_db).get_run(run_id)
    assert run.status == "rejected"
    assert [(d.agent, d.step) for d in run.decisions] == [("profiler", "save_profile")]


def test_reject_refuses_an_already_finalised_run(clean_db) -> None:
    """Rejecting a run that already completed must not rewrite its terminal state."""
    run_id = _paused_run(clean_db, status="completed")

    with pytest.raises(ValueError, match="already finalised"):
        apply_action("reject", session=clean_db, run_id=run_id)
    clean_db.rollback()

    assert AgentRunRepository(clean_db).get_run(run_id).status == "completed"


def test_reject_of_an_unknown_run_raises_lookup_error(clean_db) -> None:
    with pytest.raises(LookupError, match="no agent_run"):
        apply_action("reject", session=clean_db, run_id=uuid.uuid4())


# --- a freeze halts the gates it leaves behind ------------------------------------


def _freeze_scope(env, scope: str) -> None:
    """Commit a FROZEN posture at ``scope`` from its own session, as a freeze does."""
    with env.get_session() as session:
        autonomy = AutonomyRepository(session)
        if scope == "fund":
            autonomy.set_state(AutonomyScope.FUND, None, AutonomyState.FROZEN)
        else:
            autonomy.set_state(
                AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.FROZEN
            )
        session.commit()


@pytest.mark.parametrize(
    ("action", "scope", "extra"),
    [
        ("approve", "fund", {}),
        ("approve", "portfolio", {}),
        ("respond", "portfolio", {"pm_directive": "Trim AAA first."}),
        ("edit", "fund", {"new_value": {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}}),
    ],
)
def test_a_frozen_scope_refuses_to_drive_an_open_gate(
    paused_env, action, scope, extra
) -> None:
    """A freeze only rejects the gates committed when it lands. One opened by a round
    that was already running, or any resume racing the freeze, must still be refused:
    approving it would place a ticket under the halt the freeze promised."""
    env = paused_env
    _freeze_scope(env, scope)

    with env.get_session() as session:
        with pytest.raises(RuntimeError, match="frozen"):
            apply_action(
                action,
                session=session,
                run_id=env.run_id,
                portfolio_id=PORTFOLIO_ID,
                checkpointer=env.saver,
                store=env.store,
                model=env.model_factory(),
                **extra,
            )
        session.rollback()

    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "paused"
        assert session.query(PaperOrder).count() == 0
        assert HumanAuditLogRepository(session).list_recent(action=action) == []
        # The checkpoint was never resumed, so the gate can still be rejected.
        assert [r.run_id for r in observe.pending_hitl(session, env.saver)] == [
            env.run_id
        ]


@pytest.mark.parametrize("action", ["reject", "killswitch"])
def test_the_emergency_stops_still_settle_a_gate_while_frozen(clean_db, action) -> None:
    """Refusing approve must not trap the adviser: reject and killswitch are how a
    gate that survived the freeze is closed."""
    run_id = _paused_run(clean_db)
    AutonomyRepository(clean_db).set_state(
        AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.FROZEN
    )
    clean_db.commit()

    apply_action(action, session=clean_db, run_id=run_id, portfolio_id=PORTFOLIO_ID)
    clean_db.commit()

    assert AgentRunRepository(clean_db).get_run(run_id).status == "rejected"


class _HaltDuringResumeModel(ScriptedFundModel):
    """Scripted PM that has a freeze land while a gate resume is inside the graph."""

    halt: Any = Field(default=None, exclude=True)

    def _pm_turn(self, messages):
        if self.halt is not None and _count_tool_calls(messages, "place_orders"):
            halt, self.halt = self.halt, None
            halt()
        return super()._pm_turn(messages)


def _freeze_rejects_the_run(env) -> Any:
    """A callback committing ``rejected`` from its own session, as a freeze does."""

    def freeze_run() -> None:
        with env.get_session() as other:
            AgentRunRepository(other).finalize_run(
                env.run_id, weights={}, status="rejected"
            )
            other.commit()

    return freeze_run


def test_a_respond_does_not_revive_a_run_a_freeze_rejected_meanwhile(
    paused_env,
) -> None:
    """The respond loaded the run as ``paused`` before entering the graph. A freeze that
    commits ``rejected`` during the PM's turn must win: last-writer-wins would flip the
    halted run back to ``responded`` and re-open its gate."""
    env = paused_env
    model = _HaltDuringResumeModel(
        universe=UNIVERSE,
        weights=env.expected,
        max_orders=2,
        halt=_freeze_rejects_the_run(env),
    )
    with env.get_session() as session:
        apply_action(
            "respond",
            session=session,
            run_id=env.run_id,
            portfolio_id=PORTFOLIO_ID,
            checkpointer=env.saver,
            store=env.store,
            model=model,
            pm_directive="Trim AAA first.",
        )
        session.commit()

    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "rejected"


def test_an_approve_does_not_complete_a_run_a_freeze_rejected_meanwhile(
    paused_env,
) -> None:
    """The approve checked the freeze before entering the graph. A freeze that commits
    ``rejected`` during the PM's post-ticket turn must win: finalising ``completed``
    over it would write positions into a halted fund, so the approval is refused and
    the caller's transaction (carrying the ticket) is rolled back."""
    env = paused_env
    model = _HaltDuringResumeModel(
        universe=UNIVERSE, weights=env.expected, halt=_freeze_rejects_the_run(env)
    )
    with env.get_session() as session:
        # ``RunNotOpenError`` (a ``ValueError``): the compare-and-set finalisation
        # refuses the run the freeze already settled.
        with pytest.raises(ValueError, match="rejected"):
            apply_action(
                "approve",
                session=session,
                run_id=env.run_id,
                portfolio_id=PORTFOLIO_ID,
                checkpointer=env.saver,
                store=env.store,
                model=model,
            )
        session.rollback()

    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "rejected"
        assert PositionRepository(session).get_holdings(PORTFOLIO_ID) == []


def test_run_action_reports_a_run_halted_by_a_mid_round_freeze(clean_db) -> None:
    """A round that ends ``rejected`` because a freeze landed has no gate awaiting
    approval; the status line must not claim one."""
    run_id = _paused_run(clean_db, status="rejected")
    halted = SimpleNamespace(run_id=run_id, status="rejected")

    with mock.patch("fund.agents.graph.run_fund", return_value=halted):
        result = apply_action(
            "run",
            session=clean_db,
            portfolio_id=PORTFOLIO_ID,
            mandate=make_mandate(),
        )

    assert "frozen" in result.message
    assert "awaiting approval" not in result.message


@pytest.mark.parametrize("scope", ["portfolio", "fund"])
def test_a_freeze_records_every_run_it_halted(clean_db, scope) -> None:
    """A halted run is just a ``rejected`` row with nothing saying why. Each needs its
    own decision naming the freeze, and the audit row has to list the runs it halted,
    or an auditor cannot tell a freeze from a human or a bug rejecting them."""
    fund_run = _paused_run(clean_db)
    profiler_run = _paused_run(clean_db, step="profiler")

    apply_action(
        "freeze",
        session=clean_db,
        scope=scope,
        portfolio_id=PORTFOLIO_ID if scope == "portfolio" else None,
        rationale="drawdown breach",
    )
    clean_db.commit()

    runs = AgentRunRepository(clean_db)
    expected_gate = {fund_run: "place_orders", profiler_run: "save_profile"}
    for run_id, step in expected_gate.items():
        run = runs.get_run(run_id)
        assert run.status == "rejected"
        assert [(d.step, d.hitl_decision) for d in run.decisions] == [
            (step, {"decision": "reject", "via": "freeze"})
        ]
    row = HumanAuditLogRepository(clean_db).list_recent(action="freeze")[0]
    assert row.new_value["state"] == "frozen"
    assert row.new_value["reason"] == "drawdown breach"
    assert sorted(row.new_value["halted_runs"]) == sorted(map(str, expected_gate))
