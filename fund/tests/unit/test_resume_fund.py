"""resume_fund + _finalize_hitl: rebuild-to-resume + parity.

``run_fund`` drives the PM deep agent to the ``place_orders`` HITL gate, marking
the run ``paused`` and persisting ``thread_id = str(run_id)`` on the row. A human
observer then resumes it **without the live handle**: ``resume_fund`` rebuilds the
PM agent against the *same* checkpointer + thread in a **fresh agent instance** and
commits (approve) or discards (reject) the optimizer's ticket. Both the live-handle
``FundRun.resume`` and the rebuilt ``resume_fund`` delegate their finalisation to the
one shared ``_finalize_hitl``, so the two paths cannot diverge — asserted directly by
a parity test that resumes two identical runs one each way and compares DB state.

Zero live LLM, zero network: ``ScriptedFundModel`` + ``MemorySaver`` +
``InMemoryStore`` + the in-memory ``db_session``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import uuid
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    BrokenPlaceOrdersFundModel,
    PmFailsAfterOrderModel,
    ScriptedFundModel,
    _count_tool_calls,
    _task_call,
    classify_instruments,
    expected_weights,
    make_constraint_set,
    make_energy_exclusion_constraint_set,
    make_mandate,
    seed_panel,
)
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision, PaperOrder, PortfolioJournal
from pydantic import Field

from fund import observe
from fund.agents import graph as graph_mod
from fund.agents.graph import (
    FundRun,
    _finalize_hitl,
    _resume_command,
    _settle_hitl,
    resume_fund,
    resume_run,
    run_fund,
)
from fund.audit import (
    AgentRunRepository,
    AutonomyRepository,
    AutonomyScope,
    AutonomyState,
    MandateRepository,
    MifidProfileRepository,
    OrderRepository,
    PositionRepository,
    put_constraint_set,
)
from fund.config import settings

# A second letter-bearing portfolio id for the parity test (a distinct
# idempotency key so both approvals write their own paper ticket).
_PID_B = uuid.UUID("a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d")


# --- helpers ----------------------------------------------------------------


def _store_with_cs(portfolio_id: uuid.UUID = PORTFOLIO_ID) -> InMemoryStore:
    store = InMemoryStore()
    put_constraint_set(
        store,
        make_constraint_set(portfolio_id),
        store_key=settings.constraint_set_store_key,
    )
    return store


def _run_to_gate(
    model: ScriptedFundModel,
    db_session,
    saver: MemorySaver,
    store: InMemoryStore,
    *,
    portfolio_id: uuid.UUID = PORTFOLIO_ID,
) -> FundRun:
    return run_fund(
        model,
        make_mandate(portfolio_id),
        portfolio_id=portfolio_id,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )


def _paper_orders(
    db_session, portfolio_id: uuid.UUID | None = None
) -> list[PaperOrder]:
    query = db_session.query(PaperOrder)
    if portfolio_id is not None:
        query = query.filter(PaperOrder.portfolio_id == portfolio_id)
    return query.all()


def _decisions(db_session, run_id) -> list[AgentDecision]:
    return db_session.query(AgentDecision).filter(AgentDecision.run_id == run_id).all()


def _positions_map(
    db_session, portfolio_id: uuid.UUID = PORTFOLIO_ID
) -> dict[str, float]:
    return {
        p.ticker: p.weight
        for p in PositionRepository(db_session).get_holdings(portfolio_id)
    }


# --- run_fund persists the per-run thread + marks paused -----------------------


def test_run_fund_persists_thread_id_and_marks_paused(db_session) -> None:
    seed_panel(db_session)
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected_weights(db_session))

    run = _run_to_gate(model, db_session, MemorySaver(), _store_with_cs())

    row = AgentRunRepository(db_session).get_run(run.run_id)
    # thread_id defaults to str(run_id) when not set explicitly.
    assert row.thread_id == str(run.run_id)
    # the DB row is flipped to "paused" at the gate so observers can find it.
    assert row.status == "paused"


def test_run_fund_honours_explicit_thread_id(db_session) -> None:
    seed_panel(db_session)
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected_weights(db_session))

    run = run_fund(
        model,
        make_mandate(),
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=MemorySaver(),
        store=_store_with_cs(),
        thread_id="explicit-thread",
    )

    assert (
        AgentRunRepository(db_session).get_run(run.run_id).thread_id
        == "explicit-thread"
    )


# --- resume_fund approve: fresh agent commits ticket + upserts positions ------


def test_resume_fund_approve_commits_ticket_positions_and_completes(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    # A FRESH agent/model instance resumes — no reliance on the live FundRun handle.
    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    orders = _paper_orders(db_session)
    assert len(orders) == 1
    assert orders[0].weights == expected
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert row.weights == expected
    assert _positions_map(db_session) == expected
    executor = [
        d
        for d in _decisions(db_session, run.run_id)
        if d.agent == "executor" and d.step == "place_orders"
    ]
    assert len(executor) == 1
    assert executor[0].hitl_decision == {"decision": "approve"}


def test_resume_fund_approve_sizes_ticket_at_the_stored_mandate_capital(
    db_session,
) -> None:
    """The rebuild-to-resume path re-reads the mandate, so the ticket it commits is
    sized at the mandate's capital rather than the tool's default notional."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    mandate = make_mandate().model_copy(update={"capital": Decimal("250000")})
    MandateRepository(db_session).upsert(mandate)

    run = run_fund(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )
    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    (order,) = _paper_orders(db_session)
    assert order.notional == 250_000.0


def test_resume_fund_sizes_the_ticket_from_the_capital_the_run_opened_with(
    db_session,
) -> None:
    """The rebuild-to-resume path prices the ticket at the capital the PM saw."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    mandate = make_mandate().model_copy(update={"capital": Decimal("40000")})
    run = run_fund(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    (order,) = _paper_orders(db_session)
    assert order.notional == pytest.approx(40_000.0)


def test_resume_fund_prices_the_ticket_at_the_run_capital_after_a_mandate_edit(
    db_session,
) -> None:
    """A mandate edited while the gate is pending must not re-price the ticket."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    opened = make_mandate().model_copy(update={"capital": Decimal("40000")})
    MandateRepository(db_session).upsert(opened)
    run = run_fund(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        opened,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )
    MandateRepository(db_session).upsert(
        opened.model_copy(update={"capital": Decimal("900000")})
    )

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    (order,) = _paper_orders(db_session)
    assert order.notional == pytest.approx(40_000.0)


# --- resume_fund reject: no order, no positions, rejected --------------------


def test_resume_fund_reject_places_no_order(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert _paper_orders(db_session) == []
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "rejected"
    assert row.weights == {}
    assert _positions_map(db_session) == {}
    assert any(
        d.hitl_decision and d.hitl_decision.get("decision") == "reject"
        for d in _decisions(db_session, run.run_id)
    )


# --- parallel place_orders calls: one decision per hanging tool call -----------


def _run_to_double_gate(db_session, saver, store, expected) -> FundRun:
    """Drive a PM that emits two ``place_orders`` calls in one message to its gate."""
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected, parallel_orders=2),
        db_session,
        saver,
        store,
    )
    assert run.status == "paused"
    assert len(run.interrupt["action_requests"]) == 2
    return run


def test_resume_fund_approve_resolves_a_gate_with_parallel_place_orders(
    db_session,
) -> None:
    """Two gated calls in one PM message are both approved by one adviser approve;
    the idempotent ticket is still written once and the run completes."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_double_gate(db_session, saver, store, expected)

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected, parallel_orders=2),
    )

    assert len(_paper_orders(db_session)) == 1
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"


def test_resume_fund_reject_resolves_a_gate_with_parallel_place_orders(
    db_session,
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_double_gate(db_session, saver, store, expected)

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected, parallel_orders=2),
    )

    assert _paper_orders(db_session) == []
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "rejected"


def test_fundrun_resume_respond_resolves_a_gate_with_parallel_place_orders(
    db_session,
) -> None:
    """The live handle sizes its resume to the pending gate too, so a respond to a
    two-call interrupt neither raises nor leaves the run wedged."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_double_gate(db_session, saver, store, expected)

    run.resume("respond", pm_directive="keep it to one ticket")

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "incomplete"
    assert len(_respond_decisions(db_session, run.run_id)) == 1


# --- resume_fund guards ------------------------------------------------------


def test_resume_fund_unknown_run_id_raises(db_session) -> None:
    with pytest.raises(LookupError):
        resume_fund(
            uuid.uuid4(),
            "approve",
            session=db_session,
            checkpointer=MemorySaver(),
            store=_store_with_cs(),
            model=ScriptedFundModel(),
        )


def test_resume_fund_rejects_unknown_decision(db_session) -> None:
    with pytest.raises(ValueError, match="approve"):
        resume_fund(
            uuid.uuid4(),
            "edit",
            session=db_session,
            checkpointer=MemorySaver(),
            store=_store_with_cs(),
            model=ScriptedFundModel(),
        )


def test_resume_fund_requires_active_constraint_set(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        saver,
        _store_with_cs(),
    )
    # An empty store no longer resolves the portfolio's profile → fail fast.
    with pytest.raises(RuntimeError, match="profiler"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=InMemoryStore(),
            model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        )


# --- resume_fund under a FROZEN posture ---------------------------------------


def _freeze(db_session, scope: AutonomyScope) -> None:
    pid = PORTFOLIO_ID if scope is AutonomyScope.PORTFOLIO else None
    AutonomyRepository(db_session).set_state(
        scope, pid, AutonomyState.FROZEN, reason="killswitch"
    )


@pytest.mark.parametrize("scope", [AutonomyScope.FUND, AutonomyScope.PORTFOLIO])
@pytest.mark.parametrize("decision", ["approve", "respond"])
def test_resume_fund_refuses_to_continue_a_gate_under_a_freeze(
    db_session, scope, decision
) -> None:
    """A gate that opened before (or raced) a freeze must not be approvable after it."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _freeze(db_session, scope)

    with pytest.raises(RuntimeError, match="frozen"):
        resume_fund(
            run.run_id,
            decision,
            session=db_session,
            checkpointer=saver,
            store=store,
            model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
            pm_directive="tilt to AAA",
        )

    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


def test_resume_fund_still_rejects_a_gate_under_a_freeze(db_session) -> None:
    """Reject is the way out of a frozen gate, so the freeze must not block it."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _freeze(db_session, AutonomyScope.FUND)

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"


# --- resume_fund after the portfolio was re-profiled ---------------------------


def _add_profile_version(db_session) -> int:
    """Append the next MiFID profile version for ``PORTFOLIO_ID``; return it."""
    return (
        MifidProfileRepository(db_session)
        .add_version(
            portfolio_id=PORTFOLIO_ID,
            questionnaire={"knowledge": "basic"},
            constraint_set={"a_gamma": 2.5},
            suitability={"inconsistency_flags": []},
            store_key=settings.constraint_set_store_key,
        )
        .version
    )


def test_run_fund_records_the_profile_version_it_was_built_under(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    _add_profile_version(db_session)
    v2 = _add_profile_version(db_session)

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.optimizer_config["profile_version"] == v2


@pytest.mark.parametrize("decision", ["approve", "respond"])
def test_resume_fund_refuses_a_gate_built_under_a_superseded_profile(
    db_session, decision
) -> None:
    """A re-profile voids every gate opened under the old profile.

    ``place_orders`` sizes the ticket from the weights the optimizer produced under
    the old ``ConstraintSet``; committing them after the adviser tightened the
    bounds would trade a portfolio the new profile never validated.
    """
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    _add_profile_version(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _add_profile_version(db_session)  # the adviser re-profiles while the gate waits

    with pytest.raises(RuntimeError, match="re-profiled"):
        resume_fund(
            run.run_id,
            decision,
            session=db_session,
            checkpointer=saver,
            store=store,
            model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
            pm_directive="tilt to AAA",
        )

    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


def test_resume_fund_still_rejects_a_gate_built_under_a_superseded_profile(
    db_session,
) -> None:
    """Reject is how the adviser clears the stale gate, so it must stay open."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    _add_profile_version(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _add_profile_version(db_session)

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"


def test_resume_fund_approves_when_the_profile_is_unchanged(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    _add_profile_version(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"


# --- stale approval: an older paused run must not overwrite newer holdings -----


def _two_paused_runs(db_session):
    """Two rebalance rounds of one portfolio, both parked at the gate, older first."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    older = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    newer = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    audit = AgentRunRepository(db_session)
    # SQLite's CURRENT_TIMESTAMP has one-second resolution, so pin the ordering.
    audit.get_run(older.run_id).started_at = datetime(2026, 10, 24, 6, 0)
    audit.get_run(newer.run_id).started_at = datetime(2026, 10, 26, 7, 0)
    db_session.flush()
    return older.run_id, newer.run_id, expected, saver, store


def _resume(db_session, run_id, decision, expected, saver, store) -> None:
    resume_fund(
        run_id,
        decision,
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )


def test_resume_fund_approve_refuses_a_run_older_than_a_completed_one(
    db_session,
) -> None:
    older, newer, expected, saver, store = _two_paused_runs(db_session)
    _resume(db_session, newer, "approve", expected, saver, store)
    assert len(_paper_orders(db_session)) == 1

    with pytest.raises(ValueError, match="newer"):
        _resume(db_session, older, "approve", expected, saver, store)

    assert AgentRunRepository(db_session).get_run(older).status == "paused"
    assert len(_paper_orders(db_session)) == 1  # no second ticket


def test_resume_fund_reject_still_clears_a_stale_run(db_session) -> None:
    older, newer, expected, saver, store = _two_paused_runs(db_session)
    _resume(db_session, newer, "approve", expected, saver, store)

    _resume(db_session, older, "reject", expected, saver, store)

    assert AgentRunRepository(db_session).get_run(older).status == "rejected"


def test_resume_fund_approves_paused_runs_oldest_first(db_session) -> None:
    older, newer, expected, saver, store = _two_paused_runs(db_session)

    _resume(db_session, older, "approve", expected, saver, store)
    _resume(db_session, newer, "approve", expected, saver, store)

    audit = AgentRunRepository(db_session)
    assert audit.get_run(older).status == "completed"
    assert audit.get_run(newer).status == "completed"


@pytest.mark.parametrize("decision", ["approve", "reject", "respond"])
def test_resume_fund_refuses_a_run_that_is_not_at_a_gate(db_session, decision) -> None:
    """A ``pending`` row is a run that is still being driven or crashed before reaching
    the gate. Resuming it would re-drive the PM headlessly from the failed checkpoint
    (or race the live driver on the same thread) and then finalise it as if an adviser
    had approved a gate that never opened."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    row = AgentRunRepository(db_session).get_run(run.run_id)
    row.status = "pending"
    db_session.flush()
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(ValueError, match="pending"):
        resume_fund(
            run.run_id,
            decision,
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
            pm_directive="rebalance more slowly",
        )

    assert model.generate_calls == 0
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert row.status == "pending"


def test_fundrun_resume_refuses_a_run_that_is_not_at_a_gate(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )
    AgentRunRepository(db_session).get_run(run.run_id).status = "pending"
    db_session.flush()

    with pytest.raises(ValueError, match="pending"):
        run.resume("approve")

    assert _paper_orders(db_session) == []


# --- a run that already left its gate cannot be resumed again ---------------------

# Every terminal status a run reaches without an adviser resuming it: the checkpoint
# of such a run is left standing at (or just past) the gate.
_FINALISED_STATUSES = ("completed", "rejected", "incomplete", "blocked", "failed")


@pytest.mark.parametrize("decision", ["approve", "reject", "respond"])
@pytest.mark.parametrize("status", _FINALISED_STATUSES)
def test_resume_fund_refuses_a_run_that_is_already_finalised(
    db_session, status, decision
) -> None:
    """A freeze rejects live gates at the DB without resolving their checkpoints, and a
    stale observer can race a concurrent approve. A blind ``Command(resume=...)`` on
    such a run would re-drive ``place_orders`` against a halted run."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    row = AgentRunRepository(db_session).get_run(run.run_id)
    row.status = status
    db_session.flush()
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(ValueError, match="already finalised"):
        resume_fund(
            run.run_id,
            decision,
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
            pm_directive="rebalance more slowly",
        )

    assert model.generate_calls == 0
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert row.status == status


@pytest.mark.parametrize("status", _FINALISED_STATUSES)
def test_fundrun_resume_refuses_a_run_that_is_already_finalised(
    db_session, status
) -> None:
    """The in-process handle has no status guard of its own: it is the claim on the run
    that keeps a stale handle from re-driving the checkpointed ``place_orders``."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )
    row = AgentRunRepository(db_session).get_run(run.run_id)
    row.status = status
    db_session.flush()

    with pytest.raises(ValueError, match="already finalised"):
        run.resume("approve")

    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert row.status == status


def test_a_second_approve_of_a_completed_run_is_refused(db_session) -> None:
    """Without an idempotency key nothing but the run's status stops a second approve
    (a double click, a second observer) from placing a second ticket."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )
    tickets = [o.id for o in _paper_orders(db_session)]
    holdings = _positions_map(db_session)
    assert len(tickets) == 1
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(ValueError, match="already finalised"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
        )

    assert model.generate_calls == 0
    assert [o.id for o in _paper_orders(db_session)] == tickets
    assert _positions_map(db_session) == holdings
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"


def test_fundrun_resume_refuses_a_second_approve(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )
    run.resume("approve")
    tickets = [o.id for o in _paper_orders(db_session)]
    assert len(tickets) == 1

    with pytest.raises(ValueError, match="already finalised"):
        run.resume("approve")

    assert [o.id for o in _paper_orders(db_session)] == tickets
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"


def test_approve_stays_refused_after_a_freeze_is_rearmed(db_session) -> None:
    """A freeze rejects the live gate but leaves its checkpoint standing; once the
    adviser re-arms, only the run's rejected status keeps a stale approve from placing
    the order the freeze halted."""
    from fund.actions import apply_action
    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    apply_action(
        "freeze",
        session=db_session,
        scope="portfolio",
        portfolio_id=PORTFOLIO_ID,
        rationale="halt",
    )
    AutonomyRepository(db_session).set_state(
        AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.ARMED
    )
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(ValueError, match="already finalised"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
        )

    assert model.generate_calls == 0
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"


# --- two resolvers racing for the same gate ----------------------------------


class _ConcurrentResolutionDuringClosingTurn(ScriptedFundModel):
    """A PM whose closing turn (after ``place_orders`` ran) overlaps another resolver
    committing its own outcome for the same run — a reject, or a freeze killswitch."""

    concurrent: object = None

    def _pm_turn(self, messages):
        if self.concurrent is not None and _count_tool_calls(messages, "place_orders"):
            self.concurrent()
        return super()._pm_turn(messages)


def test_approve_does_not_overwrite_a_run_rejected_while_it_was_in_flight(
    db_session,
) -> None:
    """The adviser presses Approve, then Reject (or a freeze halts the fund) while the
    approve is still resuming the graph. The reject finalises first; the approve must
    then fail rather than stamp the run ``completed`` and rewrite the positions."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    def reject_lands() -> None:
        AgentRunRepository(db_session).finalize_run(
            run.run_id, weights={}, status="rejected"
        )

    model = _ConcurrentResolutionDuringClosingTurn(
        universe=UNIVERSE, weights=expected, concurrent=reject_lands
    )
    with pytest.raises(ValueError, match="rejected"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
        )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "rejected"
    assert _positions_map(db_session) == {}


def test_resume_fund_backs_off_when_another_resolver_finalised_the_run_first(
    db_session, monkeypatch
) -> None:
    """The status guard is a plain read, so a reject can commit between it and the
    resume. The loser must not drive the graph at all: that would run
    ``place_orders`` for a run the adviser already rejected."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    build_fund_agent = graph_mod.build_fund_agent

    def reject_lands_then_build(*args, **kwargs):
        AgentRunRepository(db_session).finalize_run(
            run.run_id, weights={}, status="rejected"
        )
        return build_fund_agent(*args, **kwargs)

    monkeypatch.setattr(graph_mod, "build_fund_agent", reject_lands_then_build)
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(ValueError, match="no longer at a live gate"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
        )

    assert model.generate_calls == 0
    assert _paper_orders(db_session) == []
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"


# --- an approval that arrives too late or out of order ---------------------------


def _age_run(db_session, run_id, *, days: int) -> None:
    """Back-date when the run opened — what an unattended gate looks like later."""
    row = AgentRunRepository(db_session).get_run(run_id)
    row.started_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    db_session.flush()


def test_resume_fund_refuses_an_approve_once_the_run_outlived_the_approval_window(
    db_session,
) -> None:
    """The ticket fills at the first close after the run's ``asof``, so approving a gate
    that has sat for weeks would backdate a fill the adviser chose with the benefit of
    the bars since. Such a run can only be rejected, and re-run on a fresh bar."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _age_run(db_session, run.run_id, days=settings.approval_max_age_days + 1)
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(ValueError, match="stale"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
        )

    assert model.generate_calls == 0
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


def test_resume_fund_still_rejects_a_run_that_outlived_the_approval_window(
    db_session,
) -> None:
    """A stale gate must stay closable, or it would block the portfolio's scheduled
    rebalances until someone edits the database."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _age_run(db_session, run.run_id, days=settings.approval_max_age_days + 1)

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"


def test_fundrun_resume_refuses_an_approve_once_the_run_outlived_the_approval_window(
    db_session,
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )
    _age_run(db_session, run.run_id, days=settings.approval_max_age_days + 1)

    with pytest.raises(ValueError, match="stale"):
        run.resume("approve")

    assert _paper_orders(db_session) == []
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


def test_resume_fund_refuses_an_approve_older_than_the_portfolio_holdings(
    db_session,
) -> None:
    """Approving an earlier bar's run after a later one completed would replace the
    current holdings snapshot with the older run's weights and ``asof``."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    later = ASOF + dt.timedelta(days=3)
    PositionRepository(db_session).set_holdings(
        PORTFOLIO_ID, [{"ticker": "AAA", "weight": 1.0}], asof=later
    )
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(ValueError, match="older"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
        )

    assert model.generate_calls == 0
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {"AAA": 1.0}
    assert {
        p.asof for p in PositionRepository(db_session).get_holdings(PORTFOLIO_ID)
    } == {later}
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


# --- a freeze that landed while the run was still being driven -----------------


def _freeze_by_name(db_session, scope: str) -> None:
    """Flip the autonomy row FROZEN without touching any run — what a freeze that
    committed before the run row became visible (an in-flight sweep) leaves behind."""
    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    repo = AutonomyRepository(db_session)
    if scope == "fund":
        repo.set_state(AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="halt")
    else:
        repo.set_state(
            AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.FROZEN, reason="halt"
        )


@pytest.mark.parametrize("scope", ["fund", "portfolio"])
@pytest.mark.parametrize("decision", ["approve", "respond"])
def test_resume_fund_refuses_to_advance_a_run_whose_scope_is_frozen(
    db_session, scope, decision
) -> None:
    """A round that was mid-flight when the killswitch fired commits as ``paused`` after
    the freeze already listed the live gates, so it is approvable under a halted
    autonomy. Approve (and the respond that keeps the run alive) must refuse."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _freeze_by_name(db_session, scope)
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    with pytest.raises(RuntimeError, match="frozen"):
        resume_fund(
            run.run_id,
            decision,
            session=db_session,
            checkpointer=saver,
            store=store,
            model=model,
            pm_directive="go on",
        )

    assert model.generate_calls == 0
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


def test_resume_fund_reject_is_still_allowed_when_frozen(db_session) -> None:
    """Rejecting is the safe direction: an adviser must be able to close the stranded
    gate of a halted fund."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _freeze_by_name(db_session, "fund")

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"
    assert _paper_orders(db_session) == []


def test_fundrun_resume_refuses_approve_when_the_fund_is_frozen(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )
    _freeze_by_name(db_session, "fund")

    with pytest.raises(RuntimeError, match="frozen"):
        run.resume("approve")

    assert _paper_orders(db_session) == []
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


# --- an approved gate that is not the order commit ------------------------------

_PRIOR_HOLDINGS = {"AAA": 0.6, "BBB": 0.4}
_EXTRA_GATES = ("optimize_portfolio", "place_orders")


def _run_to_extra_gate(db_session, saver, store, expected) -> FundRun:
    """A portfolio's second round, whose persisted mandate adds ``optimize_portfolio``
    to the gates: the PM pauses at that gate, well before ``place_orders``. The
    portfolio already holds ``_PRIOR_HOLDINGS``. The first round's forced gate set is
    spent, so the mandate's own gates apply."""
    from fund.audit import MandateRepository
    from fund.audit.autonomy_repository import AutonomyRepository

    AutonomyRepository(db_session).consume_first_round(PORTFOLIO_ID)
    PositionRepository(db_session).set_holdings(
        PORTFOLIO_ID,
        [{"ticker": t, "weight": w} for t, w in _PRIOR_HOLDINGS.items()],
        asof=dt.date(2024, 1, 2),
    )
    mandate = make_mandate().model_copy(update={"hitl_gates": _EXTRA_GATES})
    MandateRepository(db_session).upsert(mandate)
    run = run_fund(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )
    gated = [r["name"] for r in run.interrupt["action_requests"]]
    assert gated == ["optimize_portfolio"]
    return run


def _assert_still_awaiting_the_next_gate(db_session, run_id) -> None:
    row = AgentRunRepository(db_session).get_run(run_id)
    assert row.status == "paused"
    assert row.finished_at is None
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == _PRIOR_HOLDINGS
    assert db_session.query(PortfolioJournal).all() == []


def test_resume_fund_approve_of_an_extra_gate_keeps_the_run_at_the_next_gate(
    db_session,
) -> None:
    """Approving ``optimize_portfolio`` lets the PM run on to the ``place_orders`` gate,
    which pauses the graph again. The approve must not finalise the run: that would
    stamp it completed with no ticket, replace the holdings with an empty or partial
    snapshot, and strand the real gate behind the finalised-run guard."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_extra_gate(db_session, saver, store, expected)

    result = resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert result["__interrupt__"][0].value["action_requests"][0]["name"] == (
        "place_orders"
    )
    _assert_still_awaiting_the_next_gate(db_session, run.run_id)

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    (order,) = _paper_orders(db_session)
    assert order.weights == expected
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert row.weights == expected
    assert _positions_map(db_session) == expected


def test_fundrun_resume_approve_of_an_extra_gate_keeps_the_run_at_the_next_gate(
    db_session,
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_extra_gate(db_session, MemorySaver(), _store_with_cs(), expected)

    run.resume("approve")

    _assert_still_awaiting_the_next_gate(db_session, run.run_id)

    run.resume("approve")

    (order,) = _paper_orders(db_session)
    assert order.weights == expected
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert _positions_map(db_session) == expected


def test_resume_fund_reject_of_an_extra_gate_still_closes_the_run(db_session) -> None:
    """A reject is the adviser abandoning the rebalance: it closes the run even though
    the PM goes on to re-gate, so the freeze-a-run killswitch (a reject) always ends it
    and the holdings stay as they were."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_extra_gate(db_session, saver, store, expected)

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "rejected"
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == _PRIOR_HOLDINGS


# --- parity: FundRun.resume and resume_fund leave identical state ------------


def test_resume_fund_matches_fundrun_resume(db_session) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)

    # Portfolio A — resumed via the live in-process handle (FundRun.resume).
    run_a = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(PORTFOLIO_ID),
        portfolio_id=PORTFOLIO_ID,
    )
    run_a.resume("approve")

    # Portfolio B — resumed via the rebuild-to-resume path (fresh agent).
    saver_b = MemorySaver()
    store_b = _store_with_cs(_PID_B)
    run_b = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        saver_b,
        store_b,
        portfolio_id=_PID_B,
    )
    resume_fund(
        run_b.run_id,
        "approve",
        session=db_session,
        checkpointer=saver_b,
        store=store_b,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    repo = AgentRunRepository(db_session)
    row_a, row_b = repo.get_run(run_a.run_id), repo.get_run(run_b.run_id)
    # Identical terminal state (modulo run/portfolio/order ids).
    assert row_a.status == row_b.status == "completed"
    assert row_a.weights == row_b.weights == expected
    assert (
        _positions_map(db_session, PORTFOLIO_ID)
        == _positions_map(db_session, _PID_B)
        == expected
    )
    order_a = _paper_orders(db_session, PORTFOLIO_ID)
    order_b = _paper_orders(db_session, _PID_B)
    assert len(order_a) == len(order_b) == 1
    assert order_a[0].weights == order_b[0].weights == expected


# --- approve with no fill bar: a failed place_orders is never "completed" -----


def _drop_bars_after_asof(db_session) -> None:
    """Remove every priced bar past ``ASOF`` so ``place_orders`` has no fill bar."""
    from portopt_db.models.market_data.yfinance_data import PriceHistory

    db_session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
    db_session.flush()


def _assert_nothing_finalised(db_session, run_id) -> None:
    row = AgentRunRepository(db_session).get_run(run_id)
    assert row.status == "paused"
    assert row.finished_at is None
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert db_session.query(PortfolioJournal).all() == []


def _assert_gate_still_open(db_session, saver, run_id) -> None:
    """A refused approve must not consume the gate: the checkpoint keeps its interrupt,
    so the TUI Decisions queue still lists the run and an approve can be retried."""
    row = AgentRunRepository(db_session).get_run(run_id)
    assert observe.interrupt_for(saver, row) is not None
    queued = observe.pending_hitl(db_session, saver, PORTFOLIO_ID)
    assert [summary.run_id for summary in queued] == [run_id]


def test_resume_fund_approve_without_a_fill_bar_does_not_finalise(db_session) -> None:
    """``place_orders`` fails by design when no close follows ``asof``. Approving then
    must surface the failure and leave the run paused at its gate: no ``completed``
    stamp, no positions snapshot, no journal row, and the checkpoint's interrupt intact
    (the gated call never ran) so the approval stays retryable from the TUI queue."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _drop_bars_after_asof(db_session)

    with pytest.raises(RuntimeError, match="paper ticket"):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        )

    _assert_nothing_finalised(db_session, run.run_id)
    _assert_gate_still_open(db_session, saver, run.run_id)


def test_fundrun_resume_approve_without_a_fill_bar_does_not_finalise(
    db_session,
) -> None:
    """The in-process handle refuses the unfillable approve the same way."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        saver,
        _store_with_cs(),
    )
    _drop_bars_after_asof(db_session)

    with pytest.raises(RuntimeError, match="paper ticket"):
        run.resume("approve")

    _assert_nothing_finalised(db_session, run.run_id)
    _assert_gate_still_open(db_session, saver, run.run_id)


# --- a failure after the gate is consumed must not orphan the approval --------


def test_resume_fund_approve_survives_pm_failure_after_the_ticket(db_session) -> None:
    """The PM's follow-up turn can fail (429) after ``place_orders`` ran. The gate is
    consumed and the ticket written, so the approval must still be finalised rather
    than propagate: the caller rolls the session back on an exception, which would
    drop the ticket while the checkpoint keeps the tool step done."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    result = resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=PmFailsAfterOrderModel(universe=UNIVERSE, weights=expected),
    )

    assert "429" in result["resume_error"]
    assert len(_paper_orders(db_session)) == 1
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert _positions_map(db_session) == expected


def test_resume_fund_reject_survives_pm_failure_after_the_gate(db_session) -> None:
    """A reject whose follow-up PM turn fails is still recorded, not left paused."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    class _FailsOnAnyPmTurn(ScriptedFundModel):
        def _pm_turn(self, messages):
            raise RuntimeError("429 rate limited")

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=_FailsOnAnyPmTurn(universe=UNIVERSE, weights=expected),
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"
    assert _paper_orders(db_session) == []


def test_resume_fund_reraises_when_the_gate_was_not_consumed(
    db_session, monkeypatch
) -> None:
    """A failure before the interrupt is consumed leaves the gate pending, so the
    exception propagates and the adviser can simply retry."""
    from fund.agents import graph

    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    class _DeadAgent:
        def get_state(self, *_args, **_kwargs):
            return SimpleNamespace(interrupts=(), values={})

        def invoke(self, *_args, **_kwargs):
            raise ConnectionError("checkpointer unreachable")

    monkeypatch.setattr(graph, "build_fund_agent", lambda *a, **k: _DeadAgent())

    with pytest.raises(ConnectionError):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        )

    assert observe._has_pending_interrupt(saver, str(run.run_id))
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"


# --- approve of a refused place_orders must not read as a completed rebalance -


def test_resume_fund_approve_when_risk_never_checked_finalises_incomplete(
    db_session,
) -> None:
    """The adviser approving cannot make a refused placement real: with no risk
    verdict ``place_orders`` writes no ticket, so the rebuilt-to-resume run must
    end ``incomplete`` and leave the holdings alone, like ``FundRun.resume``."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected, risk_checks=False),
        db_session,
        saver,
        store,
    )

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected, risk_checks=False),
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "incomplete"
    assert not row.weights
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    closing = [
        d.hitl_decision
        for d in _decisions(db_session, run.run_id)
        if d.agent == "orchestrator"
    ]
    assert {"decision": "incomplete", "reason": "place_orders_failed"} in closing


class _ReallocatingModel(ScriptedFundModel):
    """A PM that, once the gate has placed the ticket, delegates to the allocator again.

    The re-run optimises a narrower universe, so the audit trail's latest optimizer
    weights drift away from the weights the ticket was placed (and approved) with.
    """

    def _pm_turn(self, messages):
        n_task = _count_tool_calls(messages, "task")
        if _count_tool_calls(messages, "place_orders") >= 1 and n_task == 4:
            object.__setattr__(self, "universe", UNIVERSE[:2])
            return _task_call("allocator", n_task)
        return super()._pm_turn(messages)


def _assert_finalised_with_ticket_weights(db_session, run_id, expected) -> None:
    # the model really did re-run the allocator after the ticket was placed
    assert AgentRunRepository(db_session).latest_optimizer_weights(run_id) != expected
    orders = _paper_orders(db_session)
    assert len(orders) == 1
    assert orders[0].weights == expected
    row = AgentRunRepository(db_session).get_run(run_id)
    assert row.status == "completed"
    assert row.weights == expected
    assert _positions_map(db_session) == expected


def test_resume_fund_approve_finalises_with_the_placed_ticket_weights(
    db_session,
) -> None:
    """A post-approval allocator re-run must not change what is finalised: the run's
    weights and the positions snapshot are the approved ticket's, not whatever the
    audit trail's latest optimizer decision says afterwards."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=_ReallocatingModel(universe=UNIVERSE, weights=expected),
    )

    _assert_finalised_with_ticket_weights(db_session, run.run_id, expected)


def test_fundrun_resume_approve_finalises_with_the_placed_ticket_weights(
    db_session,
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        _ReallocatingModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )

    run.resume("approve")

    _assert_finalised_with_ticket_weights(db_session, run.run_id, expected)


# --- a retried approve after the checkpoint already moved past the gate ------


@contextlib.contextmanager
def _rolled_back(db_session):
    """Run the block as the caller's open transaction, then roll it back — what the CLI
    and TUI do when an approve raises. A savepoint stands in for that transaction
    because ``db_session`` shares one outer transaction across the whole test."""
    savepoint = db_session.begin_nested()
    try:
        yield
    finally:
        savepoint.rollback()


def _assert_retry_completed_with_ticket(db_session, run_id, expected) -> None:
    (order,) = _paper_orders(db_session)
    assert order.weights == expected
    row = AgentRunRepository(db_session).get_run(run_id)
    assert row.status == "completed"
    assert row.weights == expected
    assert _positions_map(db_session) == expected
    executor = [
        d
        for d in _decisions(db_session, run_id)
        if d.agent == "executor" and d.step == "place_orders"
    ]
    assert len(executor) == 1


def test_resume_fund_retry_replaces_a_ticket_lost_with_the_rolled_back_session(
    db_session,
) -> None:
    """``place_orders`` writes on the caller's session while the LangGraph saver commits
    the checkpoint on its own connection. When the caller's transaction is lost after
    the approve (a failed finalisation or commit) the ticket is gone but the checkpoint
    has moved past the gate, so the retried approve replays nothing. It must place the
    ticket again instead of dead-ending on "no paper ticket"."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    with _rolled_back(db_session):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        )
    assert _paper_orders(db_session) == []
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"
    assert not observe._has_pending_interrupt(saver, str(run.run_id))

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    _assert_retry_completed_with_ticket(db_session, run.run_id, expected)


def test_resume_fund_retry_without_a_fill_bar_finalises_a_consumed_gate_incomplete(
    db_session,
) -> None:
    """A consumed gate has left the Decisions queue, so a retry whose own placement
    fails (the next close is not there yet) cannot be left paused: the run is closed
    ``incomplete`` with nothing written rather than orphaned."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    with _rolled_back(db_session):
        resume_fund(
            run.run_id,
            "approve",
            session=db_session,
            checkpointer=saver,
            store=store,
            model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        )
    _drop_bars_after_asof(db_session)

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert _paper_orders(db_session) == []
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "incomplete"
    assert _positions_map(db_session) == {}


def test_fundrun_resume_retry_replaces_a_ticket_lost_with_the_rolled_back_session(
    db_session,
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )

    with _rolled_back(db_session):
        run.resume("approve")
    assert _paper_orders(db_session) == []

    run.resume("approve")

    _assert_retry_completed_with_ticket(db_session, run.run_id, expected)


def test_resume_fund_retry_places_the_ticket_once_the_fill_bar_exists(
    db_session,
) -> None:
    """An approve refused for want of a fill bar leaves the gate open. Once the next
    close is ingested the same approve must place the ticket through that gate, not
    leave the run paused for good."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    # Rolling the transaction back also restores the dropped bars: the next close has
    # been ingested by the time the adviser retries.
    with _rolled_back(db_session):
        _drop_bars_after_asof(db_session)
        with pytest.raises(RuntimeError, match="paper ticket"):
            resume_fund(
                run.run_id,
                "approve",
                session=db_session,
                checkpointer=saver,
                store=store,
                model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
            )
    _assert_gate_still_open(db_session, saver, run.run_id)

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    _assert_retry_completed_with_ticket(db_session, run.run_id, expected)


# --- the positions snapshot traces back to the ticket that produced it ---------


def _assert_positions_trace_the_ticket(db_session) -> None:
    """Every holding links the paper ticket and carries its line's base notional.

    ``shares`` stays unset: the ticket's per-line ``shares`` is base notional over the
    listing price, currency-mixed for a non-base listing, so it is not a holding size.
    """
    (order,) = _paper_orders(db_session)
    notionals = {line["ticker"]: line["notional"] for line in order.lines}
    holdings = PositionRepository(db_session).get_holdings(PORTFOLIO_ID)
    assert {p.ticker for p in holdings} == set(notionals)
    for position in holdings:
        assert position.paper_order_id == order.id
        assert position.notional == notionals[position.ticker]
        assert position.shares is None


def test_resume_fund_approve_links_the_positions_to_the_paper_ticket(
    db_session,
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    _assert_positions_trace_the_ticket(db_session)


def test_fundrun_resume_approve_links_the_positions_to_the_paper_ticket(
    db_session,
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )

    run.resume("approve")

    _assert_positions_trace_the_ticket(db_session)


# --- _finalize_hitl unit surface: order_lines enrich the snapshot ------------


def _place_ticket(db_session, weights: dict[str, float]) -> None:
    """Leave the paper ticket ``place_orders`` writes for ``weights`` on the bar."""
    OrderRepository(db_session).create(
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        weights_hash=AgentRunRepository.weights_digest(weights),
        weights=weights,
        fill_date=ASOF,
        lines=[],
        notional=100000.0,
        total_commission=0.0,
        total_slippage_cost=0.0,
    )


def _clear_risk_and_place(db_session, run_id, weights: dict[str, float]) -> None:
    """Reach the state an approved, accepted ``place_orders`` call leaves behind."""
    AgentRunRepository(db_session).append_risk_verdict(
        run_id, weights=weights, passed=True, violations=[]
    )
    _place_ticket(db_session, weights)


def test_finalize_hitl_approve_uses_order_lines_when_given(db_session) -> None:
    """``order_lines`` (the filled ticket rows) supply shares/notional to the
    snapshot; without them ``_finalize_hitl`` derives weight-only rows from
    ``weights`` (exercised by the approve tests above)."""
    audit = AgentRunRepository(db_session)
    run = audit.create_run(
        portfolio_id=PORTFOLIO_ID, asof=ASOF, seed=1, universe=[], optimizer_config={}
    )
    weights = {"AAA": 0.6, "BBB": 0.4}
    _clear_risk_and_place(db_session, run.id, weights)
    order_lines = [
        {"ticker": "AAA", "weight": 0.6, "shares": 12.0, "notional": 60000.0},
        {"ticker": "BBB", "weight": 0.4, "shares": 34.0, "notional": 40000.0},
    ]

    _finalize_hitl(
        db_session,
        run_id=run.id,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        weights=weights,
        decision="approve",
        order_lines=order_lines,
    )

    holdings = {
        p.ticker: p for p in PositionRepository(db_session).get_holdings(PORTFOLIO_ID)
    }
    assert holdings["AAA"].shares == 12.0
    assert holdings["AAA"].notional == 60000.0
    assert holdings["BBB"].weight == 0.4
    row = audit.get_run(run.id)
    assert row.status == "completed"
    assert row.weights == weights


def test_settle_hitl_approve_ignores_a_ticket_this_run_never_cleared(
    db_session,
) -> None:
    """A ticket an earlier run left on the same bar and weights is not proof that
    this run placed anything: with no risk pass on this run the approval is
    refused, so the run ends ``incomplete`` and the holdings stay untouched."""
    audit = AgentRunRepository(db_session)
    run = audit.create_run(
        portfolio_id=PORTFOLIO_ID, asof=ASOF, seed=1, universe=[], optimizer_config={}
    )
    weights = {"AAA": 0.6, "BBB": 0.4}
    _place_ticket(db_session, weights)

    _settle_hitl(
        db_session,
        run_id=run.id,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        weights=weights,
        decision="approve",
        result=None,
    )

    assert audit.get_run(run.id).status == "incomplete"
    assert _positions_map(db_session) == {}


# --- respond (barge-in): thread a free-text directive, keep the run alive ------


def _respond_decisions(db_session, run_id) -> list[AgentDecision]:
    return [
        d
        for d in _decisions(db_session, run_id)
        if d.hitl_decision and d.hitl_decision.get("decision") == "respond"
    ]


def test_resume_fund_respond_regate_marks_responded_and_stays_alive(db_session) -> None:
    """A respond whose directive drives the PM to re-gate keeps the run alive as
    ``"responded"`` (not finalised) with a fresh interrupt, still awaiting approval."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    directive = "Trim AAA before committing."

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    resume_fund(
        run.run_id,
        "respond",
        session=db_session,
        checkpointer=saver,
        store=store,
        # The re-gate model drives a second place_orders call after the respond.
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected, max_orders=2),
        pm_directive=directive,
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "responded"  # alive, NOT finalised
    assert row.finished_at is None
    assert _paper_orders(db_session) == []  # nothing committed at the re-gate
    # the re-gate left a live interrupt, so the run stays approvable
    assert observe._has_pending_interrupt(saver, str(run.run_id))
    # the barge-in is recorded for the transcript/audit, carrying the directive
    responded = _respond_decisions(db_session, run.run_id)
    assert len(responded) == 1
    assert responded[0].hitl_decision["message"] == directive


def test_resume_fund_respond_terminal_finalises_incomplete(db_session) -> None:
    """A respond the PM does not re-gate on reaches a terminal result with no ticket
    through the gate → finalised ``"incomplete"`` (never ``"completed"``)."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    resume_fund(
        run.run_id,
        "respond",
        session=db_session,
        checkpointer=saver,
        store=store,
        # The default model does not re-gate, so the run ends terminally.
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        pm_directive="Looks fine, proceed.",
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "incomplete"
    assert _paper_orders(db_session) == []
    assert not observe._has_pending_interrupt(saver, str(run.run_id))
    assert len(_respond_decisions(db_session, run.run_id)) == 1


def test_resume_fund_respond_with_a_broken_place_orders_call_is_not_no_order(
    db_session,
) -> None:
    """A respond whose re-gating ``place_orders`` call is cut off by a provider output
    cap never reaches a gate: the run finalises incomplete as ``"output_truncated"``,
    not the generic ``"no_order"`` a blocking risk verdict gets."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    resume_fund(
        run.run_id,
        "respond",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=BrokenPlaceOrdersFundModel(
            universe=UNIVERSE,
            weights=expected,
            max_orders=2,
            stop_metadata={"stop_reason": "max_tokens"},
        ),
        pm_directive="Trim AAA before committing.",
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "incomplete"
    reasons = [
        d.hitl_decision["reason"]
        for d in _decisions(db_session, run.run_id)
        if d.hitl_decision and "reason" in d.hitl_decision
    ]
    assert reasons == ["output_truncated"]


def test_resume_fund_respond_threads_directive_as_place_orders_tool_message(
    db_session,
) -> None:
    """The free text threads back to the PM as a successful ``place_orders`` tool
    result (SPEC §3) — the mechanism that lets the gate continue on a barge-in.

    Asserted on the terminal path (the PM does not re-gate) so the run ends cleanly and
    the final checkpoint carries the full merged ``messages`` channel — a re-gate
    interrupt checkpoint only holds the channels its step wrote.
    """
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    directive = "Hold CCC; revisit next bar."

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    result = resume_fund(
        run.run_id,
        "respond",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        pm_directive=directive,
    )

    tool_results = [
        m
        for m in result.get("messages", [])
        if getattr(m, "type", None) == "tool"
        and getattr(m, "name", None) == "place_orders"
    ]
    assert tool_results
    assert any(directive in str(m.content) for m in tool_results)
    assert all(getattr(m, "status", None) == "success" for m in tool_results)


def test_resume_command_frames_a_respond_directive_as_an_unexecuted_gate() -> None:
    """The PM reads the respond as a ``place_orders`` tool result, so the text must
    say nothing was placed — a bare sentence reads like a successful order."""
    command = _resume_command("respond", "cap each name at 5%")

    message = command.resume["decisions"][0]["message"]
    assert "cap each name at 5%" in message
    assert message.startswith("ADVISER DIRECTIVE")
    assert "NOT executed" in message
    assert "no ticket" in message
    assert "place_orders again" in message


def test_resume_fund_respond_tells_the_pm_that_no_order_was_placed(
    db_session,
) -> None:
    """What the PM actually sees on its next turn: the directive, framed as a
    non-execution, not a bare success result."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    result = resume_fund(
        run.run_id,
        "respond",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
        pm_directive="cap each name at 5%",
    )

    content = " ".join(
        str(m.content)
        for m in result.get("messages", [])
        if getattr(m, "type", None) == "tool"
        and getattr(m, "name", None) == "place_orders"
    )
    assert "cap each name at 5%" in content
    assert "NOT executed" in content


def test_resume_run_routes_respond_with_directive_to_the_fund_gate(db_session) -> None:
    """``resume_run`` threads ``pm_directive`` through to ``resume_fund`` for a
    ``"fund"``-step run — the single entry point the CLI / TUI drive."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    directive = "Rebalance more conservatively."

    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    resume_run(
        run.run_id,
        "respond",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected, max_orders=2),
        pm_directive=directive,
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "responded"
    responded = _respond_decisions(db_session, run.run_id)
    assert responded and responded[0].hitl_decision["message"] == directive


def test_resume_run_rejects_respond_at_the_profiler_gate(db_session) -> None:
    """``respond`` is a ``place_orders``-only barge-in; the profiler ``save_profile``
    gate keeps approve/reject, so routing a respond there raises."""
    run = AgentRunRepository(db_session).create_run(
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        seed=1,
        universe=[],
        optimizer_config={"step": "profiler"},
    )
    with pytest.raises(ValueError, match="approve"):
        resume_run(
            run.id,
            "respond",
            session=db_session,
            checkpointer=MemorySaver(),
            store=_store_with_cs(),
            model=ScriptedFundModel(),
            pm_directive="whatever",
        )


def test_fundrun_resume_respond_marks_responded(db_session) -> None:
    """The live in-process ``FundRun.resume`` honours ``respond`` the same way as the
    rebuild path, so the two cannot diverge."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    directive = "Trim AAA to 50%."

    run = run_fund(
        ScriptedFundModel(universe=UNIVERSE, weights=expected, max_orders=2),
        make_mandate(),
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )
    run.resume("respond", pm_directive=directive)

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "responded"
    assert observe._has_pending_interrupt(saver, str(run.run_id))
    responded = _respond_decisions(db_session, run.run_id)
    assert responded and responded[0].hitl_decision["message"] == directive


# --- the allocator tools are bound to the mandate's base currency --------------


def _spy_run_contexts(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record every ``RunContext`` the graph hands to ``build_fund_agent``."""
    seen: list = []
    original = graph_mod.build_fund_agent

    def _spy(*args, **kwargs):
        seen.append(kwargs["ctx"])
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_mod, "build_fund_agent", _spy)
    return seen


def test_run_fund_binds_the_mandate_base_currency_to_the_run_context(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_panel(db_session)
    contexts = _spy_run_contexts(monkeypatch)
    mandate = make_mandate().model_copy(update={"base_currency": "CHF"})

    run_fund(
        ScriptedFundModel(universe=UNIVERSE, weights=expected_weights(db_session)),
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=MemorySaver(),
        store=_store_with_cs(),
    )

    assert [c.base_currency for c in contexts] == ["CHF"]


def test_resume_fund_binds_the_persisted_mandate_base_currency(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    MandateRepository(db_session).upsert(
        make_mandate().model_copy(update={"base_currency": "CHF"})
    )
    contexts = _spy_run_contexts(monkeypatch)

    resume_fund(
        run.run_id,
        "reject",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert [c.base_currency for c in contexts] == ["CHF"]


# --- a mandate refusal inside place_orders must not finalise the run completed ----


_SECTORS = {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}


def _reprofile_with_energy_exclusion(db_session, store: InMemoryStore) -> None:
    """The client is re-profiled between the proposal and the adviser's approval."""
    classify_instruments(db_session, _SECTORS)
    put_constraint_set(
        store,
        make_energy_exclusion_constraint_set(),
        store_key=settings.constraint_set_store_key,
    )


def _assert_nothing_booked(db_session, run_id) -> None:
    """No ticket, no held book, no journal row: the refusal left the books untouched."""
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}
    assert db_session.query(PortfolioJournal).count() == 0
    assert AgentRunRepository(db_session).get_run(run_id).weights in (
        None,
        {},
    )  # the refused proposal is not recorded as the run's allocation


def test_resume_fund_approve_refused_by_the_mandate_does_not_complete_the_run(
    db_session,
) -> None:
    """``place_orders`` refuses weights that breach the re-profiled mandate; the run
    must end ``incomplete``, not record the refused weights as the held book."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _reprofile_with_energy_exclusion(db_session, store)

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "incomplete"
    _assert_nothing_booked(db_session, run.run_id)
    reasons = [
        d.hitl_decision.get("reason")
        for d in _decisions(db_session, run.run_id)
        if d.hitl_decision and d.hitl_decision.get("decision") == "incomplete"
    ]
    assert reasons == ["place_orders_failed"]


def test_fundrun_resume_approve_refused_by_the_mandate_does_not_complete_the_run(
    db_session,
) -> None:
    """The live in-process handle settles a refused approve exactly like the rebuild
    path, so the two cannot diverge."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )
    _reprofile_with_energy_exclusion(db_session, store)

    run.resume("approve")

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "incomplete"
    _assert_nothing_booked(db_session, run.run_id)


def test_refused_approve_that_the_pm_re_gates_keeps_the_run_alive(db_session) -> None:
    """If the PM answers the refusal by re-gating, the run stays awaiting approval
    (``responded``) rather than being finalised under a live checkpoint."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected, max_orders=2),
        db_session,
        saver,
        store,
    )
    _reprofile_with_energy_exclusion(db_session, store)

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected, max_orders=2),
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "responded"
    assert observe._has_pending_interrupt(saver, str(run.run_id))
    assert _paper_orders(db_session) == []
    assert _positions_map(db_session) == {}


def test_a_placed_ticket_still_completes_the_run_after_the_refusal_guard(
    db_session,
) -> None:
    """Regression guard: an approve whose ticket landed finalises ``completed``."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
    )

    resume_fund(
        run.run_id,
        "approve",
        session=db_session,
        checkpointer=saver,
        store=store,
        model=ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )

    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"
    assert len(_paper_orders(db_session)) == 1
    assert _positions_map(db_session) == expected


class _ReoptimizingModel(ScriptedFundModel):
    """After a respond barge-in the PM re-delegates the allocator on a narrower
    universe and re-gates, so the run's latest audited optimizer weights change."""

    narrow_universe: list[str] = Field(default_factory=list)

    def _pm_turn(self, messages: list[BaseMessage]) -> AIMessage:
        n_task = _count_tool_calls(messages, "task")
        n_orders = _count_tool_calls(messages, "place_orders")
        if n_orders == 1 and n_task == 4:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Re-run the allocator NARROW as directed.",
                            "subagent_type": "allocator",
                        },
                        "id": "call_task_narrow",
                    }
                ],
            )
        if n_orders == 1 and n_task == 5:
            # A risk pass is bound to one weight vector, so the narrower proposal
            # must be judged again before place_orders will commit it.
            return _task_call("risk", 5)
        if n_orders == 1 and n_task == 6:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "place_orders",
                        "args": {"weights": self.weights},
                        "id": "call_place_orders_regate",
                    }
                ],
            )
        return super()._pm_turn(messages)

    def _allocator_turn(self, messages: list[BaseMessage]) -> AIMessage:
        if _count_tool_calls(messages, "optimize_portfolio") == 0:
            narrow = any(
                "NARROW" in str(m.content) for m in messages if m.type == "human"
            )
            universe = self.narrow_universe if narrow else self.universe
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "optimize_portfolio",
                        "args": {"universe": universe},
                        "id": "call_optimize",
                    }
                ],
            )
        return super()._allocator_turn(messages)


def test_fundrun_resume_approve_finalises_with_weights_reoptimised_after_respond(
    db_session,
) -> None:
    """The live handle's weights are captured before the pause; a respond that makes
    the PM re-optimise changes the audited weights the ticket is placed from, so the
    approve must finalise with those, not the stale pre-pause ones."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    model = _ReoptimizingModel(
        universe=UNIVERSE,
        weights=expected,
        narrow_universe=["AAA", "BBB"],
        max_orders=2,
    )
    run = _run_to_gate(model, db_session, MemorySaver(), _store_with_cs())
    assert set(run.weights) == set(UNIVERSE)

    run.resume("respond", pm_directive="Drop CCC and re-optimise.")
    run.resume("approve")

    row = AgentRunRepository(db_session).get_run(run.run_id)
    orders = _paper_orders(db_session)
    assert len(orders) == 1
    assert set(orders[0].weights) == {"AAA", "BBB"}
    assert row.status == "completed"
    assert row.weights == orders[0].weights
    assert _positions_map(db_session) == orders[0].weights


def test_fundrun_resume_refuses_a_run_finalised_since_the_gate_opened(
    db_session,
) -> None:
    """An external freeze finalises the row without resolving the checkpoint; the
    in-process handle must not blindly re-drive ``place_orders`` against it."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )
    AgentRunRepository(db_session).finalize_run(
        run.run_id, weights={}, status="rejected"
    )

    with pytest.raises(ValueError, match="already finalised"):
        run.resume("approve")

    assert _paper_orders(db_session) == []
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "rejected"
