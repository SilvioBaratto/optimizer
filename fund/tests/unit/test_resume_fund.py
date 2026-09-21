"""Task 6 — ``resume_fund`` + ``_finalize_hitl``: rebuild-to-resume + parity.

``run_fund`` drives the PM deep agent to the ``place_orders`` HITL gate, marking
the run ``paused`` and persisting ``thread_id = str(run_id)`` on the row. A human
observer then resumes it **without the live handle**: ``resume_fund`` rebuilds the
PM agent against the *same* checkpointer + thread in a **fresh agent instance** and
commits (approve) or discards (reject) the optimizer's ticket. Both the live-handle
``FundRun.resume`` and the rebuilt ``resume_fund`` delegate their finalisation to the
one shared ``_finalize_hitl``, so the two paths cannot diverge — asserted directly by
a parity test that resumes two identical runs one each way and compares DB state.

Zero live LLM, zero network: ``ScriptedFundModel`` + ``MemorySaver`` +
``InMemoryStore`` + the in-memory ``db_session`` (SPEC §5 / §4d).
"""

from __future__ import annotations

import uuid

import pytest
from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    ScriptedFundModel,
    expected_weights,
    make_constraint_set,
    make_mandate,
    seed_panel,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision, PaperOrder

from fund.agents.graph import FundRun, _finalize_hitl, resume_fund, run_fund
from fund.audit import AgentRunRepository, PositionRepository, put_constraint_set
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


# --- run_fund persists the per-run thread + marks paused (the Phase-7 touch) --


def test_run_fund_persists_thread_id_and_marks_paused(db_session) -> None:
    seed_panel(db_session)
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected_weights(db_session))

    run = _run_to_gate(model, db_session, MemorySaver(), _store_with_cs())

    row = AgentRunRepository(db_session).get_run(run.run_id)
    # thread_id defaults to str(run_id) (was str(portfolio_id)) and is persisted.
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
    # positions snapshot upserted from the optimizer weights.
    assert _positions_map(db_session) == expected
    # executor execution decision + HITL approve recorded (via the resumed tool).
    executor = [
        d
        for d in _decisions(db_session, run.run_id)
        if d.agent == "executor" and d.step == "place_orders"
    ]
    assert len(executor) == 1
    assert executor[0].hitl_decision == {"decision": "approve"}


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
    # no positions are written on reject (the snapshot stays empty).
    assert _positions_map(db_session) == {}
    assert any(
        d.hitl_decision and d.hitl_decision.get("decision") == "reject"
        for d in _decisions(db_session, run.run_id)
    )


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


# --- _finalize_hitl unit surface: order_lines enrich the snapshot ------------


def test_finalize_hitl_approve_uses_order_lines_when_given(db_session) -> None:
    """``order_lines`` (the filled ticket rows) supply shares/notional to the
    snapshot; without them ``_finalize_hitl`` derives weight-only rows from
    ``weights`` (exercised by the approve tests above)."""
    audit = AgentRunRepository(db_session)
    run = audit.create_run(
        portfolio_id=PORTFOLIO_ID, asof=ASOF, seed=1, universe=[], optimizer_config={}
    )
    weights = {"AAA": 0.6, "BBB": 0.4}
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
