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

from fund import observe
from fund.agents import graph as graph_mod
from fund.agents.graph import (
    FundRun,
    _finalize_hitl,
    resume_fund,
    resume_run,
    run_fund,
)
from fund.audit import (
    AgentRunRepository,
    MandateRepository,
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


# --- approve with no fill bar: a failed place_orders is never "completed" -----


def _drop_bars_after_asof(db_session) -> None:
    """Remove every priced bar past ``ASOF`` so ``place_orders`` has no fill bar."""
    from portopt_db.models.market_data.yfinance_data import PriceHistory

    db_session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
    db_session.flush()


def test_resume_fund_approve_without_fill_bar_finalises_incomplete(db_session) -> None:
    """``place_orders`` reports a missing fill bar as an ``{ok: false}`` envelope
    (it never raises), so an approve that placed no ticket must not stamp the run
    ``completed`` nor overwrite ``positions`` with weights no order backs."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    saver = MemorySaver()
    store = _store_with_cs()
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected), db_session, saver, store
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
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "incomplete"
    assert _positions_map(db_session) == {}


def test_fundrun_resume_approve_without_fill_bar_finalises_incomplete(
    db_session,
) -> None:
    """The in-process handle shares the same settle step as the rebuild path."""
    seed_panel(db_session)
    expected = expected_weights(db_session)
    run = _run_to_gate(
        ScriptedFundModel(universe=UNIVERSE, weights=expected),
        db_session,
        MemorySaver(),
        _store_with_cs(),
    )
    _drop_bars_after_asof(db_session)

    run.resume("approve")

    assert _paper_orders(db_session) == []
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "incomplete"
    assert _positions_map(db_session) == {}


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
