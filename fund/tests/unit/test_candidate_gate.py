"""The rebalance gate carries four candidates and the adviser names the one placed.

A round computes the candidate set, the ``place_orders`` gate shows all of it, and an
approve must name exactly one candidate: no default, no automatic choice, and paper
orders only from that candidate's weights. Every refusal writes nothing, so the gate
stays open for a corrected decision.

Zero live LLM, zero network: ``ScriptedFundModel`` + ``MemorySaver`` +
``InMemoryStore`` + the in-memory ``db_session``.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

import pytest
from _fund_fakes import (
    ASOF,
    MEAN_RISK,
    PORTFOLIO_ID,
    UNIVERSE,
    ScriptedFundModel,
    classify_instruments,
    make_constraint_set,
    make_energy_exclusion_constraint_set,
    make_mandate,
    mark_candidates_infeasible,
    seed_panel,
)
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision, HumanAuditLog, PaperOrder

import fund.tools as fund_tools
from fund import observe
from fund.actions import apply_action
from fund.agents.graph import FundRun, resume_fund, resume_run, run_fund
from fund.agents.graph_assembly import _general_purpose_subagent
from fund.agents.graph_stream import _UNGATED_END_TEXT
from fund.agents.toolsets import (
    TOOLS_BY_AGENT,
    RunContext,
    bind_toolset,
    place_audited_orders,
)
from fund.audit import AgentRunRepository, put_constraint_set
from fund.audit.orders_repository import weights_hash
from fund.config import settings
from fund.schemas.candidates import CANDIDATE_IDS, Candidate

# The gate shows these fields of every candidate, in id order.
_GATE_FIELDS = frozenset(
    {
        "candidate_id",
        "label",
        "feasible",
        "weights",
        "metrics",
        "constraint_check",
        "breaches",
    }
)
_UNKNOWN_CANDIDATE = "max_sharpe"
_MARKED_INFEASIBLE = "hrp"
_RETRY_CANDIDATE = "min_risk"
_AFTER_RESPOND_CANDIDATE = "hrp"
_DIRECTIVE = "re-run the allocator with the same universe"
# A long-only edit the fixture's mandate admits: fully invested, no position over
# the default cap, and the oscillating panel keeps its one-year loss well inside nu2.
_EDITED = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
_ORDERS_AFTER_RESPOND = 2
_ENERGY_SECTORS = {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
_PROFILER_STEP = {"step": "profiler"}


# --- helpers ----------------------------------------------------------------


def _store_with_cs() -> InMemoryStore:
    store = InMemoryStore()
    put_constraint_set(
        store, make_constraint_set(), store_key=settings.constraint_set_store_key
    )
    return store


def _model(**kwargs: Any) -> ScriptedFundModel:
    return ScriptedFundModel(universe=UNIVERSE, **kwargs)


def _paused_run(
    db_session, saver: MemorySaver, store: InMemoryStore, **kwargs: Any
) -> FundRun:
    seed_panel(db_session)
    return run_fund(
        _model(**kwargs),
        make_mandate(),
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )


def _decisions(db_session, run_id) -> list[AgentDecision]:
    return db_session.query(AgentDecision).filter(AgentDecision.run_id == run_id).all()


def _audit_rows(db_session, run_id) -> list[HumanAuditLog]:
    return db_session.query(HumanAuditLog).filter(HumanAuditLog.run_id == run_id).all()


def _snapshot(db_session, run_id) -> tuple[int, int, int]:
    """How many decisions, audit rows and paper orders exist: a refusal moves none."""
    return (
        len(_decisions(db_session, run_id)),
        len(_audit_rows(db_session, run_id)),
        db_session.query(PaperOrder).count(),
    )


def _assert_gate_still_open(db_session, saver, run_id, before) -> None:
    assert _snapshot(db_session, run_id) == before
    assert AgentRunRepository(db_session).get_run(run_id).status == "paused"
    assert observe._has_pending_interrupt(saver, str(run_id))


def _approve(db_session, saver, store, run_id, **kwargs: Any) -> Any:
    return apply_action(
        "approve",
        session=db_session,
        run_id=run_id,
        checkpointer=saver,
        store=store,
        model=_model(),
        **kwargs,
    )


def _resume(db_session, saver, store, run_id, decision, **kwargs: Any) -> Any:
    return resume_fund(
        run_id,
        decision,
        session=db_session,
        checkpointer=saver,
        store=store,
        model=_model(**kwargs.pop("model_kwargs", {})),
        **kwargs,
    )


def _candidate(db_session, run_id, candidate_id: str) -> Candidate:
    candidate_set = observe.candidate_set(db_session, run_id)
    assert candidate_set is not None
    chosen = candidate_set.candidate(candidate_id)
    assert chosen is not None
    return chosen


def _only_order(db_session) -> PaperOrder:
    (order,) = db_session.query(PaperOrder).all()
    return order


class _AllocatorSkipsCandidatesModel(ScriptedFundModel):
    """An allocator that reports without computing candidates."""

    def _allocator_turn(self, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="allocator report: nothing computed.")


@contextlib.contextmanager
def _rolled_back(db_session):
    """The caller's transaction rolled back after the block, as the CLI and TUI do
    when an approve raises; a savepoint stands in for it inside ``db_session``."""
    savepoint = db_session.begin_nested()
    try:
        yield
    finally:
        savepoint.rollback()


def _record_choice(
    db_session, run_id, candidate_id: str, *, with_verdict: bool = True
) -> None:
    """Leave an adviser choice of ``candidate_id`` in force, with its verdict unless
    ``with_verdict`` is false."""
    candidate_set = observe.candidate_set(db_session, run_id)
    chosen = candidate_set.candidate(candidate_id)
    repo = AgentRunRepository(db_session)
    if with_verdict:
        repo.append_risk_verdict(
            run_id, weights=chosen.weights, passed=True, violations=[]
        )
    repo.append_candidate_selection(
        run_id,
        candidate_id=candidate_id,
        candidate_set_hash=candidate_set.digest(),
        weights=dict(chosen.weights),
        metrics=chosen.metrics.model_dump(),
        settings=chosen.settings,
    )


def _legacy_paused_run(db_session, optimizer_config: dict[str, Any]) -> Any:
    """A paused run with an optimizer proposal but no candidate set, as a run left
    at its gate before candidates existed (or a profiler gate) is stored."""
    repo = AgentRunRepository(db_session)
    run = repo.create_run(
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        seed=None,
        universe=UNIVERSE,
        optimizer_config=optimizer_config,
    )
    repo.append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response='{"weights": {"AAA": 0.5, "BBB": 0.5}}',
    )
    repo.mark_paused(run.id)
    return run


# --- the gate ----------------------------------------------------------------


@pytest.mark.criterion("REQ-FUND-047")
def test_gate_payload_carries_every_candidate_and_the_selectable_ids(
    db_session,
) -> None:
    """Validates fund.agents.toolsets.build_gate_candidates_middleware: the
    ``place_orders`` gate the adviser reviews carries the run's whole candidate set
    (its hash, date and universe), every candidate in id order with its label,
    feasibility, weights, metrics, constraint check and breaches, and the selectable
    ids; never the weights the PM typed."""
    run = _paused_run(db_session, MemorySaver(), _store_with_cs())

    stored = observe.candidate_set(db_session, run.run_id)
    (request,) = run.interrupt["action_requests"]
    args = request["args"]
    assert run.status == "paused"
    assert request["name"] == "place_orders"
    assert args["candidate_set"] == {
        "hash": stored.digest(),
        "asof": ASOF.isoformat(),
        "universe": list(stored.universe),
    }
    assert [entry["candidate_id"] for entry in args["candidates"]] == list(
        CANDIDATE_IDS
    )
    for entry, candidate in zip(args["candidates"], stored.candidates, strict=True):
        assert entry == candidate.model_dump(mode="json", include=set(_GATE_FIELDS))
    assert args["selectable"] == list(stored.selectable_ids())
    assert args["selectable"]
    assert "weights" not in args


@pytest.mark.criterion("REQ-FUND-047")
@pytest.mark.parametrize(
    ("scenario", "reason"),
    [("no_set", "no_candidate_set"), ("none_feasible", "no_feasible_candidate")],
)
def test_round_with_no_feasible_candidate_ends_incomplete_without_a_gate(
    db_session, monkeypatch: pytest.MonkeyPatch, scenario: str, reason: str
) -> None:
    """Validates fund.agents.graph._settle_round: a round whose PM reaches
    ``place_orders`` with no candidate set, or with no feasible candidate, is
    finalised incomplete with that reason instead of opening a gate the adviser could
    only reject."""
    seed_panel(db_session)
    if scenario == "none_feasible":
        mark_candidates_infeasible(monkeypatch, *CANDIDATE_IDS)
    model_type = (
        _AllocatorSkipsCandidatesModel if scenario == "no_set" else ScriptedFundModel
    )

    run = run_fund(
        model_type(universe=UNIVERSE),
        make_mandate(),
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=MemorySaver(),
        store=_store_with_cs(),
    )

    assert run.status == "incomplete"
    assert run.interrupt is None
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "incomplete"
    assert {"decision": "incomplete", "reason": reason} in [
        decision.hitl_decision for decision in _decisions(db_session, run.run_id)
    ]
    assert db_session.query(PaperOrder).count() == 0
    assert reason in _UNGATED_END_TEXT


# --- refusals: nothing is written ------------------------------------------------


@pytest.mark.criterion("REQ-FUND-048")
@pytest.mark.parametrize("entry", ["apply_action", "resume_fund", "fund_run"])
def test_approve_without_a_candidate_is_refused_and_the_gate_stays_open(
    db_session, entry: str
) -> None:
    """Validates fund.agents.graph_candidates.resolve_approval through every approve
    entry point (fund.actions.apply_action, fund.agents.graph.resume_fund and
    FundRun.resume): approving a rebalance gate without naming a candidate is refused
    with the ids to choose from, writes no decision, audit row or order, and leaves
    the gate open."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    before = _snapshot(db_session, run.run_id)

    with pytest.raises(ValueError, match="requires a candidate id") as refused:
        if entry == "apply_action":
            _approve(db_session, saver, store, run.run_id)
        elif entry == "resume_fund":
            _resume(db_session, saver, store, run.run_id, "approve")
        else:
            run.resume("approve")

    assert all(candidate_id in str(refused.value) for candidate_id in CANDIDATE_IDS)
    _assert_gate_still_open(db_session, saver, run.run_id, before)


@pytest.mark.criterion("REQ-FUND-048")
@pytest.mark.parametrize(
    ("candidate_id", "message"),
    [
        (_UNKNOWN_CANDIDATE, "not in the run's candidate set"),
        (_MARKED_INFEASIBLE, "is not feasible"),
    ],
)
def test_approve_with_an_unknown_or_infeasible_candidate_is_refused(
    db_session, monkeypatch: pytest.MonkeyPatch, candidate_id: str, message: str
) -> None:
    """Validates fund.agents.graph_candidates.resolve_approval: an approve naming an
    id outside the set, or a candidate the set marks infeasible, is refused naming
    it, and nothing is written."""
    mark_candidates_infeasible(monkeypatch, _MARKED_INFEASIBLE)
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    before = _snapshot(db_session, run.run_id)

    with pytest.raises(ValueError, match=message) as refused:
        _approve(db_session, saver, store, run.run_id, candidate_id=candidate_id)

    assert candidate_id in str(refused.value)
    _assert_gate_still_open(db_session, saver, run.run_id, before)


@pytest.mark.criterion("REQ-FUND-048")
def test_run_without_a_candidate_set_cannot_be_approved(db_session) -> None:
    """Validates fund.actions._ticketed_weights: a run left paused before candidates
    existed has no set to choose from, so an approve with or without an id is
    refused (telling the adviser to reject it) and writes nothing, while a reject
    still closes it."""
    run = _legacy_paused_run(db_session, {"step": "fund"})
    before = _snapshot(db_session, run.id)

    for candidate_id in (None, MEAN_RISK):
        with pytest.raises(ValueError, match="has no candidate set; reject it"):
            apply_action(
                "approve", session=db_session, run_id=run.id, candidate_id=candidate_id
            )
        assert _snapshot(db_session, run.id) == before

    apply_action("reject", session=db_session, run_id=run.id)
    assert AgentRunRepository(db_session).get_run(run.id).status == "rejected"


@pytest.mark.criterion("REQ-FUND-048")
@pytest.mark.parametrize("action", ["reject", "respond", "killswitch"])
def test_candidate_id_on_reject_or_a_profiler_gate_is_refused(
    db_session, action: str
) -> None:
    """Validates fund.actions._require_resume_args and
    fund.agents.graph._resume_profiler_gate: a candidate id belongs to the approve of
    a rebalance gate only, so naming one on a reject, respond or killswitch, or on
    the approve of a profiler gate, is refused and writes nothing."""
    run = _legacy_paused_run(db_session, {"step": "fund"})
    before = _snapshot(db_session, run.id)
    with pytest.raises(ValueError, match="candidate id"):
        apply_action(
            action,
            session=db_session,
            run_id=run.id,
            pm_directive=_DIRECTIVE,
            candidate_id=MEAN_RISK,
        )
    assert _snapshot(db_session, run.id) == before

    profiler_run = _legacy_paused_run(db_session, _PROFILER_STEP)
    profiler_before = _snapshot(db_session, profiler_run.id)
    with pytest.raises(ValueError, match=r"inputs \(profile\) gate"):
        apply_action(
            "approve",
            session=db_session,
            run_id=profiler_run.id,
            candidate_id=MEAN_RISK,
        )
    with pytest.raises(ValueError, match=r"inputs \(profile\) gate"):
        resume_run(
            profiler_run.id,
            "approve",
            session=db_session,
            checkpointer=MemorySaver(),
            store=None,
            model=None,
            candidate_id=MEAN_RISK,
        )
    assert _snapshot(db_session, profiler_run.id) == profiler_before


@pytest.mark.criterion("REQ-FUND-049")
def test_approve_rechecks_the_chosen_candidate_against_the_current_mandate(
    db_session,
) -> None:
    """Validates fund.agents.graph_candidates.record_candidate_selection: the named
    candidate is judged again under the mandate in force at approve time, so a
    re-profile that now excludes a held sector refuses it with the violation before
    the graph resumes, and nothing (no choice, no verdict, no order) is recorded."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    classify_instruments(db_session, _ENERGY_SECTORS)
    put_constraint_set(
        store,
        make_energy_exclusion_constraint_set(),
        store_key=settings.constraint_set_store_key,
    )
    before = _snapshot(db_session, run.run_id)

    with pytest.raises(ValueError, match="breaches the current constraints") as refused:
        _resume(db_session, saver, store, run.run_id, "approve", candidate_id=MEAN_RISK)

    assert "esg" in str(refused.value).lower()
    _assert_gate_still_open(db_session, saver, run.run_id, before)


# --- a valid choice places exactly that candidate ----------------------------------


@pytest.mark.criterion("REQ-FUND-049")
@pytest.mark.parametrize("candidate_id", CANDIDATE_IDS)
def test_approve_places_orders_from_exactly_the_chosen_candidate(
    db_session, candidate_id: str
) -> None:
    """Validates fund.agents.graph_candidates.record_candidate_selection and
    fund.agents.toolsets._place_orders_impl: approving with a candidate id records
    the adviser's choice and its fresh verdict, and the paper ticket, the finalised
    run and the approve's audit row all carry exactly that candidate's weights."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    candidate_set = observe.candidate_set(db_session, run.run_id)
    chosen = candidate_set.candidate(candidate_id)

    _approve(db_session, saver, store, run.run_id, candidate_id=candidate_id)

    assert _only_order(db_session).weights == chosen.weights
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert row.weights == chosen.weights
    assert observe.selected_candidate_id(row) == candidate_id
    verdict = AgentRunRepository(db_session).risk_gate(run.run_id, chosen.weights)
    assert verdict["status"] == "passed"
    assert verdict["source"] == "candidate_selection"
    (audit,) = _audit_rows(db_session, run.run_id)
    assert audit.target == "weights"
    assert audit.new_value == {
        "candidate_id": candidate_id,
        "weights": chosen.weights,
        "weights_hash": weights_hash(chosen.weights),
        "candidate_set_hash": candidate_set.digest(),
    }


@pytest.mark.criterion("REQ-FUND-049")
def test_retried_approve_of_a_consumed_gate_places_the_named_candidate(
    db_session,
) -> None:
    """Validates fund.agents.graph.resume_fund on a consumed gate: when the first
    approve's transaction is lost after ``place_orders`` ran, the retry must name a
    candidate again and the replacement ticket is placed from that candidate."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    with _rolled_back(db_session):
        _resume(db_session, saver, store, run.run_id, "approve", candidate_id=MEAN_RISK)
    assert not observe._has_pending_interrupt(saver, str(run.run_id))
    assert db_session.query(PaperOrder).count() == 0

    with pytest.raises(ValueError, match="requires a candidate id"):
        _resume(db_session, saver, store, run.run_id, "approve")
    _resume(
        db_session, saver, store, run.run_id, "approve", candidate_id=_RETRY_CANDIDATE
    )

    retried = _candidate(db_session, run.run_id, _RETRY_CANDIDATE)
    assert _only_order(db_session).weights == retried.weights
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert row.weights == retried.weights


# --- place_orders needs the adviser's choice ------------------------------------------


@pytest.mark.criterion("REQ-FUND-050")
def test_place_orders_without_an_adviser_choice_places_nothing(db_session) -> None:
    """Validates fund.agents.toolsets._place_orders_impl: with a candidate set but no
    adviser choice, or with a choice an agent's newer set superseded, both the gated
    tool and the out-of-graph placement refuse and write no ticket."""
    run = _paused_run(db_session, MemorySaver(), _store_with_cs())
    tools = {tool.name: tool for tool in bind_toolset("orchestrator", run.ctx)}

    for refused in (place_audited_orders(run.ctx), tools["place_orders"].invoke({})):
        assert refused["ok"] is False
        assert "no candidate has been chosen for this run" in refused["error"]

    _record_choice(db_session, run.run_id, MEAN_RISK)
    stored = observe.candidate_set(db_session, run.run_id)
    AgentRunRepository(db_session).append_candidate_set(
        run.run_id, constraint_set=None, payload=stored.model_dump(mode="json")
    )
    superseded = place_audited_orders(run.ctx)
    assert superseded["ok"] is False
    assert "no candidate has been chosen for this run" in superseded["error"]
    assert db_session.query(PaperOrder).count() == 0


@pytest.mark.criterion("REQ-FUND-050")
def test_place_orders_refuses_a_choice_without_a_passing_verdict(db_session) -> None:
    """Validates fund.agents.toolsets._place_orders_impl: a choice in force whose
    weights carry no passing risk verdict (one written outside the adviser's approve,
    which records the verdict first) places nothing."""
    run = _paused_run(db_session, MemorySaver(), _store_with_cs())
    _record_choice(db_session, run.run_id, MEAN_RISK, with_verdict=False)

    refused = place_audited_orders(run.ctx)

    assert refused["ok"] is False
    assert "risk gate has not passed" in refused["error"]
    assert db_session.query(PaperOrder).count() == 0


@pytest.mark.criterion("REQ-FUND-050")
def test_edit_still_approves_without_a_candidate_id(db_session) -> None:
    """Validates fund.actions._apply_edit with fund.agents.graph_candidates: weights
    the adviser types over the candidate set's universe are an adviser choice newer
    than the set, so the edit approves the gate without a candidate id and the
    ticket carries the typed weights."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)

    apply_action(
        "edit",
        session=db_session,
        run_id=run.run_id,
        checkpointer=saver,
        store=store,
        model=_model(),
        new_value=_EDITED,
    )

    assert _only_order(db_session).weights == _EDITED
    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "completed"
    assert row.weights == _EDITED


@pytest.mark.criterion("REQ-FUND-050")
def test_a_respond_requires_a_fresh_choice(db_session) -> None:
    """Validates fund.audit.repository.current_adviser_choice through
    fund.agents.graph.resume_fund: a respond after which the PM computes a newer
    candidate set supersedes any earlier choice, so nothing can be placed and an
    approve must name a candidate of the newer set again."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    _record_choice(db_session, run.run_id, MEAN_RISK)

    _resume(
        db_session,
        saver,
        store,
        run.run_id,
        "respond",
        pm_directive=_DIRECTIVE,
        model_kwargs={"rerun_after_respond": True, "max_orders": _ORDERS_AFTER_RESPOND},
    )

    repo = AgentRunRepository(db_session)
    assert repo.get_run(run.run_id).status == "responded"
    assert repo.latest_selected_weights(run.run_id) == {}
    refused = place_audited_orders(run.ctx)
    assert "no candidate has been chosen for this run" in refused["error"]
    with pytest.raises(ValueError, match="requires a candidate id"):
        _resume(db_session, saver, store, run.run_id, "approve")

    _resume(
        db_session,
        saver,
        store,
        run.run_id,
        "approve",
        candidate_id=_AFTER_RESPOND_CANDIDATE,
        model_kwargs={"max_orders": _ORDERS_AFTER_RESPOND},
    )
    chosen = _candidate(db_session, run.run_id, _AFTER_RESPOND_CANDIDATE)
    assert _only_order(db_session).weights == chosen.weights
    assert repo.get_run(run.run_id).status == "completed"


# --- no agent path can choose ---------------------------------------------------------


@pytest.mark.criterion("REQ-FUND-051")
def test_general_purpose_subagent_binds_no_fund_tool() -> None:
    """Validates fund.agents.graph_assembly._general_purpose_subagent: the stock
    helper the PM may delegate to carries an explicit empty tool list, so it cannot
    inherit ``place_orders`` or any other fund tool from the PM."""
    spec = _general_purpose_subagent(settings)

    assert spec["name"] == "general-purpose"
    assert spec["tools"] == []
    assert "no fund tool" in spec["description"]


@pytest.mark.criterion("REQ-FUND-051")
def test_allocator_binds_compute_candidates_not_optimize_portfolio(
    db_session,
) -> None:
    """Validates fund.agents.toolsets.TOOLS_BY_AGENT and bind_toolset: the allocator
    computes the candidate set and no role can call the single optimizer, and the
    risk tools take a candidate id rather than reading one proposal."""
    assert "compute_candidates" in TOOLS_BY_AGENT["allocator"]
    assert all("optimize_portfolio" not in tools for tools in TOOLS_BY_AGENT.values())
    assert "compute_candidates" in fund_tools.__all__
    assert "optimize_portfolio" not in fund_tools.__all__
    ctx = RunContext(
        session=db_session,
        asof=ASOF,
        store=None,
        config=settings,
        run_id=uuid.uuid4(),
        portfolio_id=PORTFOLIO_ID,
    )

    allocator = {tool.name for tool in bind_toolset("allocator", ctx)}
    assert "compute_candidates" in allocator
    assert "optimize_portfolio" not in allocator
    for tool in bind_toolset("risk", ctx):
        assert set(tool.args) == {"candidate_id"}


# --- the candidate rule follows the gate that is pending ---------------------------

# Gates a mandate can add ahead of ``place_orders``: none of them shows candidates.
_ALLOCATOR_GATE = "compute_candidates"
_RISK_GATE = "risk_check"
_NON_ORDER_GATES = (_ALLOCATOR_GATE, _RISK_GATE, "backtest", "get_portfolio_history")
_ORDER_GATE = "place_orders"
_REGATING_ORDERS = 2


def _run_to_mandate_gate(
    db_session, saver: MemorySaver, store: InMemoryStore, gate: str, **kwargs: Any
) -> FundRun:
    """A second round whose persisted mandate adds ``gate`` ahead of the order gate.

    The first round's forced gate set is spent, so the mandate's own gates apply and
    the PM pauses at ``gate`` before ``place_orders``.
    """
    from fund.audit import MandateRepository
    from fund.audit.autonomy_repository import AutonomyRepository

    seed_panel(db_session)
    AutonomyRepository(db_session).consume_first_round(PORTFOLIO_ID)
    mandate = make_mandate().model_copy(update={"hitl_gates": (gate, _ORDER_GATE)})
    MandateRepository(db_session).upsert(mandate)
    run = run_fund(
        _model(**kwargs),
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=db_session,
        checkpointer=saver,
        store=store,
    )
    assert {request["name"] for request in run.interrupt["action_requests"]} == {gate}
    return run


def _pending(saver: MemorySaver, db_session, run_id) -> set[str]:
    run = AgentRunRepository(db_session).get_run(run_id)
    interrupt = observe.interrupt_for(saver, run) or {}
    return {request["name"] for request in interrupt.get("action_requests") or []}


def _choices_and_selection_verdicts(db_session, run_id) -> list[Any]:
    """Every adviser choice and choice-time verdict recorded for the run."""
    return [
        decision
        for decision in _decisions(db_session, run_id)
        if (decision.hitl_decision or {}).get("decision") == "select_candidate"
        or "candidate_selection" in (decision.llm_response or "")
    ]


def _drop_bars_after_asof(db_session) -> None:
    from portopt_db.models.market_data.yfinance_data import PriceHistory

    db_session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
    db_session.flush()


@pytest.mark.criterion("REQ-FUND-047")
@pytest.mark.parametrize(
    ("scenario", "reason"),
    [("tool_fails", "no_candidate_set"), ("none_feasible", "no_feasible_candidate")],
)
def test_a_regate_after_an_approved_extra_gate_with_nothing_to_choose_ends_incomplete(
    db_session, monkeypatch: pytest.MonkeyPatch, scenario: str, reason: str
) -> None:
    """Validates fund.agents.graph_hitl._settle_hitl: when the approved
    ``compute_candidates`` gate yields no set (the tool fails) or no feasible
    candidate and the PM still calls ``place_orders``, the re-gate is closed
    incomplete with the reason instead of staying open, so nothing can approve it."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _run_to_mandate_gate(db_session, saver, store, _ALLOCATOR_GATE)
    if scenario == "tool_fails":
        monkeypatch.setattr(
            fund_tools,
            "compute_candidates",
            lambda *_a, **_k: {"ok": False, "error": "solver unavailable"},
        )
    else:
        mark_candidates_infeasible(monkeypatch, *CANDIDATE_IDS)

    _approve(db_session, saver, store, run.run_id)

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "incomplete"
    assert {"decision": "incomplete", "reason": reason} in [
        decision.hitl_decision for decision in _decisions(db_session, run.run_id)
    ]
    assert db_session.query(PaperOrder).count() == 0
    with pytest.raises(ValueError):
        _approve(db_session, saver, store, run.run_id)
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "incomplete"


@pytest.mark.criterion("REQ-FUND-047")
def test_a_regate_after_a_refused_placement_with_nothing_feasible_ends_incomplete(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validates fund.agents.graph_hitl._settle_hitl: an approve whose ticket could not
    be placed (no fill bar) while the PM recomputes an all-infeasible set and re-gates
    closes the run incomplete rather than leaving that set at an open gate."""
    from fund.agents import graph as graph_mod

    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    monkeypatch.setattr(graph_mod, "_require_fill_bars", lambda *_a, **_k: None)
    _drop_bars_after_asof(db_session)
    mark_candidates_infeasible(monkeypatch, *CANDIDATE_IDS)

    _resume(
        db_session,
        saver,
        store,
        run.run_id,
        "approve",
        candidate_id=MEAN_RISK,
        model_kwargs={"rerun_after_respond": True, "max_orders": _REGATING_ORDERS},
    )

    row = AgentRunRepository(db_session).get_run(run.run_id)
    assert row.status == "incomplete"
    assert {"decision": "incomplete", "reason": "no_feasible_candidate"} in [
        decision.hitl_decision for decision in _decisions(db_session, run.run_id)
    ]
    assert db_session.query(PaperOrder).count() == 0


@pytest.mark.criterion("REQ-FUND-051")
@pytest.mark.parametrize("gate", [_ALLOCATOR_GATE, _RISK_GATE])
def test_approving_a_gate_ahead_of_the_order_gate_records_no_choice(
    db_session, gate: str
) -> None:
    """Validates fund.actions._ticketed_weights and
    fund.agents.graph_candidates.approve_weights: approving a mandate gate that shows
    no candidates (before the set exists, or after it) needs no id, records no
    choice, verdict or weights, places nothing, and the run goes on to the order gate,
    where the adviser chooses."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _run_to_mandate_gate(db_session, saver, store, gate)

    _approve(db_session, saver, store, run.run_id)

    assert _choices_and_selection_verdicts(db_session, run.run_id) == []
    (audit,) = _audit_rows(db_session, run.run_id)
    assert audit.target is None
    assert audit.new_value is None
    assert db_session.query(PaperOrder).count() == 0
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "paused"
    assert _pending(saver, db_session, run.run_id) == {_ORDER_GATE}

    _approve(db_session, saver, store, run.run_id, candidate_id=MEAN_RISK)
    chosen = _candidate(db_session, run.run_id, MEAN_RISK)
    assert _only_order(db_session).weights == chosen.weights


@pytest.mark.criterion("REQ-FUND-048")
@pytest.mark.parametrize("gate", _NON_ORDER_GATES)
def test_a_candidate_id_at_a_gate_that_shows_no_candidates_is_refused(
    db_session, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    """Validates fund.actions._ticketed_weights and fund.agents.graph._approved_weights:
    naming a candidate while a gate other than ``place_orders`` is pending is refused,
    through the actions layer and the rebuild path alike, and writes nothing, even
    when an older candidate set exists."""
    from fund.agents import graph as graph_mod

    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    monkeypatch.setattr(
        observe, "interrupt_for", lambda *_a: {"action_requests": [{"name": gate}]}
    )
    monkeypatch.setattr(graph_mod, "_pending_gate_names", lambda *_a: (gate,))
    before = _snapshot(db_session, run.run_id)

    with pytest.raises(ValueError, match=f"{gate} gate"):
        _approve(db_session, saver, store, run.run_id, candidate_id=MEAN_RISK)
    with pytest.raises(ValueError, match=f"{gate} gate"):
        _resume(db_session, saver, store, run.run_id, "approve", candidate_id=MEAN_RISK)

    assert _snapshot(db_session, run.run_id) == before


@pytest.mark.criterion("REQ-FUND-048")
@pytest.mark.parametrize("gate", _NON_ORDER_GATES)
def test_approving_a_gate_that_shows_no_candidates_commits_no_weights(
    db_session, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    """Validates fund.actions._ticketed_weights: an approve without an id while a gate
    other than ``place_orders`` is pending records an approve row carrying no weights
    and no candidate, and hands the resume no choice to place."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _paused_run(db_session, saver, store)
    monkeypatch.setattr(
        observe, "interrupt_for", lambda *_a: {"action_requests": [{"name": gate}]}
    )
    resumed: dict[str, Any] = {}

    def fake_resume(_rid, _decision, **kwargs):
        resumed.update(kwargs)
        return {}

    monkeypatch.setattr("fund.agents.graph.resume_run", fake_resume)

    _approve(db_session, saver, store, run.run_id)

    (audit,) = _audit_rows(db_session, run.run_id)
    assert (audit.target, audit.new_value) == (None, None)
    assert resumed["candidate_id"] is None
    assert _choices_and_selection_verdicts(db_session, run.run_id) == []


@pytest.mark.criterion("REQ-FUND-050")
def test_a_respond_then_a_compute_candidates_gate_needs_no_choice_with_an_old_set(
    db_session,
) -> None:
    """Validates fund.agents.graph_candidates.approve_weights through
    fund.agents.graph.resume_fund: after a respond the PM recomputes the candidates
    behind a mandate gate on ``compute_candidates``; the old set still exists, yet
    approving that gate needs no id and records no choice, and the order gate that
    follows places the candidate named there from the newer set."""
    saver, store = MemorySaver(), _store_with_cs()
    run = _run_to_mandate_gate(
        db_session,
        saver,
        store,
        _ALLOCATOR_GATE,
        rerun_after_respond=True,
        max_orders=_REGATING_ORDERS,
    )
    regating = {"rerun_after_respond": True, "max_orders": _REGATING_ORDERS}
    _resume(db_session, saver, store, run.run_id, "approve", model_kwargs=regating)
    assert _pending(saver, db_session, run.run_id) == {_ORDER_GATE}
    _resume(
        db_session,
        saver,
        store,
        run.run_id,
        "respond",
        pm_directive=_DIRECTIVE,
        model_kwargs=regating,
    )
    assert _pending(saver, db_session, run.run_id) == {_ALLOCATOR_GATE}
    assert observe.candidate_set(db_session, run.run_id) is not None

    _resume(db_session, saver, store, run.run_id, "approve", model_kwargs=regating)

    assert _choices_and_selection_verdicts(db_session, run.run_id) == []
    assert _pending(saver, db_session, run.run_id) == {_ORDER_GATE}
    _resume(
        db_session,
        saver,
        store,
        run.run_id,
        "approve",
        candidate_id=MEAN_RISK,
        model_kwargs=regating,
    )
    chosen = _candidate(db_session, run.run_id, MEAN_RISK)
    assert _only_order(db_session).weights == chosen.weights
    assert AgentRunRepository(db_session).get_run(run.run_id).status == "completed"
