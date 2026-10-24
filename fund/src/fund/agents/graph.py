"""Phase-7 orchestration graph: the subagents + the PM deep agent.

Assembles the Phase-7A foundations (model / backend / toolsets / prompts / skills)
into one running deep agent. A PM/orchestrator (``create_deep_agent``) delegates
via the built-in ``task`` tool to four stateless subagents in a fixed pipeline —
**economist → allocator → risk (blocking gate) → executor** — and owns the single
gated action: committing the paper trades behind the HITL ``place_orders`` gate.

Two shapes live here (Task 5 builds them; Task 6 drives them end-to-end):

* ``build_<role>_subagent(ctx)`` → a deepagents ``SubAgent`` dict. Subagents do
  **not** inherit tools/skills, so each carries its own — the per-run toolset
  (:func:`fund.agents.toolsets.bind_toolset`, session already bound) and the
  per-role skills. **No subagent sets ``interrupt_on``** (HITL is PM-level only);
  the critical roles (allocator/risk/executor) attach a
  ``ModelFallbackMiddleware`` when a fallback model is supplied.
* ``build_fund_agent(...)`` → the PM ``create_deep_agent`` wiring the orchestrator
  toolset (the top-level gated ``place_orders``), the four subagents, the
  root-relative skills, the ``virtual_mode`` backend, the checkpointer/store, and
  the effective ``interrupt_on`` gate set (config baseline ∪ the mandate's
  ``hitl_gates``, computed by the caller).

**Risk R1 (verified):** deepagents loads skills **through the backend**, which
runs in ``virtual_mode`` and blocks absolute paths outside its root — so every
agent's ``skills`` are the root-relative
:func:`fund.agents.skills.skill_sources` (the per-role dirs the backend stages),
never the absolute ``skill_paths``.

The agent stack (``deepagents`` / ``langchain``) is imported **inside** the
builders, so a bare ``import fund.agents.graph`` drags in no agent runtime and
needs no environment (mirrors :mod:`fund.agents.profiler`).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, cast

from fund.agents.backend import build_backend, read_only_permissions
from fund.agents.prompts import (
    ALLOCATOR_SYSTEM_PROMPT,
    ECONOMIST_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
)
from fund.agents.skills import skill_sources
from fund.agents.summarization import summarization_override
from fund.config import DEFAULT_LOOKBACK_DAYS, FundConfig, settings

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterator, Sequence

    # ``fund.agents.toolsets`` reaches ``fund.tools`` (the agent stack) at import;
    # keep it type-only here and import ``bind_toolset`` lazily inside the builders
    # so a bare ``import fund.agents.graph`` stays agent-stack-free (profiler style).
    # The stream DTO below is type-only too; the generator imports it lazily.
    from fund.agents.toolsets import RunContext
    from fund.schemas import ConstraintSet
    from fund.schemas.mandate import PortfolioMandate
    from fund.tui.stream import StreamEvent

__all__ = [
    "FundRun",
    "build_allocator_subagent",
    "build_economist_subagent",
    "build_executor_subagent",
    "build_fund_agent",
    "build_risk_subagent",
    "resume_fund",
    "resume_run",
    "run_fund",
    "stream_fund",
]

# The moments/backtest lookback recorded on every run (the ``RunContext.lookback_days``
# default — 3y rolling, ~756 trading days).
_LOOKBACK_DAYS = DEFAULT_LOOKBACK_DAYS

# One-line descriptions the PM reads (via the built-in ``task`` tool) to decide
# when to delegate — each names the role's fixed pipeline position.
_ECONOMIST_DESCRIPTION = (
    "Pipeline step 1. Reads the macro and price backdrop and hands the allocator a "
    "qualitative regime narrative plus candidate views. Delegate first."
)
_ALLOCATOR_DESCRIPTION = (
    "Pipeline step 2. Pre-selects the candidate tickers it is handed, estimates "
    "moments, and runs the optimizer to produce candidate weights under the "
    "resolved constraints; the economist's views are advisory context, not an "
    "optimizer input. Delegate after the economist."
)
_RISK_DESCRIPTION = (
    "Pipeline step 3 — the blocking gate. Checks the allocator's proposal against "
    "the MiFID/ESG/validation limits and reports pass or fail; nothing proceeds to "
    "the executor until it passes."
)
_EXECUTOR_DESCRIPTION = (
    "Pipeline step 4. Turns the risk-approved target weights into a paper rebalance "
    "proposal for the PM to commit. Delegate only once risk has passed."
)


def _subagent(
    role: str,
    *,
    description: str,
    system_prompt: str,
    ctx: RunContext,
    fallback: Any | None = None,
    summarization: Sequence[Any] = (),
) -> dict[str, Any]:
    """Build one ``SubAgent`` dict for ``role`` from the per-run context.

    Carries the role's own per-run tools (``session`` already bound) and its
    root-relative skills; never sets ``interrupt_on``. When ``fallback`` is given
    (critical roles only), attaches a single ``ModelFallbackMiddleware`` so a
    primary-model failure retries on the fallback (a langchain middleware, not a
    hand-rolled ``.with_fallbacks``). Every role also carries the per-delegation
    model-call cap (:func:`_subagent_call_cap_middleware`). ``summarization`` carries
    the operator's override from
    :func:`fund.agents.summarization.summarization_override`; a declarative subagent
    inherits no middleware from the PM, so it is attached to each spec explicitly.
    """
    from fund.agents.toolsets import bind_toolset

    middleware: list[Any] = []
    if fallback is not None:
        from langchain.agents.middleware import ModelFallbackMiddleware

        middleware.append(ModelFallbackMiddleware(fallback))
    middleware.append(
        _subagent_call_cap_middleware(ctx.config.max_subagent_model_calls)
    )
    middleware.extend(summarization)
    return {
        "name": role,
        "description": description,
        "system_prompt": system_prompt,
        "tools": bind_toolset(role, ctx),
        "skills": skill_sources(role),
        "middleware": middleware,
    }


def build_economist_subagent(
    ctx: RunContext, *, summarization: Sequence[Any] = ()
) -> dict[str, Any]:
    """The economist subagent (step 1) — narrative regime read, no fallback."""
    return _subagent(
        "economist",
        description=_ECONOMIST_DESCRIPTION,
        system_prompt=ECONOMIST_SYSTEM_PROMPT,
        ctx=ctx,
        summarization=summarization,
    )


def build_allocator_subagent(
    ctx: RunContext,
    *,
    fallback: Any | None = None,
    summarization: Sequence[Any] = (),
) -> dict[str, Any]:
    """The allocator subagent (step 2) — the optimizer path; critical role."""
    return _subagent(
        "allocator",
        description=_ALLOCATOR_DESCRIPTION,
        system_prompt=ALLOCATOR_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
        summarization=summarization,
    )


def build_risk_subagent(
    ctx: RunContext,
    *,
    fallback: Any | None = None,
    summarization: Sequence[Any] = (),
) -> dict[str, Any]:
    """The risk subagent (step 3) — the blocking gate; critical role."""
    return _subagent(
        "risk",
        description=_RISK_DESCRIPTION,
        system_prompt=RISK_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
        summarization=summarization,
    )


def build_executor_subagent(
    ctx: RunContext,
    *,
    fallback: Any | None = None,
    summarization: Sequence[Any] = (),
) -> dict[str, Any]:
    """The executor subagent (step 4) — proposes the rebalance; critical role.

    Binds no tool from the frozen eight (``place_orders`` is the PM's tool, behind
    the HITL gate): the executor *proposes*, the PM *commits*.
    """
    return _subagent(
        "executor",
        description=_EXECUTOR_DESCRIPTION,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
        summarization=summarization,
    )


def _subagent_call_cap_middleware(max_model_calls: int) -> Any:
    """Bound one subagent delegation to ``max_model_calls`` model calls.

    The PM's ``recursion_limit`` and :func:`_pm_round_cap_middleware` count only PM
    supersteps, and the ``task`` tool runs a subagent graph that carries langgraph's
    own 9999 bound (the subagent's bound config wins over the parent's), so a
    subagent that keeps retrying a failing tool would otherwise run for thousands
    of model calls inside a single PM round. ``run_limit`` resets on every ``task``
    invocation; ``exit_behavior="end"`` hands the PM a final "limits exceeded"
    message instead of raising across the delegation. Imported lazily to keep a
    bare ``import fund.agents.graph`` agent-stack-free.
    """
    from langchain.agents.middleware import ModelCallLimitMiddleware

    return ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end")


def _general_purpose_subagent(
    config: FundConfig, *, summarization: Sequence[Any] = ()
) -> dict[str, Any]:
    """Explicit ``general-purpose`` spec carrying the same call cap as the roles.

    deepagents auto-adds its own ``general-purpose`` subagent to the ``task`` tool
    unless the caller supplies one by that name, and the auto-added spec has no
    model-call bound. Supplying the stock spec (prompt, description, inherited PM
    tools and skills) plus the cap keeps the default behaviour while closing that
    runaway path.
    """
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    return {
        **GENERAL_PURPOSE_SUBAGENT,
        "skills": skill_sources("orchestrator"),
        "middleware": [
            _subagent_call_cap_middleware(config.max_subagent_model_calls),
            *summarization,
        ],
    }


def _pm_round_cap_middleware(max_pm_rounds: int) -> Any:
    """Build the PM-level delegation round-cap middleware (D22 cost cap).

    Defined lazily — the class subclasses ``AgentMiddleware`` — so a bare
    ``import fund.agents.graph`` drags in no agent stack (mirrors the builders).
    ``after_model`` counts each PM model round; once ``max_pm_rounds`` rounds have
    elapsed, ``before_model`` jumps straight to ``end`` (never ``Command(goto=…)``;
    the ``@hook_config(can_jump_to=["end"])`` decorator is what makes the jump edge
    exist) and stamps ``pm_round_cap_reached`` — which :func:`run_fund` maps to an
    ``incomplete`` finalise (reason ``"pm_round_cap"``). This is a deterministic
    cap distinct from the langgraph ``recursion_limit`` guard (reason
    ``"round_cap"``): whichever bound is lower stops the runaway PM first.
    """
    from langchain.agents.middleware import (
        AgentMiddleware,
        AgentState,
        hook_config,
    )
    from langchain_core.messages import AIMessage

    class _PMRoundCapState(AgentState):
        pm_rounds: NotRequired[int]
        pm_round_cap_reached: NotRequired[bool]

    class PMRoundCapMiddleware(AgentMiddleware):
        """Bounds the PM's delegation rounds, jumping to ``end`` at the cap."""

        state_schema = _PMRoundCapState

        def __init__(self, cap: int) -> None:
            super().__init__()
            self._cap = cap

        @hook_config(can_jump_to=["end"])
        def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
            if state.get("pm_rounds", 0) >= self._cap:
                return {
                    "jump_to": "end",
                    "pm_round_cap_reached": True,
                    "messages": [
                        AIMessage(
                            content=(
                                f"PM delegation round cap ({self._cap}) reached; "
                                "halting without a paper ticket."
                            )
                        )
                    ],
                }
            return None

        def after_model(self, state: Any, runtime: Any) -> dict[str, Any]:
            return {"pm_rounds": state.get("pm_rounds", 0) + 1}

    return PMRoundCapMiddleware(max_pm_rounds)


def build_fund_agent(
    model: Any,
    *,
    checkpointer: Any,
    store: Any | None,
    ctx: RunContext,
    config: FundConfig = settings,
    fallback: Any | None = None,
    interrupt_on: dict[str, Any] | None = None,
) -> Any:
    """Assemble the PM ``deepagents`` agent that orchestrates one fund run.

    The PM runs under ``PM_SYSTEM_PROMPT`` with the orchestrator toolset — the
    single top-level ``place_orders`` tool, gated by ``interrupt_on`` (which
    requires the ``checkpointer``) so the commit always pauses for the adviser. The
    gate set is the ``interrupt_on`` override when supplied (the caller folds the
    mandate's ``hitl_gates`` in), else the config default (``place_orders``). It
    delegates through the built-in ``task`` tool to the four subagents in pipeline
    order (economist → allocator → risk → executor); the critical three carry the
    ``fallback`` middleware when one is supplied, and every delegation target (the
    ``general-purpose`` one included) is bounded by the per-delegation model-call
    cap. Skills are the root-relative
    staged sources (Risk R1) resolved through the ``virtual_mode`` ``backend``, which
    every role may read but none may write (:func:`read_only_permissions`).
    ``temperature=0`` / the DeepSeek route ride on ``model`` (built from
    ``FundConfig``), not passed here.
    """
    from deepagents import create_deep_agent

    from fund.agents.toolsets import bind_toolset, build_gate_weights_middleware

    gates = interrupt_on if interrupt_on is not None else config.interrupt_on_map()
    backend = build_backend(config)

    def summarization() -> list[Any]:
        return summarization_override(model, backend, config)

    subagents = [
        build_economist_subagent(ctx, summarization=summarization()),
        build_allocator_subagent(ctx, fallback=fallback, summarization=summarization()),
        build_risk_subagent(ctx, fallback=fallback, summarization=summarization()),
        build_executor_subagent(ctx, fallback=fallback, summarization=summarization()),
        _general_purpose_subagent(config, summarization=summarization()),
    ]
    return create_deep_agent(
        model=model,
        tools=bind_toolset("orchestrator", ctx),
        system_prompt=PM_SYSTEM_PROMPT,
        subagents=cast("list[Any]", subagents),
        skills=skill_sources("orchestrator"),
        backend=backend,
        permissions=read_only_permissions(),
        interrupt_on=cast("dict[str, Any]", gates),
        middleware=[
            _pm_round_cap_middleware(config.max_pm_rounds),
            build_gate_weights_middleware(ctx),
            *summarization(),
        ],
        checkpointer=checkpointer,
        store=store,
    )


# The full set of PM-level gates round-0 forces (every shipped orchestrator gate).
# A brand-new portfolio's first round pauses at all of them regardless of the
# mandate; ``first_round_gated`` then flips so later rounds respect the mandate.
# before_risk is deferred (needs a synthetic checkpoint tool), so place_orders is
# the only real PM gate today — grow this tuple as gates ship.
_ALL_ORCHESTRATOR_GATES: tuple[str, ...] = ("place_orders",)


def _effective_gates(config: FundConfig, mandate: PortfolioMandate) -> dict[str, bool]:
    """Union the config's gated tools with the mandate's ``hitl_gates``.

    The effective ``interrupt_on`` for a run: every tool the config gates plus every
    tool the mandate adds, each mapped to ``True``. A default mandate leaves the gate
    set at the config baseline (``place_orders``); a mandate can only widen it.
    """
    gates = frozenset(config.interrupt_on) | frozenset(mandate.hitl_gates)
    return dict.fromkeys(gates, True)


# ---------------------------------------------------------------------------
# ``run_fund`` + ``FundRun``: the one complete paper-run path.
#
# Mirrors ``run_profiler`` / ``ProfilerRun``: resolve the run's inputs, build the
# PM agent, invoke it so the ``place_orders`` gate pauses for the adviser, and hand
# back a ``FundRun`` whose ``.resume(decision)`` either commits the paper ticket
# (approve) or discards it (reject). The agent stack (deepagents / langgraph) is
# imported lazily so a bare ``import fund.agents.graph`` stays agent-stack-free.
# ---------------------------------------------------------------------------


def _coerce_uuid(portfolio_id: uuid.UUID | str) -> uuid.UUID:
    """Normalise ``portfolio_id`` to a ``UUID`` (accepts a canonical string)."""
    if isinstance(portfolio_id, uuid.UUID):
        return portfolio_id
    return uuid.UUID(portfolio_id)


def _extract_interrupt(result: Any) -> dict[str, Any] | None:
    """Pull the HITL interrupt payload from a compiled-graph invoke result."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    value = interrupts[0].value
    if not isinstance(value, dict):
        return None
    return cast("dict[str, Any]", value)


def _optimizer_weights(session: Any, run_id: uuid.UUID) -> dict[str, float]:
    """The latest allocator ``optimize_portfolio`` weights logged for this run.

    The allocator's bound tool logs the load-bearing decision (constraint set +
    effective optimizer config + the optimizer's weights) inside its closure during
    the initial invoke. Reading them back from the audit trail — never from what
    the PM/LLM passed — keeps ``optimize_portfolio`` the single source of weights.
    Delegates to :meth:`AgentRunRepository.latest_optimizer_weights` so the paper
    ticket and the finalised ``agent_runs.weights`` derive from one source and
    cannot diverge. Returns ``{}`` when the allocator never produced a proposal.
    """
    from fund.audit import AgentRunRepository

    return AgentRunRepository(session).latest_optimizer_weights(run_id)


def _trades_from_lines(
    lines: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Index committed paper-order fill lines by ticker for the overlay.

    ``lines`` come straight from ``paper_orders.lines`` (a JSON column), so the
    values are already JSON-safe. Returns ``None`` when there are no fills.
    """
    if not lines:
        return None
    trades: dict[str, Any] = {}
    for line in lines:
        ticker = line.get("ticker")
        if ticker is None:
            continue
        trades[str(ticker)] = {k: v for k, v in line.items() if k != "ticker"}
    return trades or None


def _drift_vs_prior(
    prior: dict[str, float], target: dict[str, float]
) -> dict[str, Any]:
    """Per-ticker weight delta from prior holdings to the new target weights.

    Covers the union of held and targeted tickers (missing side counts as 0),
    so both exits and new entries show up. Bands-crossed classification is out
    of scope here — ``_finalize_hitl`` has no constraint set — and is deferred.
    """
    out: dict[str, Any] = {}
    for ticker in sorted(set(prior) | set(target)):
        before = float(prior.get(ticker, 0.0))
        after = float(target.get(ticker, 0.0))
        out[ticker] = {
            "prior": before,
            "target": after,
            "delta": round(after - before, 6),
        }
    return out


def _portfolio_narrative(
    asof: dt.date,
    allocation: dict[str, float],
    trades: dict[str, Any] | None,
    drift: dict[str, Any],
) -> str:
    """Deterministic one-line summary of an approved rebalance.

    A pure function of the overlay sections — the same inputs reproduce the
    string byte-for-byte, keeping the overlay reproducible for audit.
    """
    n_fills = len(trades) if trades else 0
    moved = sorted(t for t, d in drift.items() if abs(d.get("delta", 0.0)) > 1e-9)
    moves = ", ".join(moved) if moved else "none"
    return (
        f"Rebalance approved for {asof.isoformat()}: "
        f"{len(allocation)} target position(s), {n_fills} fill(s); "
        f"weight moves: {moves}."
    )


def _write_portfolio_journal(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
    order_lines: list[dict[str, Any]] | None,
    prior_weights: dict[str, float],
) -> None:
    """Upsert the per-portfolio overlay row for an approved rebalance.

    Assembles ``allocation`` (target weights), ``trades`` (committed fills), and
    ``drift`` (vs ``prior_weights``) plus a deterministic narrative, then upserts
    keyed on ``(portfolio_id, asof)``. Repos are imported lazily — ``fund.audit``
    drags in the agent stack, so a bare ``import fund.agents.graph`` must not — and
    no ``commit`` is issued (the caller owns the transaction).
    """
    from portopt_db.repositories.orders.portfolio_journal_repository import (
        PortfolioJournalRepository,
    )

    from fund.audit import OrderRepository

    lines = order_lines
    if lines is None:
        order = OrderRepository(session).latest_for_portfolio(portfolio_id)
        if order is not None and order.asof == asof:
            lines = order.lines

    allocation = {ticker: float(weight) for ticker, weight in weights.items()}
    trades = _trades_from_lines(lines)
    drift = _drift_vs_prior(prior_weights, allocation)
    narrative = _portfolio_narrative(asof, allocation, trades, drift)

    PortfolioJournalRepository(session).upsert(
        portfolio_id,
        asof,
        run_id=run_id,
        trades=trades,
        allocation=allocation,
        drift=drift,
        narrative=narrative,
    )


def _finalize_hitl(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
    decision: str,
    order_lines: list[dict[str, Any]] | None = None,
) -> None:
    """Finalise a HITL decision — the single write path both resume routes share.

    Reached through :func:`_settle_hitl` by :meth:`FundRun.resume` (the live in-process
    handle) **and** :func:`resume_fund` (the cross-process rebuild), so the two cannot
    diverge.
    ``approve`` finalises the run ``completed`` with the optimizer ``weights`` and
    replaces the portfolio's ``positions`` snapshot built from them (O1 = Option B
    per-ticker row dicts; ``order_lines``, when supplied, carries the filled
    shares/notional). Any other ``decision`` writes no order, records the
    orchestrator's HITL choice, and finalises the run ``rejected``.

    The ``agent.invoke(Command(resume=…))`` that actually runs (approve) or denies
    (reject) the gated ``place_orders`` tool is the **caller's** job — inside that
    invoke the executor's execution decision and the paper ticket are written. This
    helper only stamps the terminal audit state + the current-holdings snapshot. No
    ``commit`` (the caller owns the transaction).
    """
    from fund.audit import AgentRunRepository, PositionRepository

    audit = AgentRunRepository(session)
    if decision == "approve":
        audit.finalize_run(run_id, weights=weights, status="completed")
        positions = PositionRepository(session)
        # Capture prior holdings *before* set_holdings overwrites the snapshot, so
        # the overlay's drift is this bar's change rather than a no-op against itself.
        prior_weights = {
            p.ticker: float(p.weight) for p in positions.get_holdings(portfolio_id)
        }
        holdings = order_lines or [
            {"ticker": ticker, "weight": weight} for ticker, weight in weights.items()
        ]
        positions.set_holdings(portfolio_id, holdings, asof=asof)
        _write_portfolio_journal(
            session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            asof=asof,
            weights=weights,
            order_lines=order_lines,
            prior_weights=prior_weights,
        )
    else:
        audit.append_decision(
            run_id,
            agent="orchestrator",
            step="place_orders",
            hitl_decision={"decision": decision},
        )
        audit.finalize_run(run_id, weights={}, status="rejected")


def _ticket_placed(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
) -> bool:
    """Whether this run's approved ``place_orders`` call left a ticket for ``weights``.

    ``place_orders`` can decline an approved gate (a missing fill bar, the current
    mandate no longer admits the audited weights, or the risk gate had not passed
    them) and still return normally to the graph, so a resume that completed without
    raising proves nothing about the ticket. The ticket row, keyed
    ``(portfolio_id, asof, weights_hash)`` like the tool's own idempotency lookup, is
    the only evidence that the book moved; an idempotent re-run that returned the
    stored ticket counts as placed. The tool is refused unless the risk gate passed
    exactly these weights, so a ticket an earlier run left on the same bar does not
    count when risk never passed this one. No weights means nothing could have been
    placed.
    """
    from fund.audit import AgentRunRepository, OrderRepository
    from fund.audit.orders_repository import weights_hash

    if not weights or not AgentRunRepository(session).risk_passed(run_id, weights):
        return False
    ticket = OrderRepository(session).get_by_key(
        portfolio_id=portfolio_id, asof=asof, weights_hash=weights_hash(weights)
    )
    return ticket is not None


def _settle_hitl(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
    decision: str,
    result: Any,
) -> None:
    """Finalise an approve/reject resume, verifying an approve actually ticketed.

    ``place_orders`` reports a missing fill bar, an unknown ticker or a mandate breach
    as an ``{ok: false}`` envelope instead of raising, so resuming the gate succeeds
    even when no ticket was written. An approve with no ticket must not stamp the run
    ``completed`` nor replace the ``positions`` snapshot with weights no order backs:
    it finalises ``incomplete`` (``"place_orders_failed"``). When the resumed PM
    answered by re-gating (``result`` carries a fresh interrupt) the run stays alive as
    ``"responded"`` instead, with the declined approve on the audit trail. Reject and
    a ticketed approve go through :func:`_finalize_hitl`. No commit.
    """
    if decision != "approve" or _ticket_placed(
        session, run_id=run_id, portfolio_id=portfolio_id, asof=asof, weights=weights
    ):
        _finalize_hitl(
            session,
            run_id=run_id,
            portfolio_id=portfolio_id,
            asof=asof,
            weights=weights,
            decision=decision,
        )
        return

    from fund.audit import AgentRunRepository

    audit = AgentRunRepository(session)
    if _extract_interrupt(result) is None:
        _finalize_incomplete(audit, run_id, "place_orders_failed")
        return
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "approve", "reason": "place_orders_failed"},
    )
    audit.mark_responded(run_id)


_HISTORY_SUMMARY_BUDGET = 800
_HISTORY_DIGEST_WINDOW_DAYS = 14


def _history_summary(
    session: Any,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    *,
    budget: int = _HISTORY_SUMMARY_BUDGET,
) -> str:
    """Compact "since last rebalance" recap for the PM seed (empty on first run).

    Reads the portfolio's most recent prior overlay (allocation + whether it
    traded) and the latest global market digest, both bounded strictly/at ``asof``
    (no look-ahead), and folds them into one short string. Repos are imported
    lazily so a bare ``import fund.agents.graph`` stays agent-stack-free, and the
    result is capped to ``budget`` chars so it never crowds the model context or
    the PM round cap.

    Args:
        session: A sync ``portopt_db`` session; not owned here.
        portfolio_id: The portfolio whose history to recap.
        asof: The current decision bar; only activity before/at it is considered.
        budget: Maximum character length of the returned recap.

    Returns:
        The recap string, or ``""`` when the portfolio has no prior rebalance and
        no digest is available.
    """
    import datetime as dt

    from portopt_db.repositories.market_data.market_journal_repository import (
        MarketJournalRepository,
    )
    from portopt_db.repositories.orders.portfolio_journal_repository import (
        PortfolioJournalRepository,
    )

    parts: list[str] = []

    overlays = PortfolioJournalRepository(session).list_for_portfolio(portfolio_id)
    prior = next((o for o in overlays if o.as_of < asof), None)
    if prior is not None:
        allocation = prior.allocation or {}
        alloc_str = ", ".join(f"{t} {w}" for t, w in sorted(allocation.items()))
        parts.append(f"Prior rebalance {prior.as_of.isoformat()}: {alloc_str}.")
        if prior.trades:
            parts.append(f"Traded: {', '.join(sorted(prior.trades))}.")

    start = asof - dt.timedelta(days=_HISTORY_DIGEST_WINDOW_DAYS)
    digests = MarketJournalRepository(session).get_range(start, asof)
    if digests:
        latest = digests[-1]  # get_range is ascending → the freshest at/before asof
        parts.append(f"Latest digest {latest.as_of.isoformat()}: {latest.narrative}")

    if not parts:
        return ""
    summary = "Since last rebalance — " + " ".join(parts)
    if len(summary) > budget:
        summary = summary[: budget - 3].rstrip() + "..."
    return summary


_DIRECTIVES_BUDGET = 1200
_DIRECTIVE_MAX_CHARS = 300


def _directives_block(
    mandate: PortfolioMandate, *, budget: int = _DIRECTIVES_BUDGET
) -> str:
    """Render the adviser's remembered directives, or ``""`` when there are none.

    Each directive is collapsed onto a single bullet line so free text cannot pose
    as a separate instruction to the PM, capped to ``_DIRECTIVE_MAX_CHARS``, and the
    newest directives win when the total would exceed ``budget``, so a long-lived
    mandate cannot crowd the PM context.
    """
    kept: list[str] = []
    used = 0
    for directive in reversed(mandate.remembered_directives):
        text = " ".join(directive.split())
        if not text:
            continue
        if len(text) > _DIRECTIVE_MAX_CHARS:
            text = text[: _DIRECTIVE_MAX_CHARS - 3].rstrip() + "..."
        if used + len(text) > budget:
            break
        kept.append(text)
        used += len(text)
    if not kept:
        return ""
    bullets = "\n".join(f"- {text}" for text in reversed(kept))
    return (
        "Adviser standing directives (authored by the human adviser and remembered "
        "across rounds; soft guardrails). Pass them on in every delegation brief and "
        "have risk flag any proposal that breaches one. They never relax the "
        f"ConstraintSet limits, the risk gate or the adviser approval:\n{bullets}\n\n"
    )


def _constraint_digest(constraint_set: ConstraintSet) -> str:
    """One-line read-only digest of the limits the PM must relay to its subagents.

    The ``ConstraintSet`` lives in the langgraph Store, which no agent tool can
    read, so the seed turn is the only place the PM can learn the objective,
    bounds and drawdown ceilings the skills tell it to pass on. It is rendered
    from the resolved set, never model-authored, so the risk and executor steps
    compare against the real limits instead of invented ones.
    """
    bounds = constraint_set.bounds
    exclusions = ", ".join(s.value for s in constraint_set.esg.exclusions) or "none"
    position_cap = constraint_set.universe_filters.max_position_cap
    cap = "" if position_cap is None else f", per-name position cap {position_cap}"
    return (
        f"ConstraintSet: objective {constraint_set.objective.value}, "
        f"risk measure {constraint_set.risk_measure.value}, "
        f"risk aversion {constraint_set.a_gamma}, tail beta {constraint_set.beta}, "
        f"weight bounds [{bounds.min_weights}, {bounds.max_weights}]{cap}, "
        f"budget {bounds.budget}, drawdown ceilings (fractions of capital) "
        f"nu1 {constraint_set.nu1} / nu2 {constraint_set.nu2} / "
        f"nu3 {constraint_set.nu3}, ESG exclusions {exclusions}."
    )


def _run_instruction(
    mandate: PortfolioMandate,
    portfolio_id: str,
    asof: dt.date,
    history_summary: str = "",
    *,
    constraint_set: ConstraintSet | None = None,
) -> str:
    """The human turn that drives the PM through one paper rebalance.

    When ``history_summary`` is non-empty it is prepended as a context block so
    the PM sees the portfolio's recent allocation/trades and the latest market
    digest before deciding; an empty summary (first run) yields the bare seed.
    The mandate's ``remembered_directives`` follow as one bullet each: the adviser
    saved them to steer every later round, including unattended scheduled ones. The
    mandate line carries the drift band and HITL gates, and a given ``constraint_set``
    adds its limits digest, so the PM can pass real numbers to the subagents rather
    than guessing them.
    """
    benchmark = f", benchmark {mandate.benchmark}" if mandate.benchmark else ""
    history = f"{history_summary}\n\n" if history_summary else ""
    gates = ", ".join(mandate.hitl_gates) or "none"
    limits = f" {_constraint_digest(constraint_set)}" if constraint_set else ""
    return (
        f"{history}{_directives_block(mandate)}Produce an allocation for portfolio "
        f"{portfolio_id} as of {asof.isoformat()}. Mandate: capital {mandate.capital} "
        f"{mandate.base_currency}{benchmark}, no-trade band (drift band (L1)) "
        f"{mandate.drift_l1_threshold}, HITL gates {gates}.{limits} Delegate in "
        f"order economist -> allocator -> risk -> executor, then commit the "
        f"risk-approved trades through place_orders."
    )


@dataclass(frozen=True)
class FundRun:
    """Handle for one fund run paused at the ``place_orders`` adviser gate.

    Mirrors :class:`~fund.agents.profiler.ProfilerRun`. Carries the resolved
    ``constraint_set``, the load-bearing optimizer ``weights`` (captured from the
    audit trail before the pause), the ``interrupt`` payload the adviser reviews,
    the run ``status``, and enough state to resume. ``resume("approve")`` commits
    the paper ticket and finalises the run completed, or incomplete when
    ``place_orders`` wrote no ticket (see :func:`_settle_hitl`); ``resume("reject")``
    commits no order, records the HITL choice, and finalises the run rejected.
    """

    constraint_set: ConstraintSet
    run_id: uuid.UUID
    interrupt: dict[str, Any] | None
    weights: dict[str, float]
    status: str
    agent: Any = field(repr=False, compare=False)
    thread_config: dict[str, Any] = field(repr=False, compare=False)
    session: Any = field(repr=False, compare=False)

    def resume(
        self, decision: str, *, pm_directive: str | None = None
    ) -> dict[str, Any]:
        """Resume the paused agent with an adviser ``decision``.

        ``approve`` runs the gated ``place_orders`` tool (the paper ticket is written
        and the executor decision logged inside the closure), then finalises the run
        as completed with the optimizer's weights — unless ``place_orders`` placed no
        ticket (a missing fill bar, the audited weights breach the current mandate, or
        the risk gate had not passed them), in which case nothing is booked and the run
        ends ``incomplete`` (or stays ``responded`` if the PM re-gated). ``reject``
        writes no order, records the HITL choice, and finalises rejected. ``respond``
        threads ``pm_directive`` back to the PM and lets it continue: if it re-gates
        the run stays alive as ``"responded"`` (still awaiting approval), else it
        finalises ``incomplete`` — delegated to :func:`_finalize_respond`.

        Approve/reject finalisation is delegated to the shared :func:`_settle_hitl`,
        and respond to :func:`_finalize_respond`, so this in-process path and the
        cross-process :func:`resume_fund` cannot diverge — both stamp the audit trail
        identically. The run's ``portfolio_id`` / ``asof`` (which ``FundRun`` does not
        carry) are read back from its persisted row for the approve/reject path, and
        so are the audited weights: the vector captured at the pause may have been
        superseded by the time the adviser decides, and finalisation must record what
        ``place_orders`` placed.
        """
        from fund.audit import AgentRunRepository

        result = _resume_agent(self.agent, self.thread_config, decision, pm_directive)
        if decision == "respond":
            _finalize_respond(
                self.session,
                run_id=self.run_id,
                result=result,
                pm_directive=pm_directive,
            )
            return result
        run = AgentRunRepository(self.session).get_run(self.run_id)
        if run is None:
            raise LookupError(f"no agent_run {self.run_id}")
        if run.portfolio_id is None:
            raise ValueError(f"agent_run {self.run_id} has no portfolio_id")
        _settle_hitl(
            self.session,
            run_id=self.run_id,
            portfolio_id=run.portfolio_id,
            asof=run.asof,
            weights=_optimizer_weights(self.session, self.run_id),
            decision=decision,
            result=result,
        )
        return result


def _finalize_incomplete(audit: Any, run_id: uuid.UUID, reason: str) -> None:
    """Finalise a run that ended without a ticket — the shared no-ticket end.

    Records the orchestrator's incomplete decision with ``reason`` (``"round_cap"`` /
    ``"pm_round_cap"`` / ``"no_order"`` / ``"place_orders_failed"``) and finalises the
    run ``incomplete`` with no weights. Shared by the blocking :func:`run_fund`, the
    streaming :func:`stream_fund` and the approve resume so they cannot diverge on the
    no-ticket path. No commit.
    """
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "incomplete", "reason": reason},
    )
    audit.finalize_run(run_id, weights={}, status="incomplete")


def _resume_command(
    decision: str, pm_directive: str | None, *, actions: int = 1
) -> Any:
    """Build the HITL ``Command(resume=…)`` for a gate decision.

    ``approve`` / ``reject`` carry only the decision type; ``respond`` carries the
    adviser's free-text ``pm_directive`` as the gated tool's result, threading it back
    to the PM (a successful ``place_orders`` ``ToolMessage``) so the gate continues
    instead of terminating. The middleware demands exactly one decision per gated tool
    call, so the same decision is repeated ``actions`` times (a PM turn that emits
    parallel gated calls pauses with several action requests). Imports ``langgraph``
    lazily to keep a bare ``import fund.agents.graph`` light.
    """
    from langgraph.types import Command

    decisions: list[dict[str, Any]] = [
        {"type": "respond", "message": pm_directive}
        if decision == "respond"
        else {"type": decision}
        for _ in range(max(actions, 1))
    ]
    return Command(resume={"decisions": decisions})


def _pending_action_count(agent: Any, thread_config: dict[str, Any]) -> int:
    """How many gated tool calls the paused PM is awaiting a decision on.

    Read from the checkpointed interrupt rather than a cached payload, because a
    ``respond`` can re-gate with a different number of calls than the first pause.
    ``1`` when no interrupt carries action requests.
    """
    for pending in agent.get_state(thread_config).interrupts:
        value = pending.value
        requests = value.get("action_requests") if isinstance(value, dict) else None
        if requests:
            return len(requests)
    return 1


def _resume_agent(
    agent: Any,
    thread_config: dict[str, Any],
    decision: str,
    pm_directive: str | None,
) -> dict[str, Any]:
    """Resume the paused PM, answering every gated tool call it is waiting on."""
    result: dict[str, Any] = agent.invoke(
        _resume_command(
            decision,
            pm_directive,
            actions=_pending_action_count(agent, thread_config),
        ),
        config=thread_config,
    )
    return result


def _finalize_respond(
    session: Any,
    *,
    run_id: uuid.UUID,
    result: Any,
    pm_directive: str | None,
) -> None:
    """Settle a ``respond`` barge-in: keep the run alive iff the PM re-gated.

    Shared by :meth:`FundRun.resume` and :func:`resume_fund` so the in-process and
    rebuild-to-resume paths cannot diverge. The respond is always appended to the audit
    trail (carrying the directive). When the resumed PM re-gates — ``result`` carries a
    fresh interrupt — the run is flipped to ``"responded"`` (alive, still awaiting
    approval) and is **not** finalised; a genuinely terminal result (no re-gate, so no
    ticket passed the gate) finalises the run ``incomplete`` — the same no-gate end as
    :func:`run_fund`. No commit (the caller owns the transaction).
    """
    from fund.audit import AgentRunRepository

    audit = AgentRunRepository(session)
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "respond", "message": pm_directive},
    )
    if _extract_interrupt(result) is not None:
        audit.mark_responded(run_id)
    else:
        reason = (
            "pm_round_cap"
            if isinstance(result, dict) and result.get("pm_round_cap_reached")
            else "no_order"
        )
        _finalize_incomplete(audit, run_id, reason)


def _frozen_scope(session: Any, portfolio_id: uuid.UUID) -> str | None:
    """The autonomy scope FROZEN for ``portfolio_id`` (``"fund"`` wins), else ``None``.

    Reads the database rather than the session's cached rows: a run session loads the
    armed rows when it opens, and a freeze committed by the operator's session while
    the round is in flight is otherwise invisible to it. Read-only: it never creates a
    row.
    """
    from fund.audit.autonomy_repository import AutonomyRepository

    scope = AutonomyRepository(session).frozen_scope(portfolio_id)
    return scope.value if scope is not None else None


def _raise_if_frozen(session: Any, portfolio_id: uuid.UUID, *, doing: str) -> None:
    """Raise ``RuntimeError`` when the fund or the portfolio is FROZEN (re-arm first).

    ``doing`` completes the message ("re-arm before ``doing``").
    """
    scope = _frozen_scope(session, portfolio_id)
    if scope == "fund":
        raise RuntimeError(f"fund autonomy is frozen; re-arm before {doing}")
    if scope == "portfolio":
        raise RuntimeError(f"portfolio {portfolio_id} is frozen; re-arm before {doing}")


def _raise_if_profile_superseded(session: Any, run: Any) -> None:
    """Raise ``RuntimeError`` when the portfolio was re-profiled after ``run`` opened.

    A gate parks the optimizer weights computed under the profile that was active when
    the round opened, and ``place_orders`` sizes the ticket from exactly those weights.
    A re-profile appends a new ``mifid_profiles`` version (the old one stays), so a
    gate left waiting would otherwise still be approvable and trade a book the new
    ``ConstraintSet`` never validated. A run opened before the version was recorded
    carries none and is not checked.

    Args:
        session: A sync ``portopt_db`` session; not owned here.
        run: The ``agent_run`` row being resumed; its ``optimizer_config`` holds the
            ``profile_version`` it was opened under.

    Raises:
        RuntimeError: A newer profile version exists; the message tells the adviser
            to reject the run and start a new round.
    """
    from fund.audit import MifidProfileRepository

    built_under = (run.optimizer_config or {}).get("profile_version")
    if built_under is None:
        return
    current = MifidProfileRepository(session).get_active(run.portfolio_id)
    if current is not None and current.version > built_under:
        raise RuntimeError(
            f"portfolio {run.portfolio_id} was re-profiled (v{built_under} -> "
            f"v{current.version}) after run {run.id} opened; reject this run and "
            "start a new round under the current profile"
        )


def _halt_if_frozen(
    session: Any, audit: Any, run_id: uuid.UUID, portfolio_id: uuid.UUID
) -> bool:
    """Finalise a run ``rejected`` when a freeze landed while it was in flight.

    The freeze verbs only see runs already committed at a gate, so a round still
    executing when the operator freezes would otherwise open its gate afterwards and
    be approvable under a FROZEN posture. Returns ``True`` when the run was halted.
    No commit.
    """
    scope = _frozen_scope(session, portfolio_id)
    if scope is None:
        return False
    audit.append_decision(
        run_id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "halted", "reason": "frozen", "scope": scope},
    )
    audit.finalize_run(run_id, weights={}, status="rejected")
    return True


@dataclass(frozen=True)
class _FundRunSetup:
    """The built agent + opened run + seed turn one fund round is driven from."""

    agent: Any = field(repr=False, compare=False)
    thread_config: dict[str, Any] = field(repr=False, compare=False)
    run: Any = field(repr=False, compare=False)
    constraint_set: ConstraintSet = field(repr=False, compare=False)
    seed_input: dict[str, Any] = field(repr=False, compare=False)
    audit: Any = field(repr=False, compare=False)


def _prepare_fund_run(
    model: Any,
    mandate: PortfolioMandate,
    *,
    portfolio_id: uuid.UUID | str,
    asof: dt.date,
    session: Any,
    checkpointer: Any,
    store: Any | None,
    config: FundConfig,
    fallback: Any | None,
    thread_id: str | None,
    seed: int | None,
) -> _FundRunSetup:
    """Resolve the constraint set, open the run, and build the PM agent for one round.

    The setup :func:`run_fund` and :func:`stream_fund` share so neither drifts from the
    other: a FROZEN fund or portfolio raises ``RuntimeError`` (re-arm first — Q3, fund
    scope wins); a missing active ``ConstraintSet`` raises ``RuntimeError`` (profile
    first); a pending ``agent_run`` is opened recording the seed + ``temperature`` + 3y
    lookback (D31) + the mandate's ``capital`` (the ticket's gross book) + the active
    MiFID ``profile_version`` (``None`` when unprofiled); its
    checkpointer thread is pinned to ``str(run_id)`` so the rebuild-to-resume path
    recovers it; and the PM agent is built against this run's
    ``RunContext`` with the effective gate set — the portfolio's **first** round forces
    every PM gate (``_ALL_ORCHESTRATOR_GATES``), later rounds use config baseline ∪
    ``mandate.hitl_gates``. The caller drives the agent (invoke or stream). Every write
    goes through ``session`` (caller owns the transaction).
    """
    import secrets

    from fund.agents.toolsets import RunContext
    from fund.audit import (
        AgentRunRepository,
        MifidProfileRepository,
        resolve_constraint_set,
    )
    from fund.audit.autonomy_repository import AutonomyRepository
    from fund.schemas import ConstraintSetRef

    pid_uuid = _coerce_uuid(portfolio_id)
    pid_str = str(pid_uuid)

    # FROZEN blocks every new run — scheduled and human-triggered — until an explicit
    # re-arm (Q3). Fund scope wins over portfolio scope. The scheduler pre-checks and
    # skips cleanly; this raise is the backstop that blocks a manual run before any run
    # row is opened. Read-only on purpose: a row created here would sit uncommitted in
    # the round's transaction for its whole run, and a concurrent freeze would then
    # wait on the unique index until the round ends instead of acting at once.
    _raise_if_frozen(session, pid_uuid, doing="running a rebalance")
    autonomy = AutonomyRepository(session)

    constraint_set = None
    if store is not None:
        constraint_set = resolve_constraint_set(
            store,
            ConstraintSetRef(
                portfolio_id=pid_str, store_key=config.constraint_set_store_key
            ),
        )
    if constraint_set is None:
        raise RuntimeError(
            f"No active ConstraintSet for portfolio {pid_str}; run the profiler "
            f"first to establish a risk profile."
        )

    # 31 bits: ``agent_runs.seed`` is a signed int4 on Postgres, and a 32-bit draw
    # overflows it on about half of all runs.
    run_seed = seed if seed is not None else secrets.randbits(31)
    active_profile = MifidProfileRepository(session).get_active(pid_uuid)
    audit = AgentRunRepository(session)
    run = audit.create_run(
        portfolio_id=pid_uuid,
        asof=asof,
        seed=run_seed,
        universe=[],
        optimizer_config={
            "step": "fund",
            "temperature": config.model_temperature,
            "lookback_days": _LOOKBACK_DAYS,
            # Recorded so the rebuild-to-resume path sizes the ticket at the capital
            # the PM was told, even if the mandate is edited before approval.
            "capital": float(mandate.capital),
            # Recorded so approving the gate can tell the portfolio was re-profiled
            # while the round was running or waiting.
            "profile_version": (
                active_profile.version if active_profile is not None else None
            ),
        },
    )
    resolved_thread_id = thread_id or str(run.id)
    audit.set_thread_id(run.id, resolved_thread_id)

    ctx = RunContext(
        session=session,
        asof=asof,
        store=store,
        config=config,
        run_id=run.id,
        portfolio_id=pid_uuid,
        base_currency=mandate.base_currency,
        notional=float(mandate.capital),
    )
    effective_gates = _effective_gates(config, mandate)
    if autonomy.consume_first_round(pid_uuid):
        # A portfolio's first round forces every PM gate for maximum oversight,
        # overriding the mandate; the flag is now consumed, so later rounds respect
        # the mandate's gate set.
        effective_gates = dict.fromkeys(_ALL_ORCHESTRATOR_GATES, True)
    agent = build_fund_agent(
        model,
        checkpointer=checkpointer,
        store=store,
        ctx=ctx,
        config=config,
        fallback=fallback,
        interrupt_on=effective_gates,
    )
    thread_config = {
        "configurable": {"thread_id": resolved_thread_id},
        "recursion_limit": config.recursion_limit,
    }
    seed_input = {
        "messages": [
            {
                "role": "user",
                "content": _run_instruction(
                    mandate,
                    pid_str,
                    asof,
                    _history_summary(session, pid_uuid, asof),
                    constraint_set=constraint_set,
                ),
            }
        ]
    }
    return _FundRunSetup(
        agent=agent,
        thread_config=thread_config,
        run=run,
        constraint_set=constraint_set,
        seed_input=seed_input,
        audit=audit,
    )


def run_fund(
    model: Any,
    mandate: PortfolioMandate,
    *,
    portfolio_id: uuid.UUID | str,
    asof: dt.date,
    session: Any,
    checkpointer: Any,
    store: Any | None = None,
    config: FundConfig = settings,
    fallback: Any | None = None,
    thread_id: str | None = None,
    seed: int | None = None,
) -> FundRun:
    """Run one paper rebalance to the adviser-confirmation gate.

    Orchestrates: (1) resolve the portfolio's active ``ConstraintSet`` from the
    Store via a Phase-4 ``ConstraintSetRef`` — **missing raises ``RuntimeError``**,
    routing the caller to the profiler (a fund run needs a risk profile). (2) open a
    pending ``agent_run`` recording the seed + ``temperature`` + 3y lookback for
    reproducibility (D31). (3) build the PM agent with per-run toolsets bound to this
    run's ``RunContext`` (``asof`` bounds every price read; no look-ahead). (4) invoke
    with the mandate + directive; the pipeline (economist → allocator → risk →
    executor) runs and the run pauses at the ``place_orders`` HITL gate.

    Returns a :class:`FundRun`. When the pipeline never reaches the gate — a blocking
    risk verdict, or the delegation round cap tripping the recursion limit — the run
    is finalised ``incomplete`` (no ticket) and surfaced for review instead. A freeze
    committed while the round was running finalises it ``rejected`` rather than
    opening a gate under a FROZEN posture.

    ``model`` must be a tool-calling chat model. Every write goes through the injected
    ``session`` (caller owns the transaction) and, when given, the ``store``.
    """
    from langgraph.errors import GraphRecursionError

    setup = _prepare_fund_run(
        model,
        mandate,
        portfolio_id=portfolio_id,
        asof=asof,
        session=session,
        checkpointer=checkpointer,
        store=store,
        config=config,
        fallback=fallback,
        thread_id=thread_id,
        seed=seed,
    )
    agent = setup.agent
    thread_config = setup.thread_config
    run = setup.run
    constraint_set = setup.constraint_set
    audit = setup.audit

    # The place_orders gate pauses the run for the adviser.
    try:
        result = agent.invoke(setup.seed_input, config=thread_config)
    except GraphRecursionError:
        # Round cap: the PM kept delegating past the bound without committing.
        _finalize_incomplete(audit, run.id, "round_cap")
        return FundRun(
            constraint_set=constraint_set,
            run_id=run.id,
            interrupt=None,
            weights={},
            status="incomplete",
            agent=agent,
            thread_config=thread_config,
            session=session,
        )

    interrupt = _extract_interrupt(result)
    weights = _optimizer_weights(session, run.id)
    if interrupt is None:
        # The pipeline ended before the HITL gate: the PM tripped the delegation round
        # cap (reason "pm_round_cap") or a blocking risk verdict left nothing to commit
        # (reason "no_order"). Keep both distinct from the recursion-limit "round_cap".
        reason = "pm_round_cap" if result.get("pm_round_cap_reached") else "no_order"
        _finalize_incomplete(audit, run.id, reason)
        return FundRun(
            constraint_set=constraint_set,
            run_id=run.id,
            interrupt=None,
            weights=weights,
            status="incomplete",
            agent=agent,
            thread_config=thread_config,
            session=session,
        )

    if _halt_if_frozen(session, audit, run.id, _coerce_uuid(portfolio_id)):
        return FundRun(
            constraint_set=constraint_set,
            run_id=run.id,
            interrupt=None,
            weights={},
            status="rejected",
            agent=agent,
            thread_config=thread_config,
            session=session,
        )

    # Flip the row to "paused" so observers (CLI/TUI) can find awaiting-approval
    # runs and the rebuild-to-resume path knows a gate is live.
    audit.mark_paused(run.id)
    return FundRun(
        constraint_set=constraint_set,
        run_id=run.id,
        interrupt=interrupt,
        weights=weights,
        status="paused",
        agent=agent,
        thread_config=thread_config,
        session=session,
    )


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    """The structured tool calls on a streamed model message (empty when none)."""
    calls = getattr(message, "tool_calls", None) or []
    return [call for call in calls if isinstance(call, dict)]


def _message_text(message: Any) -> str:
    """A streamed message's content as text (``""`` when absent or non-string)."""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _gate_why(interrupt_value: dict[str, Any] | None) -> str:
    """The HITL gate's 'why' from its first action request (safe default otherwise)."""
    requests = (interrupt_value or {}).get("action_requests") or []
    request = requests[0] if requests else {}
    return request.get("description") or "review before it executes"


def stream_fund(
    model: Any,
    mandate: PortfolioMandate,
    *,
    portfolio_id: uuid.UUID | str,
    asof: dt.date,
    session: Any,
    checkpointer: Any,
    store: Any | None = None,
    config: FundConfig = settings,
    fallback: Any | None = None,
    thread_id: str | None = None,
    seed: int | None = None,
) -> Iterator[StreamEvent]:
    """Stream one paper rebalance to the adviser gate, yielding ordered step events.

    The live-deliberation counterpart to :func:`run_fund`: it shares
    :func:`_prepare_fund_run`'s setup, then drives the PM agent with
    ``agent.stream(stream_mode="updates")`` and yields one
    :class:`~fund.tui.stream.StreamEvent` per meaningful pipeline step — each subagent
    report (economist → allocator → risk → executor), then the ``place_orders``
    commit — in emission order, filtering the interleaved middleware-hook chunks. The
    stream ends on exactly one terminal event: ``paused`` once the ``place_orders`` HITL
    gate is live (the run is flipped ``paused`` so the Decisions queue finds it), or
    ``incomplete`` on a round-cap / no-order end (the run is finalised ``incomplete``)
    or when a freeze landed mid-round (the run is finalised ``rejected``).

    The agent stack and the ``StreamEvent`` DTO are imported lazily, so a bare
    ``import fund.agents.graph`` stays agent-stack-free. Every write goes through
    ``session`` (caller owns the transaction). Not thread-safe — one stream per thread.

    Yields:
        :class:`~fund.tui.stream.StreamEvent` in emission order, terminated by a single
        ``paused`` / ``incomplete`` event.
    """
    from langgraph.errors import GraphRecursionError

    from fund.tui.stream import StepStatus, StreamEvent

    setup = _prepare_fund_run(
        model,
        mandate,
        portfolio_id=portfolio_id,
        asof=asof,
        session=session,
        checkpointer=checkpointer,
        store=store,
        config=config,
        fallback=fallback,
        thread_id=thread_id,
        seed=seed,
    )
    run_id = setup.run.id
    audit = setup.audit

    index = 0
    pending_sub: str | None = None
    interrupt_value: dict[str, Any] | None = None
    pm_round_cap_reached = False
    try:
        for chunk in setup.agent.stream(
            setup.seed_input, config=setup.thread_config, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue
            if "__interrupt__" in chunk:
                interrupt_value = _extract_interrupt(chunk)
                continue
            # Only the PM's own model/tools nodes carry the pipeline; the middleware
            # hooks (Skills/PMRoundCap/HumanInTheLoop/PatchToolCalls) are noise here.
            for node, update in chunk.items():
                if isinstance(update, dict) and update.get("pm_round_cap_reached"):
                    # Mirror run_fund: a PM-round-cap end must finalise as
                    # "pm_round_cap", not "no_order". The updates stream never carries
                    # the final merged state, so latch the flag as the cap's state
                    # update streams past.
                    pm_round_cap_reached = True
                messages = update.get("messages") if isinstance(update, dict) else None
                message = messages[-1] if messages else None
                if node == "model":
                    for call in _tool_calls(message):
                        name = call.get("name")
                        if name == "task":
                            # A delegation names its subagent; attribute the result that
                            # follows (the tools node) to it.
                            pending_sub = (call.get("args") or {}).get("subagent_type")
                        elif name == "place_orders":
                            yield StreamEvent(
                                kind="step",
                                run_id=run_id,
                                index=index,
                                agent="orchestrator",
                                step="place_orders",
                                status=StepStatus.LIVE,
                                text="committing the risk-approved trades",
                            )
                            index += 1
                elif node == "tools" and getattr(message, "name", None) == "task":
                    yield StreamEvent(
                        kind="step",
                        run_id=run_id,
                        index=index,
                        agent=pending_sub or "subagent",
                        step="report",
                        status=StepStatus.LIVE,
                        text=_message_text(message),
                    )
                    index += 1
    except GraphRecursionError:
        _finalize_incomplete(audit, run_id, "round_cap")
        yield StreamEvent(
            kind="incomplete",
            run_id=run_id,
            index=index,
            agent="orchestrator",
            step="place_orders",
            status=StepStatus.PENDING,
            text="round cap reached before a ticket",
        )
        return

    if interrupt_value is None:
        reason = "pm_round_cap" if pm_round_cap_reached else "no_order"
        _finalize_incomplete(audit, run_id, reason)
        yield StreamEvent(
            kind="incomplete",
            run_id=run_id,
            index=index,
            agent="orchestrator",
            step="place_orders",
            status=StepStatus.PENDING,
            text=(
                "delegation round cap reached before a ticket"
                if pm_round_cap_reached
                else "pipeline ended before the gate"
            ),
        )
        return

    if _halt_if_frozen(session, audit, run_id, _coerce_uuid(portfolio_id)):
        yield StreamEvent(
            kind="incomplete",
            run_id=run_id,
            index=index,
            agent="orchestrator",
            step="place_orders",
            status=StepStatus.PENDING,
            text="halted: autonomy was frozen while the round was running",
        )
        return

    # Flip to "paused" so the Decisions queue / observers find the live gate.
    audit.mark_paused(run_id)
    yield StreamEvent(
        kind="paused",
        run_id=run_id,
        index=index,
        agent="orchestrator",
        step="place_orders",
        status=StepStatus.PENDING,
        text=_gate_why(interrupt_value),
        payload=interrupt_value,
    )


def _ticket_capital(run: Any, mandate: PortfolioMandate | None) -> float | None:
    """Capital the paper ticket is priced at when a gate resumes.

    The capital recorded when the run opened wins: it is the book the PM was told
    to allocate, so a mandate edited while the gate was pending must not silently
    re-price the ticket. Runs opened before capital was recorded fall back to the
    current mandate, and ``None`` leaves the paper-cost model's default notional.

    Args:
        run: The persisted ``agent_run`` row being resumed.
        mandate: The portfolio's current mandate, or ``None`` if it vanished.

    Returns:
        The notional to size the ticket with, or ``None`` for the default book.
    """
    recorded = (run.optimizer_config or {}).get("capital")
    if recorded is not None:
        return float(recorded)
    return float(mandate.capital) if mandate is not None else None


def resume_fund(
    run_id: uuid.UUID | str,
    decision: str,
    *,
    session: Any,
    checkpointer: Any,
    store: Any | None,
    model: Any,
    config: FundConfig = settings,
    fallback: Any | None = None,
    pm_directive: str | None = None,
) -> dict[str, Any]:
    """Resume a paused fund run from a fresh process — the rebuild-to-resume path.

    :class:`FundRun` is not serialisable across processes (its ``agent``/``session``
    are live objects), so an observer (the CLI ``approve``/``reject`` commands, the
    TUI HITL queue) resumes a paused ``place_orders`` gate by **rebuilding** the PM
    agent against the *same* ``checkpointer`` + ``thread_id`` and issuing
    ``Command(resume=…)`` itself. Approving executes the interrupted commit inside
    the graph (so a ``model`` is required to resume); the shared :func:`_settle_hitl`
    then stamps the terminal audit state and, on an approve whose ticket was placed, the
    positions snapshot — leaving DB state identical to :meth:`FundRun.resume`. A PM
    follow-up turn that fails after the gate was consumed does not raise: the decision
    stands and the failure is returned as ``{"resume_error": ...}``.

    Loads the ``agent_run`` to recover its ``thread_id`` / ``portfolio_id`` / ``asof``
    (an unknown ``run_id`` raises ``LookupError``), re-guards the portfolio's active
    ``ConstraintSet`` from the Store (missing raises ``RuntimeError`` — a run whose
    profile vanished cannot be committed), re-reads the mandate so a gate set edited
    since the run opened applies on resume too (a vanished mandate falls back to the
    config default gate), rebuilds the ``RunContext`` + PM agent, resumes, and
    finalises. ``decision`` is ``approve``, ``reject``, or ``respond``.
    Any other value raises ``ValueError``; ``approve`` / ``respond`` raise
    ``RuntimeError`` while the fund or portfolio is FROZEN, or when the portfolio was
    re-profiled after the run opened (``reject`` always passes). A ``respond`` threads
    ``pm_directive`` (the adviser's free text) back to the PM via
    :func:`_finalize_respond`, keeping the run
    alive as ``responded`` if it re-gates; the resume ``thread_config`` keeps the full
    ``recursion_limit`` so a re-gate has headroom. Every write goes through the injected
    ``session`` (caller owns the transaction); the agent stack is imported lazily so a
    bare ``import fund.agents.graph`` stays clean.
    """
    if decision not in ("approve", "reject", "respond"):
        raise ValueError(
            f"decision must be 'approve', 'reject', or 'respond', not {decision!r}"
        )

    from fund.agents.toolsets import RunContext
    from fund.audit import AgentRunRepository, resolve_constraint_set
    from fund.schemas import ConstraintSetRef

    rid = _coerce_uuid(run_id)
    audit = AgentRunRepository(session)
    run = audit.get_run(rid)
    if run is None:
        raise LookupError(f"no agent_run {rid}")
    if run.portfolio_id is None:
        raise ValueError(f"agent_run {rid} has no portfolio_id")
    # A run already finalised must not be resumed. An external freeze marks live gates
    # "rejected" at the DB without resolving the checkpoint (a completed / incomplete /
    # blocked run, or a stale observer racing a concurrent approve, is the same shape):
    # the LangGraph checkpoint is left standing, so without this guard a blind
    # Command(resume=…) would re-drive place_orders and ship live orders against a
    # halted run. Pre-gate states (pending/paused/responded) stay resumable.
    if run.status in ("completed", "rejected", "incomplete", "blocked"):
        raise ValueError(
            f"agent_run {rid} is {run.status!r}, already finalised and not resumable "
            "(it completed, or was rejected/frozen/blocked since the gate opened)"
        )

    # set_holdings replaces the whole snapshot, so approving a round that started before
    # an already-completed one would silently roll the book back to older weights.
    # Rejecting a stale run stays allowed: that is how an adviser clears it.
    if decision == "approve":
        newer = audit.newer_completed_rebalance_run(run)
        if newer is not None:
            raise ValueError(
                f"agent_run {rid} is stale: newer run {newer.id} for portfolio "
                f"{run.portfolio_id} already completed, and approving would overwrite "
                "its holdings with older weights; reject this run instead"
            )

    portfolio_id = run.portfolio_id
    pid_str = str(portfolio_id)
    resolved_thread_id = run.thread_id or str(rid)

    # A gate that opened before (or raced) a freeze, or under a since-replaced profile,
    # stays approvable at the DB level, so approve/respond re-check both. Reject is the
    # way out and always passes.
    if decision != "reject":
        _raise_if_frozen(session, portfolio_id, doing="resuming a run")
        _raise_if_profile_superseded(session, run)

    # Re-guard the active ConstraintSet — the run's risk profile must still exist.
    constraint_set = None
    if store is not None:
        constraint_set = resolve_constraint_set(
            store,
            ConstraintSetRef(
                portfolio_id=pid_str, store_key=config.constraint_set_store_key
            ),
        )
    if constraint_set is None:
        raise RuntimeError(
            f"No active ConstraintSet for portfolio {pid_str}; run the profiler "
            f"first to establish a risk profile."
        )

    # Re-read the mandate so a gate set edited since the run opened takes effect on
    # resume too (mirrors the ConstraintSet re-guard above); a vanished mandate falls
    # back to the config default gate and the tool's default notional.
    from fund import observe

    mandate = observe.get_mandate(session, portfolio_id)
    ctx = RunContext(
        session=session,
        asof=run.asof,
        store=store,
        config=config,
        run_id=rid,
        portfolio_id=portfolio_id,
        base_currency=mandate.base_currency if mandate is not None else None,
        notional=_ticket_capital(run, mandate),
    )
    agent = build_fund_agent(
        model,
        checkpointer=checkpointer,
        store=store,
        ctx=ctx,
        config=config,
        fallback=fallback,
        interrupt_on=_effective_gates(config, mandate) if mandate is not None else None,
    )
    thread_config = {
        "configurable": {"thread_id": resolved_thread_id},
        "recursion_limit": config.recursion_limit,
    }
    try:
        result = _resume_agent(agent, thread_config, decision, pm_directive)
    except Exception as exc:
        # The checkpointer commits on its own connection while the ticket and
        # positions live in the caller's uncommitted session. If the PM's follow-up
        # turn fails after the gate was consumed, propagating would roll the ticket
        # back yet leave the checkpoint past the tool step: the run would stay
        # "paused" with no pending interrupt, gone from the queue and unretryable.
        # So once the gate is consumed the decision stands and is finalised; a gate
        # still pending means nothing moved and the failure is safely retryable.
        if observe._has_pending_interrupt(checkpointer, resolved_thread_id):
            raise
        _log.warning(
            "resume of run %s failed after the gate was consumed", rid, exc_info=True
        )
        result = {"resume_error": f"{type(exc).__name__}: {exc}"}
    if decision == "respond":
        _finalize_respond(session, run_id=rid, result=result, pm_directive=pm_directive)
        return result
    _settle_hitl(
        session,
        run_id=rid,
        portfolio_id=portfolio_id,
        asof=run.asof,
        weights=_optimizer_weights(session, rid),
        decision=decision,
        result=result,
    )
    return result


def resume_run(
    run_id: uuid.UUID | str,
    decision: str,
    *,
    session: Any,
    checkpointer: Any,
    store: Any | None,
    model: Any,
    config: FundConfig = settings,
    fallback: Any | None = None,
    pm_directive: str | None = None,
) -> dict[str, Any]:
    """Resume a paused run by its gate — the single entry point observers call.

    A run pauses at one of two HITL gates: the profiler's ``save_profile`` (step
    ``"profiler"``) or the rebalance PM's ``place_orders`` (step ``"fund"``). This
    dispatches on the run's recorded ``optimizer_config["step"]`` to
    :func:`~fund.agents.profiler.resume_profiler` or :func:`resume_fund`, so the CLI
    ``approve``/``reject`` and the TUI HITL queue drive either gate through one call
    and cannot diverge on which path a given run takes. An unknown ``run_id`` raises
    ``LookupError``; ``decision`` validation is delegated to the chosen resumer.

    ``pm_directive`` carries the adviser's free text for a ``respond`` barge-in; it is
    threaded only to the ``place_orders`` gate. The profiler gate keeps approve/reject
    (``resume_profiler`` rejects ``respond``), so a respond there raises ``ValueError``.
    """
    from fund.audit import AgentRunRepository

    rid = _coerce_uuid(run_id)
    run = AgentRunRepository(session).get_run(rid)
    if run is None:
        raise LookupError(f"no agent_run {rid}")

    if (run.optimizer_config or {}).get("step") == "profiler":
        from fund.agents.profiler import resume_profiler

        return resume_profiler(
            rid,
            decision,
            session=session,
            checkpointer=checkpointer,
            store=store,
            model=model,
            config=config,
        )
    return resume_fund(
        rid,
        decision,
        session=session,
        checkpointer=checkpointer,
        store=store,
        model=model,
        config=config,
        fallback=fallback,
        pm_directive=pm_directive,
    )
