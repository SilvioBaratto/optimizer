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

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, cast

from fund.agents.backend import build_backend
from fund.agents.prompts import (
    ALLOCATOR_SYSTEM_PROMPT,
    ECONOMIST_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
)
from fund.agents.skills import skill_sources
from fund.config import FundConfig, settings

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterator

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

# The moments/backtest lookback recorded on every run (mirrors the
# ``RunContext.lookback_days`` default — 3y rolling, ~756 trading days).
_LOOKBACK_DAYS = 756

# One-line descriptions the PM reads (via the built-in ``task`` tool) to decide
# when to delegate — each names the role's fixed pipeline position.
_ECONOMIST_DESCRIPTION = (
    "Pipeline step 1. Reads the macro and price backdrop and hands the allocator a "
    "qualitative regime narrative plus candidate views. Delegate first."
)
_ALLOCATOR_DESCRIPTION = (
    "Pipeline step 2. Filters the universe, estimates moments folding in the "
    "economist's views, and runs the optimizer to produce candidate weights under "
    "the resolved constraints. Delegate after the economist."
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
) -> dict[str, Any]:
    """Build one ``SubAgent`` dict for ``role`` from the per-run context.

    Carries the role's own per-run tools (``session`` already bound) and its
    root-relative skills; never sets ``interrupt_on``. When ``fallback`` is given
    (critical roles only), attaches a single ``ModelFallbackMiddleware`` so a
    primary-model failure retries on the fallback (a langchain middleware, not a
    hand-rolled ``.with_fallbacks``).
    """
    from fund.agents.toolsets import bind_toolset

    sub: dict[str, Any] = {
        "name": role,
        "description": description,
        "system_prompt": system_prompt,
        "tools": bind_toolset(role, ctx),
        "skills": skill_sources(role),
    }
    if fallback is not None:
        from langchain.agents.middleware import ModelFallbackMiddleware

        sub["middleware"] = [ModelFallbackMiddleware(fallback)]
    return sub


def build_economist_subagent(ctx: RunContext) -> dict[str, Any]:
    """The economist subagent (step 1) — narrative regime read, no fallback."""
    return _subagent(
        "economist",
        description=_ECONOMIST_DESCRIPTION,
        system_prompt=ECONOMIST_SYSTEM_PROMPT,
        ctx=ctx,
    )


def build_allocator_subagent(
    ctx: RunContext, *, fallback: Any | None = None
) -> dict[str, Any]:
    """The allocator subagent (step 2) — the optimizer path; critical role."""
    return _subagent(
        "allocator",
        description=_ALLOCATOR_DESCRIPTION,
        system_prompt=ALLOCATOR_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
    )


def build_risk_subagent(
    ctx: RunContext, *, fallback: Any | None = None
) -> dict[str, Any]:
    """The risk subagent (step 3) — the blocking gate; critical role."""
    return _subagent(
        "risk",
        description=_RISK_DESCRIPTION,
        system_prompt=RISK_SYSTEM_PROMPT,
        ctx=ctx,
        fallback=fallback,
    )


def build_executor_subagent(
    ctx: RunContext, *, fallback: Any | None = None
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
    )


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
    ``fallback`` middleware when one is supplied. Skills are the root-relative
    staged sources (Risk R1) resolved through the ``virtual_mode`` ``backend``.
    ``temperature=0`` / the DeepSeek route ride on ``model`` (built from
    ``FundConfig``), not passed here.
    """
    from deepagents import create_deep_agent

    from fund.agents.toolsets import bind_toolset

    gates = interrupt_on if interrupt_on is not None else config.interrupt_on_map()
    subagents = [
        build_economist_subagent(ctx),
        build_allocator_subagent(ctx, fallback=fallback),
        build_risk_subagent(ctx, fallback=fallback),
        build_executor_subagent(ctx, fallback=fallback),
    ]
    return create_deep_agent(
        model=model,
        tools=bind_toolset("orchestrator", ctx),
        system_prompt=PM_SYSTEM_PROMPT,
        subagents=cast("list[Any]", subagents),
        skills=skill_sources("orchestrator"),
        backend=build_backend(config),
        interrupt_on=cast("dict[str, Any]", gates),
        middleware=[_pm_round_cap_middleware(config.max_pm_rounds)],
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

    Used by :meth:`FundRun.resume` (the live in-process handle) **and**
    :func:`resume_fund` (the cross-process rebuild), so the two cannot diverge.
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
    session: Any, *, portfolio_id: uuid.UUID, asof: dt.date, weights: dict[str, float]
) -> bool:
    """Whether a paper ticket exists for this bar's audited ``weights``.

    Uses the same ``(portfolio_id, asof, weights_hash)`` key ``place_orders`` writes
    under, so an idempotent re-run that returned the stored ticket counts as placed.
    """
    from fund.audit import OrderRepository
    from fund.tools.orders import _weights_hash

    ticket = OrderRepository(session).get_by_key(
        portfolio_id=portfolio_id, asof=asof, weights_hash=_weights_hash(weights)
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

    ``place_orders`` reports a missing fill bar or unknown ticker as an ``{ok: false}``
    envelope instead of raising, so resuming the gate succeeds even when no ticket was
    written. An approve with no ticket must not stamp the run ``completed`` nor replace
    the ``positions`` snapshot with weights no order backs: it finalises ``incomplete``
    (``"place_orders_failed"``), or, when the resumed PM re-gated, is left awaiting the
    adviser. Reject and a ticketed approve go through :func:`_finalize_hitl`. No commit.
    """
    from fund.audit import AgentRunRepository

    if decision == "approve" and not _ticket_placed(
        session, portfolio_id=portfolio_id, asof=asof, weights=weights
    ):
        if _extract_interrupt(result) is None:
            _finalize_incomplete(
                AgentRunRepository(session), run_id, "place_orders_failed"
            )
        return
    _finalize_hitl(
        session,
        run_id=run_id,
        portfolio_id=portfolio_id,
        asof=asof,
        weights=weights,
        decision=decision,
    )


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


def _run_instruction(
    mandate: PortfolioMandate,
    portfolio_id: str,
    asof: dt.date,
    history_summary: str = "",
) -> str:
    """The human turn that drives the PM through one paper rebalance.

    When ``history_summary`` is non-empty it is prepended as a context block so
    the PM sees the portfolio's recent allocation/trades and the latest market
    digest before deciding; an empty summary (first run) yields the bare seed.
    """
    benchmark = f", benchmark {mandate.benchmark}" if mandate.benchmark else ""
    history = f"{history_summary}\n\n" if history_summary else ""
    return (
        f"{history}Produce an allocation for portfolio {portfolio_id} as of "
        f"{asof.isoformat()}. Mandate: capital {mandate.capital} "
        f"{mandate.base_currency}{benchmark}. Delegate in order economist -> "
        f"allocator -> risk -> executor, then commit the risk-approved trades "
        f"through place_orders."
    )


@dataclass(frozen=True)
class FundRun:
    """Handle for one fund run paused at the ``place_orders`` adviser gate.

    Mirrors :class:`~fund.agents.profiler.ProfilerRun`. Carries the resolved
    ``constraint_set``, the load-bearing optimizer ``weights`` (captured from the
    audit trail before the pause), the ``interrupt`` payload the adviser reviews,
    the run ``status``, and enough state to resume. ``resume("approve")`` commits
    the paper ticket and finalises the run completed; any other decision commits no
    order, records the HITL choice, and finalises the run rejected.
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
        as completed with the optimizer's weights. ``reject`` writes no order, records
        the HITL choice, and finalises rejected. ``respond`` threads ``pm_directive``
        back to the PM and lets it continue: if it re-gates the run stays alive as
        ``"responded"`` (still awaiting approval), else it finalises ``incomplete`` —
        delegated to :func:`_finalize_respond`.

        Approve/reject finalisation is delegated to the shared :func:`_finalize_hitl`,
        and respond to :func:`_finalize_respond`, so this in-process path and the
        cross-process :func:`resume_fund` cannot diverge — both stamp the audit trail
        identically. The run's ``portfolio_id`` / ``asof`` (which ``FundRun`` does not
        carry) are read back from its persisted row for the approve/reject path.
        """
        from fund.audit import AgentRunRepository

        result: dict[str, Any] = self.agent.invoke(
            _resume_command(decision, pm_directive),
            config=self.thread_config,
        )
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
            weights=self.weights,
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


def _resume_command(decision: str, pm_directive: str | None) -> Any:
    """Build the HITL ``Command(resume=…)`` for a gate decision.

    ``approve`` / ``reject`` carry only the decision type; ``respond`` carries the
    adviser's free-text ``pm_directive`` as the gated tool's result, threading it back
    to the PM (a successful ``place_orders`` ``ToolMessage``) so the gate continues
    instead of terminating. Imports ``langgraph`` lazily to keep a bare
    ``import fund.agents.graph`` light.
    """
    from langgraph.types import Command

    if decision == "respond":
        return Command(
            resume={"decisions": [{"type": "respond", "message": pm_directive}]}
        )
    return Command(resume={"decisions": [{"type": decision}]})


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
    lookback (D31); its checkpointer thread is pinned to ``str(run_id)`` so the
    rebuild-to-resume path recovers it; and the PM agent is built against this run's
    ``RunContext`` with the effective gate set — the portfolio's **first** round forces
    every PM gate (``_ALL_ORCHESTRATOR_GATES``), later rounds use config baseline ∪
    ``mandate.hitl_gates``. The caller drives the agent (invoke or stream). Every write
    goes through ``session`` (caller owns the transaction).
    """
    import secrets

    from fund.agents.toolsets import RunContext
    from fund.audit import AgentRunRepository, resolve_constraint_set
    from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState
    from fund.schemas import ConstraintSetRef

    pid_uuid = _coerce_uuid(portfolio_id)
    pid_str = str(pid_uuid)

    # FROZEN blocks every new run — scheduled and human-triggered — until an explicit
    # re-arm (Q3). Fund scope wins over portfolio scope. The scheduler pre-checks and
    # skips cleanly; this raise is the backstop that blocks a manual run before any run
    # row is opened.
    autonomy = AutonomyRepository(session)
    if autonomy.fund_state().state == AutonomyState.FROZEN:
        raise RuntimeError("fund autonomy is frozen; re-arm before running a rebalance")
    if autonomy.portfolio_state(pid_uuid).state == AutonomyState.FROZEN:
        raise RuntimeError(
            f"portfolio {pid_str} is frozen; re-arm before running a rebalance"
        )

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

    run_seed = seed if seed is not None else secrets.randbits(32)
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
                    mandate, pid_str, asof, _history_summary(session, pid_uuid, asof)
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
    is finalised ``incomplete`` (no ticket) and surfaced for review instead.

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
    ``incomplete`` on a round-cap / no-order end (the run is finalised ``incomplete``).

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
    the graph (so a ``model`` is required to resume); the shared
    :func:`_finalize_hitl` then stamps the terminal audit state and, on approve, the
    positions snapshot — leaving DB state identical to :meth:`FundRun.resume`.

    Loads the ``agent_run`` to recover its ``thread_id`` / ``portfolio_id`` / ``asof``
    (an unknown ``run_id`` raises ``LookupError``), re-guards the portfolio's active
    ``ConstraintSet`` from the Store (missing raises ``RuntimeError`` — a run whose
    profile vanished cannot be committed), re-reads the mandate so a gate set edited
    since the run opened applies on resume too (a vanished mandate falls back to the
    config default gate), rebuilds the ``RunContext`` + PM agent, resumes, and
    finalises. ``decision`` is ``approve``, ``reject``, or ``respond``.
    Any other value raises ``ValueError``. A ``respond`` threads ``pm_directive`` (the
    adviser's free text) back to the PM via :func:`_finalize_respond`, keeping the run
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

    portfolio_id = run.portfolio_id
    pid_str = str(portfolio_id)
    resolved_thread_id = run.thread_id or str(rid)

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
    # back to the config default gate.
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
    result: dict[str, Any] = agent.invoke(
        _resume_command(decision, pm_directive),
        config=thread_config,
    )
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
