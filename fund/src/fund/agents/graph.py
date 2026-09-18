"""T7.5 — Phase-7 orchestration graph: the subagents + the PM deep agent.

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
  ``interrupt_on={"place_orders": True}``.

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

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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

    # ``fund.agents.toolsets`` reaches ``fund.tools`` (the agent stack) at import;
    # keep it type-only here and import ``bind_toolset`` lazily inside the builders
    # so a bare ``import fund.agents.graph`` stays agent-stack-free (profiler style).
    from fund.agents.toolsets import RunContext
    from fund.schemas import ConstraintSet
    from fund.schemas.mandate import PortfolioMandate

__all__ = [
    "FundRun",
    "build_allocator_subagent",
    "build_economist_subagent",
    "build_executor_subagent",
    "build_fund_agent",
    "build_risk_subagent",
    "run_fund",
]

# D18: the moments/backtest lookback recorded on every run (mirrors the
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


def build_fund_agent(
    model: Any,
    *,
    checkpointer: Any,
    store: Any | None,
    ctx: RunContext,
    config: FundConfig = settings,
    fallback: Any | None = None,
) -> Any:
    """Assemble the PM ``deepagents`` agent that orchestrates one fund run.

    The PM runs under ``PM_SYSTEM_PROMPT`` with the orchestrator toolset — the
    single top-level ``place_orders`` tool, gated by ``interrupt_on`` (which
    requires the ``checkpointer``) so the commit always pauses for the adviser. It
    delegates through the built-in ``task`` tool to the four subagents in pipeline
    order (economist → allocator → risk → executor); the critical three carry the
    ``fallback`` middleware when one is supplied. Skills are the root-relative
    staged sources (Risk R1) resolved through the ``virtual_mode`` ``backend``.
    ``temperature=0`` / the DeepSeek route ride on ``model`` (built from
    ``FundConfig``), not passed here.
    """
    from deepagents import create_deep_agent

    from fund.agents.toolsets import bind_toolset

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
        subagents=subagents,
        skills=skill_sources("orchestrator"),
        backend=build_backend(config),
        interrupt_on=config.interrupt_on_map(),
        checkpointer=checkpointer,
        store=store,
    )


# ---------------------------------------------------------------------------
# Task 6 — ``run_fund`` + ``FundRun``: the one complete paper-run path.
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
    mapped optimizer config + the optimizer's weights) inside its closure during
    the initial invoke. Reading them back from the audit trail — never from what
    the PM/LLM passed — keeps ``optimize_portfolio`` the single source of weights.
    Returns ``{}`` when the allocator never produced a proposal.
    """
    from portopt_db.models import AgentDecision
    from sqlalchemy import select

    stmt = (
        select(AgentDecision)
        .where(
            AgentDecision.run_id == run_id,
            AgentDecision.agent == "allocator",
            AgentDecision.step == "optimize_portfolio",
        )
        .order_by(AgentDecision.decision_index.desc())
    )
    row = session.execute(stmt).scalars().first()
    if row is None or not row.llm_response:
        return {}
    try:
        payload = json.loads(row.llm_response)
    except (ValueError, TypeError):
        return {}
    weights = payload.get("weights") or {}
    return {str(k): float(v) for k, v in weights.items()}


def _run_instruction(
    mandate: PortfolioMandate, portfolio_id: str, asof: dt.date
) -> str:
    """The human turn that drives the PM through one paper rebalance."""
    benchmark = f", benchmark {mandate.benchmark}" if mandate.benchmark else ""
    return (
        f"Produce an allocation for portfolio {portfolio_id} as of "
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

    def resume(self, decision: str) -> dict[str, Any]:
        """Resume the paused agent with an adviser ``decision`` (approve / reject).

        ``approve`` runs the gated ``place_orders`` tool (the paper ticket is written
        and the executor decision logged inside the closure), then finalises the run
        as completed with the optimizer's weights. Any other decision writes no
        order, records the HITL choice in the audit trail, and finalises rejected.
        """
        from langgraph.types import Command

        from fund.audit import AgentRunRepository

        result: dict[str, Any] = self.agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            config=self.thread_config,
        )
        audit = AgentRunRepository(self.session)
        if decision == "approve":
            audit.finalize_run(self.run_id, weights=self.weights, status="completed")
        else:
            audit.append_decision(
                self.run_id,
                agent="orchestrator",
                step="place_orders",
                hitl_decision={"decision": decision},
            )
            audit.finalize_run(self.run_id, weights={}, status="rejected")
        return result


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
    import secrets

    from langgraph.errors import GraphRecursionError

    from fund.agents.toolsets import RunContext
    from fund.audit import AgentRunRepository, resolve_constraint_set
    from fund.schemas import ConstraintSetRef

    pid_uuid = _coerce_uuid(portfolio_id)
    pid_str = str(pid_uuid)

    # (1) Resolve the active ConstraintSet first — fail fast, no orphan run.
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

    # (2) Open the pending run; record seed + temperature=0 + lookback (D31).
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

    # (3) Bind the per-run toolsets and assemble the PM agent.
    ctx = RunContext(
        session=session,
        asof=asof,
        store=store,
        config=config,
        run_id=run.id,
        portfolio_id=pid_uuid,
    )
    agent = build_fund_agent(
        model,
        checkpointer=checkpointer,
        store=store,
        ctx=ctx,
        config=config,
        fallback=fallback,
    )
    thread_config = {
        "configurable": {"thread_id": thread_id or pid_str},
        "recursion_limit": config.recursion_limit,
    }

    # (4) Invoke; the place_orders gate pauses the run for the adviser.
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": _run_instruction(mandate, pid_str, asof),
                    }
                ]
            },
            config=thread_config,
        )
    except GraphRecursionError:
        # Round cap: the PM kept delegating past the bound without committing. Stop,
        # finalise incomplete, and surface the open issue for the adviser (D22).
        audit.append_decision(
            run.id,
            agent="orchestrator",
            step="place_orders",
            hitl_decision={"decision": "incomplete", "reason": "round_cap"},
        )
        audit.finalize_run(run.id, weights={}, status="incomplete")
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
        # The pipeline ended before the HITL gate — a blocking risk verdict left
        # nothing to commit. Finalise incomplete; the executor never proposed.
        audit.append_decision(
            run.id,
            agent="orchestrator",
            step="place_orders",
            hitl_decision={"decision": "incomplete", "reason": "no_order"},
        )
        audit.finalize_run(run.id, weights={}, status="incomplete")
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
