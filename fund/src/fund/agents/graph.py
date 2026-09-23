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
    "resume_fund",
    "resume_run",
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
        middleware=[_pm_round_cap_middleware(config.max_pm_rounds)],
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
    effective optimizer config + the optimizer's weights) inside its closure during
    the initial invoke. Reading them back from the audit trail — never from what
    the PM/LLM passed — keeps ``optimize_portfolio`` the single source of weights.
    Delegates to :meth:`AgentRunRepository.latest_optimizer_weights` so the paper
    ticket and the finalised ``agent_runs.weights`` derive from one source and
    cannot diverge. Returns ``{}`` when the allocator never produced a proposal.
    """
    from fund.audit import AgentRunRepository

    return AgentRunRepository(session).latest_optimizer_weights(run_id)


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
        holdings = order_lines or [
            {"ticker": ticker, "weight": weight} for ticker, weight in weights.items()
        ]
        PositionRepository(session).set_holdings(portfolio_id, holdings, asof=asof)
    else:
        audit.append_decision(
            run_id,
            agent="orchestrator",
            step="place_orders",
            hitl_decision={"decision": decision},
        )
        audit.finalize_run(run_id, weights={}, status="rejected")


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

        Finalisation is delegated to the shared :func:`_finalize_hitl` so this
        in-process path and the cross-process :func:`resume_fund` cannot diverge —
        both stamp the audit trail identically and, on approve, upsert positions.
        The run's ``portfolio_id`` / ``asof`` (which ``FundRun`` does not carry) are
        read back from its persisted row.
        """
        from langgraph.types import Command

        from fund.audit import AgentRunRepository

        result: dict[str, Any] = self.agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            config=self.thread_config,
        )
        run = AgentRunRepository(self.session).get_run(self.run_id)
        _finalize_hitl(
            self.session,
            run_id=self.run_id,
            portfolio_id=run.portfolio_id,
            asof=run.asof,
            weights=self.weights,
            decision=decision,
        )
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
    # Per-run thread (Phase 8): default the checkpointer thread to str(run_id) (was
    # str(portfolio_id)) so each run keeps its own verbatim transcript and the
    # rebuild-to-resume path can recover it. Persist it on the new nullable column.
    resolved_thread_id = thread_id or str(run.id)
    audit.set_thread_id(run.id, resolved_thread_id)

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
        "configurable": {"thread_id": resolved_thread_id},
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
        # The pipeline ended before the HITL gate. Two distinct causes: the PM
        # tripped the delegation round cap (PMRoundCapMiddleware jumped to end) —
        # reason "pm_round_cap" — or a blocking risk verdict left nothing to commit
        # and the executor never proposed — reason "no_order". Both finalise
        # incomplete; keep the reason distinct from the recursion-limit "round_cap".
        reason = "pm_round_cap" if result.get("pm_round_cap_reached") else "no_order"
        audit.append_decision(
            run.id,
            agent="orchestrator",
            step="place_orders",
            hitl_decision={"decision": "incomplete", "reason": reason},
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
    profile vanished cannot be committed), rebuilds the ``RunContext`` + PM agent,
    resumes, and finalises. ``decision`` must be ``"approve"`` or ``"reject"`` (any
    other raises ``ValueError`` — HITL edit is an ask-first future item). Every write
    goes through the injected ``session`` (caller owns the transaction). The agent
    stack is imported lazily so a bare ``import fund.agents.graph`` stays clean.
    """
    if decision not in ("approve", "reject"):
        raise ValueError(f"decision must be 'approve' or 'reject', not {decision!r}")

    from langgraph.types import Command

    from fund.agents.toolsets import RunContext
    from fund.audit import AgentRunRepository, resolve_constraint_set
    from fund.schemas import ConstraintSetRef

    rid = _coerce_uuid(run_id)
    audit = AgentRunRepository(session)
    run = audit.get_run(rid)
    if run is None:
        raise LookupError(f"no agent_run {rid}")

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

    ctx = RunContext(
        session=session,
        asof=run.asof,
        store=store,
        config=config,
        run_id=rid,
        portfolio_id=portfolio_id,
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
        "configurable": {"thread_id": resolved_thread_id},
        "recursion_limit": config.recursion_limit,
    }
    result: dict[str, Any] = agent.invoke(
        Command(resume={"decisions": [{"type": decision}]}),
        config=thread_config,
    )
    _finalize_hitl(
        session,
        run_id=rid,
        portfolio_id=portfolio_id,
        asof=run.asof,
        weights=_optimizer_weights(session, rid),
        decision=decision,
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
) -> dict[str, Any]:
    """Resume a paused run by its gate — the single entry point observers call.

    A run pauses at one of two HITL gates: the profiler's ``save_profile`` (step
    ``"profiler"``) or the rebalance PM's ``place_orders`` (step ``"fund"``). This
    dispatches on the run's recorded ``optimizer_config["step"]`` to
    :func:`~fund.agents.profiler.resume_profiler` or :func:`resume_fund`, so the CLI
    ``approve``/``reject`` and the TUI HITL queue drive either gate through one call
    and cannot diverge on which path a given run takes. An unknown ``run_id`` raises
    ``LookupError``; ``decision`` validation is delegated to the chosen resumer.
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
    )
