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

from typing import TYPE_CHECKING, Any

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
    # ``fund.agents.toolsets`` reaches ``fund.tools`` (the agent stack) at import;
    # keep it type-only here and import ``bind_toolset`` lazily inside the builders
    # so a bare ``import fund.agents.graph`` stays agent-stack-free (profiler style).
    from fund.agents.toolsets import RunContext

__all__ = [
    "build_allocator_subagent",
    "build_economist_subagent",
    "build_executor_subagent",
    "build_fund_agent",
    "build_risk_subagent",
]

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
