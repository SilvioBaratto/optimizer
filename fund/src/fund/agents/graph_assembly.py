"""The shared subagent factory and the caps the PM deep agent is built with.

:func:`_subagent` builds the spec each role builder in :mod:`fund.agents.graph`
returns; the general-purpose helper and the model-call and round caps complete
what :func:`fund.agents.graph.build_fund_agent` wires into the PM. The agent
stack is imported inside these functions, so importing this module needs no
agent runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired

from fund.agents.skills import skill_sources
from fund.config import FundConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    # ``fund.agents.toolsets`` reaches ``fund.tools`` (the agent stack) at import;
    # keep it type-only here; the builders import ``bind_toolset`` lazily.
    from fund.agents.toolsets import RunContext


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
