"""T7.3 — per-run toolset binding: the load-bearing tool adapter.

Each Phase-3 tool (``fund.tools.*``) is a frozen ``@tool_envelope`` function whose
**first positional arg is a ``Session``**. A deep agent must never see that
session — nor pick the decision date, the owning portfolio, or the risk profile.
So for one run we bind those from a :class:`RunContext` and expose to the model
**only** the structured inputs it is allowed to choose (a universe, a weight
vector, a set of series names). This mirrors the profiler's ``_make_save_profile``:
a langchain ``@tool`` closure captures the run context; its args schema carries no
``Session``.

The bound closures also carry the load-bearing invariant into the audit trail:

* the resolved :class:`~fund.schemas.ConstraintSet` is translated **only into the
  tools' existing arg dicts** (``optimize_portfolio`` honours the ``bounds``
  subset; ESG exclusions → ``universe_filter`` criteria; bounds → ``risk_check``
  constraints). The Phase-3 backbone stays frozen — no new tool business logic
  (Risk R2: structural ``nu``/ESG enforcement is ask-first, out of Phase-7 exit,
  so a criterion the frozen tool does not model is passed but ignored). The full
  MiFID mapping (objective/risk-measure/risk-aversion/beta/l1/l2/cardinality) is
  inert on the allocation today, so it is **not** logged as applied — the client's
  mapped intent stays auditable in the decision's ``constraint_set``;
* ``optimize_portfolio`` appends an ``agent_decision`` (allocator: the constraint
  set + the *effective* optimizer config actually applied + the optimizer's
  weights); the PM-level ``place_orders`` appends the executor's execution decision
  + HITL outcome.

The whole closure body degrades through :func:`~fund.tools._base.tool_envelope`,
so a mis-resolved profile or a failed audit write returns ``{ok: false, error}``
instead of raising across the tool boundary.

The langchain import (``langchain_core.tools.tool``) lives **inside**
:func:`bind_toolset`'s builders, so ``import fund.agents.toolsets`` pulls in no
agent-stack runtime (Task 7 keeps the bare package import agent-stack-free).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fund import tools as _tools
from fund.tools._base import ToolResult, tool_envelope

if TYPE_CHECKING:
    import datetime as dt
    import uuid

    from langchain_core.tools import BaseTool
    from langgraph.store.base import BaseStore
    from sqlalchemy.orm import Session

    from fund.config import FundConfig
    from fund.schemas import ConstraintSet

# Canonical role → tool-name map. Keys are the ``skills.SKILLS_BY_AGENT`` role
# keys verbatim; the values **partition** the eight real ``fund.tools`` exports
# (a bijection — every tool assigned to exactly one role, none twice). The
# economist reads the market/macro panels narratively (D-regime), the allocator
# owns the critical path, the risk agent owns the blocking gate, and the PM
# commits orders (``place_orders`` is a top-level PM tool behind the HITL gate —
# the executor *proposes*, the PM *commits*). The profiler's ``save_profile`` is
# built separately (not one of these eight), so ``profiler``/``executor`` bind no
# tool from this set.
TOOLS_BY_AGENT: dict[str, tuple[str, ...]] = {
    "orchestrator": ("place_orders",),
    "profiler": (),
    "economist": ("get_macro_series", "get_prices"),
    "allocator": ("universe_filter", "estimate_moments", "optimize_portfolio"),
    "risk": ("risk_check", "backtest"),
    "executor": (),
}


@dataclass(frozen=True)
class RunContext:
    """Immutable per-run binding context for the toolset closures.

    Carries everything a bound tool needs but the model must not choose: the
    injected sync ``session`` (D1), the ``asof`` decision bar that bounds every
    price read (no look-ahead), the LangGraph ``store`` the active
    ``ConstraintSet`` is resolved from, the run ``config`` (store key, etc.), the
    audit ``run_id``, and the owning ``portfolio_id``.
    """

    session: Session
    asof: dt.date
    store: BaseStore | None
    config: FundConfig
    run_id: uuid.UUID
    portfolio_id: uuid.UUID
    # D18: the moments/backtest lookback — 3y rolling (~756 trading days), a
    # run-config value (never a ``ConstraintSet`` field). Wired into ``backtest``
    # (which accepts a ``window``); ``estimate_moments`` / ``optimize_portfolio``
    # take no start bound in the frozen Phase-3 backbone, so it is recorded on the
    # run but inert there (the ESG-exclusions precedent — R2, structural
    # enforcement is ask-first). Additive default keeps existing callers unchanged.
    lookback_days: int = 756


# ---------------------------------------------------------------------------
# ConstraintSet resolution + translation into the frozen tools' arg dicts.
# ---------------------------------------------------------------------------


def _resolve_constraint_set(ctx: RunContext) -> ConstraintSet | None:
    """Resolve the run's active ``ConstraintSet`` from the store (``None`` if no
    store, or nothing cached for the portfolio)."""
    if ctx.store is None:
        return None
    from fund.audit import resolve_constraint_set
    from fund.schemas import ConstraintSetRef

    ref = ConstraintSetRef(
        portfolio_id=str(ctx.portfolio_id),
        store_key=ctx.config.constraint_set_store_key,
    )
    return resolve_constraint_set(ctx.store, ref)


def _bounds_args(cs: ConstraintSet) -> dict[str, Any]:
    """Long-only bounds honoured by ``optimize_portfolio`` / ``risk_check``."""
    return {
        "min_weights": cs.bounds.min_weights,
        "max_weights": cs.bounds.max_weights,
        "budget": cs.bounds.budget,
    }


def _universe_criteria(cs: ConstraintSet) -> dict[str, Any] | None:
    """ESG exclusions as ``universe_filter`` criteria (``None`` when empty).

    The frozen ``universe_filter`` only honours ``PreSelectionConfig`` fields, so
    ``exclusions`` is passed but ignored today (R2 — structural ESG enforcement is
    ask-first). Translating it here keeps the intent auditable and wires the arg
    for the day the tool models it, without touching the frozen backbone.
    """
    exclusions = [sector.value for sector in cs.esg.exclusions]
    if not exclusions:
        return None
    return {"exclusions": exclusions}


def _effective_optimizer_config(constraints: dict[str, Any] | None) -> dict[str, Any]:
    """The ``MeanRiskConfig`` ``optimize_portfolio`` ACTUALLY applied: min-variance
    defaults (D17) + the honoured min/max/budget bounds. The mapped MiFID knobs
    (objective/risk_measure/risk_aversion/beta/l1/l2/cardinality) are NOT consumed
    by the frozen Phase-3 tool (R2, ask-first), so they are intentionally absent
    here — the client's full mapped intent stays in the decision's ``constraint_set``.
    Logging only the applied config keeps the audit honest (reuses the tool's own
    ``_resolve_config`` as the single source of truth)."""
    from fund.tools.optimize import _resolve_config

    return dataclasses.asdict(_resolve_config(constraints))


# ---------------------------------------------------------------------------
# Audit — load-bearing facts logged from inside the bound closures.
# ---------------------------------------------------------------------------


def _append_allocator_decision(
    ctx: RunContext,
    cs: ConstraintSet | None,
    optimizer_config: dict[str, Any],
    weights: dict[str, float],
) -> None:
    """Append the allocator's optimize decision: constraint set + the effective
    (applied) optimizer config + the optimizer's weights (never model-emitted)."""
    from fund.audit import AgentRunRepository

    payload = json.dumps(
        {"optimizer_config": optimizer_config, "weights": weights},
        sort_keys=True,
        default=str,
    )
    AgentRunRepository(ctx.session).append_decision(
        ctx.run_id,
        agent="allocator",
        step="optimize_portfolio",
        constraint_set=cs.model_dump(mode="json") if cs is not None else None,
        llm_response=payload,
        llm_response_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def _append_executor_decision(ctx: RunContext, ticket: dict[str, Any]) -> None:
    """Append the executor's execution decision + HITL outcome (the order ran, so
    the adviser approved)."""
    from fund.audit import AgentRunRepository

    payload = json.dumps(
        {"order_id": ticket.get("order_id"), "weights": ticket.get("weights")},
        sort_keys=True,
        default=str,
    )
    AgentRunRepository(ctx.session).append_decision(
        ctx.run_id,
        agent="executor",
        step="place_orders",
        llm_response=payload,
        llm_response_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        hitl_decision={"decision": "approve"},
    )


# ---------------------------------------------------------------------------
# Bound implementations — session/asof/portfolio bound, model-facing arg free.
# Each is wrapped by ``tool_envelope`` so the closure never raises.
# ---------------------------------------------------------------------------


@tool_envelope
def _get_prices_impl(ctx: RunContext, tickers: list[str]) -> ToolResult:
    return _tools.get_prices(ctx.session, ctx.asof, tickers)


@tool_envelope
def _get_macro_impl(ctx: RunContext, names: list[str]) -> ToolResult:
    return _tools.get_macro_series(ctx.session, names, ctx.asof)


@tool_envelope
def _universe_filter_impl(ctx: RunContext, universe: list[str]) -> ToolResult:
    cs = _resolve_constraint_set(ctx)
    criteria = _universe_criteria(cs) if cs is not None else None
    return _tools.universe_filter(ctx.session, ctx.asof, universe, criteria=criteria)


@tool_envelope
def _estimate_moments_impl(ctx: RunContext, universe: list[str]) -> ToolResult:
    # D18: ``ctx.lookback_days`` is recorded on the run but inert here — the frozen
    # ``estimate_moments`` takes no start bound, so moments use full history up to
    # ``asof`` (the ESG-exclusion precedent; structural enforcement is ask-first).
    return _tools.estimate_moments(ctx.session, ctx.asof, universe)


@tool_envelope
def _optimize_impl(ctx: RunContext, universe: list[str]) -> ToolResult:
    cs = _resolve_constraint_set(ctx)
    constraints = _bounds_args(cs) if cs is not None else None
    result = _tools.optimize_portfolio(
        ctx.session, ctx.asof, universe, constraints=constraints
    )
    if result.get("ok"):
        # Log the config the frozen tool ACTUALLY applied (min-variance defaults +
        # honoured bounds), not the full mapped MiFID intent — that intent stays in
        # the decision's ``constraint_set``. Keeps the audit honest.
        _append_allocator_decision(
            ctx,
            cs,
            _effective_optimizer_config(constraints),
            result["data"]["weights"],
        )
    return result


@tool_envelope
def _risk_check_impl(ctx: RunContext, weights: dict[str, float]) -> ToolResult:
    cs = _resolve_constraint_set(ctx)
    constraints = _bounds_args(cs) if cs is not None else None
    return _tools.risk_check(weights, constraints=constraints)


@tool_envelope
def _backtest_impl(ctx: RunContext, weights: dict[str, float]) -> ToolResult:
    # D18: bound the walk-forward training block to the run's 3y rolling lookback
    # (the one frozen tool that accepts a window). A panel shorter than the window
    # falls back to the full sample inside ``backtest`` (no leakage either way).
    return _tools.backtest(
        ctx.session, ctx.asof, weights, window={"train_size": ctx.lookback_days}
    )


@tool_envelope
def _place_orders_impl(ctx: RunContext, weights: dict[str, float]) -> ToolResult:
    # Load-bearing invariant: skfolio computes the weights, never the LLM. The
    # ``weights`` arg is the model's *proposal* only — the paper ticket is placed
    # from the allocator's audited ``optimize_portfolio`` output, read back from the
    # run's audit trail (the single source of weights). No audited proposal ⇒ empty
    # weights ⇒ the frozen tool returns ``err("no weights to place")`` (safe
    # no-ticket outcome).
    from fund.audit import AgentRunRepository

    audited = AgentRunRepository(ctx.session).latest_optimizer_weights(ctx.run_id)
    result = _tools.place_orders(ctx.session, ctx.asof, audited, ctx.portfolio_id)
    data = result.get("data") if result.get("ok") else None
    # Log once, on the real placement; the idempotent HITL re-run (D3) does not
    # double-log (mirrors the profiler's save_profile).
    if data is not None and not data.get("idempotent", False):
        _append_executor_decision(ctx, data)
    return result


# ---------------------------------------------------------------------------
# Builders — wrap each bound impl in a langchain ``@tool`` (agent-stack import is
# lazy, keeping ``import fund.agents.toolsets`` agent-stack-free).
# ---------------------------------------------------------------------------


def _bind_get_prices(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def get_prices(tickers: list[str]) -> ToolResult:
        """Daily-close coverage summary for ``tickers`` as of the run date (no
        look-ahead). Returns the ``{ok, data}`` envelope with a shape/coverage
        summary — never the raw price matrix."""
        return _get_prices_impl(ctx, tickers)

    return get_prices


def _bind_get_macro(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def get_macro_series(names: list[str]) -> ToolResult:
        """Macro-series coverage summary for the FRED ``names`` as of the run date.
        Returns the ``{ok, data}`` envelope; absent series are flagged in
        ``missing``, never raised."""
        return _get_macro_impl(ctx, names)

    return get_macro_series


def _bind_universe_filter(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def universe_filter(universe: list[str]) -> ToolResult:
        """Prune ``universe`` with the pre-selection stack (the run's risk profile
        supplies the criteria). Returns the ``{ok, data}`` envelope with the
        surviving tickers, in requested order."""
        return _universe_filter_impl(ctx, universe)

    return universe_filter


def _bind_estimate_moments(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def estimate_moments(universe: list[str]) -> ToolResult:
        """Estimate ``(mu, cov)`` for ``universe`` as of the run date (Ledoit-Wolf
        covariance + empirical mean). Returns the ``{ok, data}`` envelope."""
        return _estimate_moments_impl(ctx, universe)

    return estimate_moments


def _bind_optimize(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def optimize_portfolio(universe: list[str]) -> ToolResult:
        """Compute long-only portfolio weights for ``universe`` as of the run date.
        skfolio computes the weights from the run's risk profile — you never emit a
        weight. Returns the ``{ok, data}`` envelope with ``weights`` and in-sample
        ``metrics``."""
        return _optimize_impl(ctx, universe)

    return optimize_portfolio


def _bind_risk_check(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def risk_check(weights: dict[str, float]) -> ToolResult:
        """Validate ``weights`` against the run's risk profile bounds. Returns the
        ``{ok, data}`` envelope with ``passed`` and structured ``violations``."""
        return _risk_check_impl(ctx, weights)

    return risk_check


def _bind_backtest(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def backtest(weights: dict[str, float]) -> ToolResult:
        """Backtest fixed ``weights`` out-of-sample over prices up to the run date
        (walk-forward, no leakage). Returns the ``{ok, data}`` envelope with
        ``metrics``."""
        return _backtest_impl(ctx, weights)

    return backtest


def _bind_place_orders(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def place_orders(weights: dict[str, float]) -> ToolResult:
        """Place the idempotent **paper** ticket for the optimizer's ``weights``,
        filled at the next close after the run date. Call once, after adviser
        approval. Returns the ``{ok, data}`` envelope with the order ticket."""
        return _place_orders_impl(ctx, weights)

    return place_orders


_BUILDERS: dict[str, Any] = {
    "get_prices": _bind_get_prices,
    "get_macro_series": _bind_get_macro,
    "universe_filter": _bind_universe_filter,
    "estimate_moments": _bind_estimate_moments,
    "optimize_portfolio": _bind_optimize,
    "risk_check": _bind_risk_check,
    "backtest": _bind_backtest,
    "place_orders": _bind_place_orders,
}


def bind_toolset(role: str, ctx: RunContext) -> list[BaseTool]:
    """Bind ``role``'s tools into per-run langchain ``@tool`` closures.

    Args:
        role: One of the canonical role keys in :data:`TOOLS_BY_AGENT`.
        ctx: The immutable per-run binding context.

    Returns:
        The role's tools as langchain ``BaseTool`` closures whose args schema
        omits ``session`` (and every other bound-context field). An empty list for
        roles that bind no tool from the frozen eight (``profiler``, ``executor``).

    Raises:
        KeyError: If ``role`` is not a known role.
    """
    return [_BUILDERS[name](ctx) for name in TOOLS_BY_AGENT[role]]


__all__ = ["TOOLS_BY_AGENT", "RunContext", "bind_toolset"]
