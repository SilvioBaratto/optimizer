"""Per-run toolset binding: the load-bearing tool adapter.

Each Phase-3 tool (``fund.tools.*``) is a frozen ``@tool_envelope`` function whose
**first positional arg is a ``Session``**. A deep agent must never see that
session — nor pick the decision date, the owning portfolio, or the risk profile.
So for one run we bind those from a :class:`RunContext` and expose to the model
**only** the structured inputs it is allowed to choose (a universe, a weight
vector, a set of series names). This mirrors the profiler's ``_make_save_profile``:
a langchain ``@tool`` closure captures the run context; its args schema carries no
``Session``.

The bound closures also carry the load-bearing invariant into the audit trail:

* the resolved :class:`~fund.schemas.ConstraintSet` is translated into the tools'
  arg dicts: ``optimize_portfolio`` receives the full mapped ``MeanRiskConfig``
  (objective, risk measure, risk aversion, tail beta, regularisation, cardinality)
  with ``max_weights`` tightened to the K&E ``max_position_cap``, and
  ``risk_check`` receives the same effective bounds. The decision logs the config
  that was actually applied, and the client's mapped intent stays auditable in the
  decision's ``constraint_set``;
* the mandate's ESG sector exclusions and low-K&E instrument restrictions are
  screened in code (:func:`fund.tools.mandate.mandate_breaches`): ``universe_filter``
  drops offending names, ``optimize_portfolio`` refuses a universe holding one,
  ``risk_check`` fails a held one, and ``place_orders`` re-checks the audited
  weights against the *current* mandate before a ticket exists. Screening fails
  closed on an instrument it cannot classify. ``backtest`` compares its drawdown
  to the client's ``nu1/nu2/nu3`` ceilings and reports the breached tiers;
* ``optimize_portfolio`` appends an ``agent_decision`` (allocator: the constraint
  set + the *effective* optimizer config actually applied + the optimizer's
  weights); the PM-level ``place_orders`` appends the executor's execution decision
  + HITL outcome.

The whole closure body degrades through :func:`~fund.tools._base.tool_envelope`,
so a mis-resolved profile or a failed audit write returns ``{ok: false, error}``
instead of raising across the tool boundary.

Every body also runs alone (a per-run lock) and inside a SAVEPOINT. LangGraph
dispatches the parallel tool calls of one model turn on worker threads, but a run
shares one non-thread-safe ``Session``; a failed call rolls back only its own
savepoint, so it cannot poison the transaction every later tool and the run's own
rows live in.

The langchain import (``langchain_core.tools.tool``) lives **inside**
:func:`bind_toolset`'s builders, so ``import fund.agents.toolsets`` pulls in no
agent-stack runtime (keeping the bare package import agent-stack-free).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec

from fund import tools as _tools
from fund.tools._base import ToolResult, err, ok, tool_envelope
from fund.tools.mandate import HELD_TOL, constraint_set_breaches, held_tickers

if TYPE_CHECKING:
    import datetime as dt
    import uuid

    from langchain_core.tools import BaseTool
    from langgraph.store.base import BaseStore
    from sqlalchemy.orm import Session

    from fund.config import FundConfig
    from fund.schemas import ConstraintSet

# Canonical role → tool-name map. Keys are the ``skills.SKILLS_BY_AGENT`` role
# keys verbatim; the values **partition** the real ``fund.tools`` exports
# (a bijection — every tool assigned to exactly one role, none twice). The
# economist reads the market/macro panels and the regime statistics (D-regime), the
# allocator owns the critical path, the risk agent owns the blocking gate, and the
# PM commits orders (``place_orders`` is a top-level PM tool behind the HITL gate —
# the executor *proposes*, the PM *commits*). The profiler's ``save_profile`` is
# built separately (not one of these), so ``profiler``/``executor`` bind no tool
# from this set.
TOOLS_BY_AGENT: dict[str, tuple[str, ...]] = {
    "orchestrator": ("place_orders", "get_portfolio_history"),
    "profiler": (),
    "economist": (
        "get_macro_series",
        "get_prices",
        "get_recent_events",
        "get_regime_stats",
    ),
    "allocator": ("universe_filter", "estimate_moments", "optimize_portfolio"),
    "risk": ("risk_check", "backtest"),
    "executor": (),
}


@dataclass(frozen=True)
class RunContext:
    """Immutable per-run binding context for the toolset closures.

    Carries everything a bound tool needs but the model must not choose: the
    injected sync ``session``, the ``asof`` decision bar that bounds every
    price read (no look-ahead), the LangGraph ``store`` the active
    ``ConstraintSet`` is resolved from, the run ``config`` (store key, etc.), the
    audit ``run_id``, and the owning ``portfolio_id``. ``base_currency`` is the
    mandate's reporting currency: the allocator tools FX-convert prices to it before
    deriving returns, so they optimise over the series the NAV tracker marks
    (``None`` keeps listing currency — only sound for a single-currency universe).
    """

    session: Session
    asof: dt.date
    store: BaseStore | None
    config: FundConfig
    run_id: uuid.UUID
    portfolio_id: uuid.UUID
    # The moments/backtest lookback — 3y rolling (~756 trading days), a
    # run-config value (never a ``ConstraintSet`` field). Wired into ``backtest``
    # (which accepts a ``window``) and ``get_regime_stats``;
    # ``estimate_moments`` / ``optimize_portfolio`` take no start bound in the frozen
    # Phase-3 backbone, so it is recorded on the run but inert there. Additive
    # default keeps existing callers unchanged.
    lookback_days: int = 756
    base_currency: str | None = None
    # Held for the whole body of every bound tool: the parallel tool calls of one
    # model turn run on worker threads, but ``session`` is not thread-safe.
    lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )


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


def _effective_max_weight(cs: ConstraintSet) -> float:
    """Per-asset weight ceiling: the bound, tightened by the K&E position cap."""
    cap = cs.universe_filters.max_position_cap
    return cs.bounds.max_weights if cap is None else min(cs.bounds.max_weights, cap)


def _bounds_args(cs: ConstraintSet) -> dict[str, Any]:
    """Long-only bounds enforced by ``risk_check`` (position cap included)."""
    return {
        "min_weights": cs.bounds.min_weights,
        "max_weights": _effective_max_weight(cs),
        "budget": cs.bounds.budget,
    }


def _optimizer_constraints(cs: ConstraintSet) -> dict[str, Any]:
    """The mapped ``MeanRiskConfig`` fields ``optimize_portfolio`` consumes."""
    from fund.tools.optimize import _CONSTRAINT_FIELDS

    config = dataclasses.replace(
        cs.to_mean_risk_config(), max_weights=_effective_max_weight(cs)
    )
    return {name: getattr(config, name) for name in _CONSTRAINT_FIELDS}


def _instrument_breaches(
    ctx: RunContext, cs: ConstraintSet, tickers: list[str]
) -> list[dict[str, Any]]:
    """ESG-sector and K&E instrument breaches among ``tickers`` (empty if none)."""
    return constraint_set_breaches(ctx.session, cs, tickers)


def _describe_breaches(breaches: list[dict[str, Any]]) -> str:
    """A one-line, model-readable summary of structured mandate breaches."""
    parts = []
    for breach in breaches:
        detail = breach.get("sector") or breach.get("instrument_type") or ""
        parts.append(
            f"{breach['asset']} ({breach['type']}{': ' + detail if detail else ''})"
        )
    return "; ".join(parts)


def _effective_optimizer_config(constraints: dict[str, Any] | None) -> dict[str, Any]:
    """The ``MeanRiskConfig`` ``optimize_portfolio`` ACTUALLY applied. Reuses the
    tool's own ``_resolve_config`` as the single source of truth, so the audit can
    never drift from what shaped the weights."""
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

P = ParamSpec("P")


def _serialised(
    fn: Callable[Concatenate[RunContext, P], ToolResult],
) -> Callable[Concatenate[RunContext, P], ToolResult]:
    """Run a bound tool body alone on the run's session, inside a SAVEPOINT.

    A raised exception or an ``ok: false`` result rolls back the body's own writes
    and nothing else, so a swallowed failure (the envelope returns it to the model)
    leaves the run's transaction usable for the next tool.
    """

    @functools.wraps(fn)
    def wrapper(ctx: RunContext, *args: P.args, **kwargs: P.kwargs) -> ToolResult:
        with ctx.lock:
            savepoint = ctx.session.begin_nested()
            try:
                result = fn(ctx, *args, **kwargs)
                if result.get("ok") is False:
                    savepoint.rollback()
                else:
                    savepoint.commit()
            except BaseException:
                # A failed flush leaves the savepoint inactive but still current;
                # only an explicit rollback hands the session back usable.
                savepoint.rollback()
                raise
            return result

    return wrapper


@tool_envelope
@_serialised
def _get_prices_impl(ctx: RunContext, tickers: list[str]) -> ToolResult:
    return _tools.get_prices(ctx.session, ctx.asof, tickers)


@tool_envelope
@_serialised
def _get_macro_impl(ctx: RunContext, names: list[str]) -> ToolResult:
    return _tools.get_macro_series(ctx.session, names, ctx.asof)


@tool_envelope
@_serialised
def _universe_filter_impl(ctx: RunContext, universe: list[str]) -> ToolResult:
    cs = _resolve_constraint_set(ctx)
    excluded: dict[str, list[str]] = {}
    if cs is not None:
        for breach in _instrument_breaches(ctx, cs, universe):
            kinds = excluded.setdefault(breach["asset"], [])
            if breach["type"] not in kinds:
                kinds.append(breach["type"])
    allowed = [ticker for ticker in universe if ticker not in excluded]
    result = _tools.universe_filter(
        ctx.session,
        ctx.asof,
        allowed,
        base_currency=ctx.base_currency,
    )
    if excluded and result.get("ok"):
        result["mandate_excluded"] = excluded
    return result


@tool_envelope
@_serialised
def _estimate_moments_impl(ctx: RunContext, universe: list[str]) -> ToolResult:
    # ``ctx.lookback_days`` is recorded on the run but inert here — the frozen
    # ``estimate_moments`` takes no start bound, so moments use full history up to
    # ``asof``.
    return _tools.estimate_moments(
        ctx.session, ctx.asof, universe, base_currency=ctx.base_currency
    )


@tool_envelope
@_serialised
def _optimize_impl(ctx: RunContext, universe: list[str]) -> ToolResult:
    cs = _resolve_constraint_set(ctx)
    if cs is not None:
        breaches = _instrument_breaches(ctx, cs, universe)
        if breaches:
            return err(
                "universe breaches the client's mandate: "
                f"{_describe_breaches(breaches)}; remove these names and retry"
            )
    constraints = _optimizer_constraints(cs) if cs is not None else None
    result = _tools.optimize_portfolio(
        ctx.session,
        ctx.asof,
        universe,
        constraints=constraints,
        base_currency=ctx.base_currency,
    )
    if result.get("ok"):
        _append_allocator_decision(
            ctx,
            cs,
            _effective_optimizer_config(constraints),
            result["data"]["weights"],
        )
    return result


@tool_envelope
@_serialised
def _risk_check_impl(ctx: RunContext, weights: dict[str, float]) -> ToolResult:
    cs = _resolve_constraint_set(ctx)
    constraints = _bounds_args(cs) if cs is not None else None
    result = _tools.risk_check(weights, constraints=constraints)
    if cs is None or not result.get("ok"):
        return result
    breaches = _instrument_breaches(ctx, cs, held_tickers(weights))
    if not breaches:
        return result
    return ok(
        {"passed": False, "violations": [*result["data"]["violations"], *breaches]}
    )


def check_weights_against_mandate(
    ctx: RunContext, weights: dict[str, float]
) -> ToolResult:
    """The live ``risk_check`` verdict on ``weights``, for callers outside the agent.

    The adviser ``edit`` action validates typed weights with this so a human override
    is held to the same bounds, position cap and ESG / K&E screen the risk agent's
    tool applies, not to the bare long-only defaults.
    """
    return _risk_check_impl(ctx, weights)


@tool_envelope
@_serialised
def _backtest_impl(ctx: RunContext, weights: dict[str, float]) -> ToolResult:
    # Bound the walk-forward training block to the run's 3y rolling lookback
    # (the one frozen tool that accepts a window). A panel shorter than the window
    # falls back to the full sample inside ``backtest`` (no leakage either way).
    result = _tools.backtest(
        ctx.session,
        ctx.asof,
        weights,
        window={"train_size": ctx.lookback_days},
        base_currency=ctx.base_currency,
    )
    cs = _resolve_constraint_set(ctx)
    if cs is None or not result.get("ok"):
        return result
    # The drawdown ceilings are compared to the optimizer's number here, in code,
    # so the risk agent reads a verdict instead of recalling the client's limits.
    # A non-finite drawdown compares False against every ceiling, i.e. breaches all.
    max_drawdown = result["data"]["metrics"]["max_drawdown"]
    ceilings = {"nu1": cs.nu1, "nu2": cs.nu2, "nu3": cs.nu3}
    result["data"]["drawdown_tiers"] = {
        **ceilings,
        "max_drawdown": max_drawdown,
        "breached": [
            tier for tier, ceiling in ceilings.items() if not max_drawdown <= ceiling
        ],
    }
    return result


@tool_envelope
@_serialised
def _place_orders_impl(ctx: RunContext, weights: dict[str, float]) -> ToolResult:
    # Load-bearing invariant: skfolio computes the weights, never the LLM. The
    # ``weights`` arg is the model's *proposal* only — the paper ticket is placed
    # from the allocator's audited ``optimize_portfolio`` output, read back from the
    # run's audit trail (the single source of weights). No audited proposal ⇒ empty
    # weights ⇒ the frozen tool returns ``err("no weights to place")`` (safe
    # no-ticket outcome).
    from fund.audit import AgentRunRepository

    audited = AgentRunRepository(ctx.session).latest_optimizer_weights(ctx.run_id)
    cs = _resolve_constraint_set(ctx)
    if cs is not None and audited:
        # The mandate can change between the audited proposal and the order (a
        # re-profile), so the weights are re-checked against the current one.
        cap = _effective_max_weight(cs)
        violations = _instrument_breaches(ctx, cs, held_tickers(audited)) + [
            {"type": "max_weight", "asset": ticker, "value": weight, "limit": cap}
            for ticker, weight in audited.items()
            if weight > cap + HELD_TOL
        ]
        if violations:
            return err(
                "audited weights breach the client's mandate: "
                f"{_describe_breaches(violations)}; send the proposal back to the "
                "allocator"
            )
    result = _tools.place_orders(ctx.session, ctx.asof, audited, ctx.portfolio_id)
    data = result.get("data") if result.get("ok") else None
    # Log once, on the real placement; the idempotent HITL re-run does not
    # double-log (mirrors the profiler's save_profile).
    if data is not None and not data.get("idempotent", False):
        _append_executor_decision(ctx, data)
    return result


@tool_envelope
@_serialised
def _get_regime_stats_impl(ctx: RunContext, tickers: list[str]) -> ToolResult:
    return _tools.get_regime_stats(
        ctx.session, ctx.asof, tickers, window=ctx.lookback_days
    )


@tool_envelope
@_serialised
def _get_recent_events_impl(ctx: RunContext, window: int = 14) -> ToolResult:
    return _tools.get_recent_events(ctx.session, ctx.asof, window=window)


@tool_envelope
@_serialised
def _get_portfolio_history_impl(ctx: RunContext, lookback: int = 90) -> ToolResult:
    # ``portfolio_id`` is bound from the run context, never chosen by the model —
    # an agent reads only its own portfolio's history.
    return _tools.get_portfolio_history(
        ctx.session, ctx.portfolio_id, ctx.asof, lookback=lookback
    )


# ---------------------------------------------------------------------------
# Builders — wrap each bound impl in a langchain ``@tool`` (agent-stack import is
# lazy, keeping ``import fund.agents.toolsets`` agent-stack-free).
# ---------------------------------------------------------------------------


def _bind_get_prices(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def get_prices(tickers: list[str]) -> ToolResult:
        """Daily-close statistics for ``tickers`` as of the run date (no
        look-ahead). Returns the ``{ok, data}`` envelope: per ticker the last close,
        its 1m/3m/12m change, trailing-year realised volatility and drawdown (current
        and maximum, non-negative fractions below the peak: 0.2 means 20% down), plus
        a coverage summary and any ``missing`` tickers — never the raw price matrix."""
        return _get_prices_impl(ctx, tickers)

    return get_prices


def _bind_get_macro(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def get_macro_series(names: list[str]) -> ToolResult:
        """Macro-series statistics for the FRED ``names`` as of the run date.
        Returns the ``{ok, data}`` envelope: per series the last level, its
        1m/3m/12m change and z-score against its history, its ``latest`` observation
        and a short ``recent`` tail of dated values, plus a coverage summary.
        Observations appear only after an estimated publication lag, so the newest
        monthly/quarterly prints may be absent and the NBER recession flags
        (``USREC``/``USRECDM``) are never shown. Absent series are flagged in
        ``missing``, never raised."""
        return _get_macro_impl(ctx, names)

    return get_macro_series


def _bind_universe_filter(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def universe_filter(universe: list[str]) -> ToolResult:
        """Prune ``universe`` with the pre-selection stack, dropping any name the
        client's mandate forbids (ESG-excluded sector, complex or leveraged product
        for a restricted client, or an unclassifiable instrument). Returns the
        ``{ok, data}`` envelope with the surviving tickers, in requested order, and
        ``mandate_excluded`` naming each dropped ticker and why."""
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
        weight. Fails, naming the offenders, if ``universe`` holds a name the
        client's mandate forbids or is too small for the position cap. Returns the
        ``{ok, data}`` envelope with ``weights`` and in-sample ``metrics``."""
        return _optimize_impl(ctx, universe)

    return optimize_portfolio


def _bind_risk_check(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def risk_check(weights: dict[str, float]) -> ToolResult:
        """Validate ``weights`` against the run's risk profile: weight bounds and
        position cap, ESG sector exclusions and low-K&E instrument restrictions on
        every held name. Returns the ``{ok, data}`` envelope with ``passed`` and
        structured ``violations``."""
        return _risk_check_impl(ctx, weights)

    return risk_check


def _bind_backtest(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def backtest(weights: dict[str, float]) -> ToolResult:
        """Backtest fixed ``weights`` out-of-sample over prices up to the run date
        (walk-forward, no leakage). Returns the ``{ok, data}`` envelope with
        ``metrics`` and ``drawdown_tiers``: the client's ``nu1`` (soft warning),
        ``nu2`` (hard ceiling) and ``nu3`` (absolute stop) drawdown ceilings, the
        backtest ``max_drawdown``, and the ``breached`` tiers."""
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


def _bind_get_regime_stats(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def get_regime_stats(tickers: list[str]) -> ToolResult:
        """Regime statistics for the joint behaviour of ``tickers`` (use broad
        market proxies) up to the run date: financial turbulence with its window
        percentile, the absorption ratio and its change over a quarter, and the
        two-state stress probability. The optimizer side computes them — you only
        interpret. Returns the ``{ok, data}`` envelope; needs two priced assets and
        ~60 trading days."""
        return _get_regime_stats_impl(ctx, tickers)

    return get_regime_stats


def _bind_get_recent_events(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def get_recent_events(window: int = 14) -> ToolResult:
        """Recent global market/macro/news digests in the ``window`` days up to the
        run date (no look-ahead). Returns the ``{ok, data}`` envelope with one
        compact entry per day (narrative + top themes) — never a raw matrix."""
        return _get_recent_events_impl(ctx, window)

    return get_recent_events


def _bind_get_portfolio_history(ctx: RunContext) -> BaseTool:
    from langchain_core.tools import tool

    @tool
    def get_portfolio_history(lookback: int = 90) -> ToolResult:
        """This portfolio's own recent activity in the ``lookback`` days up to the
        run date: prior rebalances, runs, orders, and current holdings. Returns the
        ``{ok, data}`` envelope with compact summaries — never a raw matrix."""
        return _get_portfolio_history_impl(ctx, lookback)

    return get_portfolio_history


_BUILDERS: dict[str, Any] = {
    "get_prices": _bind_get_prices,
    "get_macro_series": _bind_get_macro,
    "get_recent_events": _bind_get_recent_events,
    "get_regime_stats": _bind_get_regime_stats,
    "universe_filter": _bind_universe_filter,
    "estimate_moments": _bind_estimate_moments,
    "optimize_portfolio": _bind_optimize,
    "risk_check": _bind_risk_check,
    "backtest": _bind_backtest,
    "place_orders": _bind_place_orders,
    "get_portfolio_history": _bind_get_portfolio_history,
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
