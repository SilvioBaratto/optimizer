"""``optimize_portfolio``: skfolio computes the weights, never the LLM.

Given a price panel as of ``asof``, builds a ``MeanRisk`` optimiser
(by default minimum-variance, long-only, fully invested — ``min_weights=0``,
``max_weights=1``, ``budget=1``), fits it on linear returns, and returns
the resulting weights plus a small metrics summary. ``constraints`` carries the
client's mapped objective, risk measure and weight bounds.

The tool is a pure, deterministic function of ``(session, asof, universe,
constraints)``: the min-variance QP is convex, so identical seeded data ⇒
identical weights. Returns are computed **outside** any pipeline via
``prices_to_returns`` (linear, not log).

Contract (via ``tool_envelope``):

* empty ``universe`` ⇒ ``{ok: false, error}``;
* no priced assets, or fewer than two observations, ⇒ ``{ok: false, error}``;
* a ``max_weights`` cap too tight for the number of priced assets to reach the
  budget ⇒ ``{ok: false, error}`` naming the minimum asset count;
* a bad ``constraints`` value is caught by the envelope and returned as
  ``{ok: false}``, never raised.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
)
from optimizer.preprocessing import prices_to_returns

from fund.tools._base import ToolResult, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Each key maps 1:1 onto a ``MeanRiskConfig`` field; anything else is silently ignored.
# These are exactly the fields ``ConstraintSet.to_mean_risk_config`` populates.
_CONSTRAINT_FIELDS: frozenset[str] = frozenset(
    {
        "objective",
        "risk_measure",
        "risk_aversion",
        "min_weights",
        "max_weights",
        "budget",
        "cardinality",
        "cvar_beta",
        "cdar_beta",
        "l1_coef",
        "l2_coef",
        "risk_free_rate",
    }
)

_BUDGET_TOL = 1e-9


def _resolve_config(constraints: dict[str, Any] | None) -> MeanRiskConfig:
    """Build a long-only ``MeanRiskConfig`` (minimum-variance unless overridden).

    Caller-supplied ``constraints`` override the defaults field by field; an
    ``objective`` / ``risk_measure`` given as its string value is coerced to the
    optimizer enum.
    """
    base = MeanRiskConfig()
    if not constraints:
        return base
    overrides = {k: constraints[k] for k in _CONSTRAINT_FIELDS if k in constraints}
    if "objective" in overrides:
        overrides["objective"] = ObjectiveFunctionType(overrides["objective"])
    if "risk_measure" in overrides:
        overrides["risk_measure"] = RiskMeasureType(overrides["risk_measure"])
    return replace(base, **overrides) if overrides else base


def _infeasible_cap_error(config: MeanRiskConfig, n_assets: int) -> str | None:
    """Explain a per-asset cap that cannot sum to the budget, else ``None``."""
    cap, budget = config.max_weights, config.budget
    if cap is None or budget is None or cap <= 0.0:
        return None
    if cap * n_assets >= budget - _BUDGET_TOL:
        return None
    needed = math.ceil(budget / cap - _BUDGET_TOL)
    return (
        f"max_weights {cap:g} cannot reach budget {budget:g} with {n_assets} priced "
        f"assets; widen the universe to at least {needed} assets"
    )


@tool_envelope
def optimize_portfolio(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    constraints: dict[str, Any] | None = None,
    base_currency: str | None = None,
) -> ToolResult:
    """Optimise weights for ``universe`` from prices as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        universe: yfinance tickers to allocate across.
        constraints: Optional overrides for the ``MeanRiskConfig`` fields in
            ``_CONSTRAINT_FIELDS`` (objective, risk measure, risk aversion, weight
            bounds, budget, cardinality, tail betas, regularisation); defaults to
            long-only, fully invested, minimum-variance.
        base_currency: Mandate base currency prices are FX-converted to before
            returns are derived (``None`` keeps listing currency; see
            :func:`fund.tools.prices.load_price_frame`).

    Returns:
        ``ok`` with ``weights`` (``{ticker: float}`` from skfolio), ``metrics``
        (in-sample ``mean`` / ``standard_deviation`` / ``sharpe_ratio``),
        ``assets``, ``n_observations`` and ``missing``. ``err`` on an empty
        universe, no priced assets, too little history, or a weight cap too tight
        for the priced assets to reach the budget.
    """
    if not universe:
        return err("empty universe")

    frame, missing = load_price_frame(
        session, asof, universe, base_currency=base_currency
    )
    if frame.shape[1] == 0:
        return err("no priced assets in universe")
    if frame.shape[0] < 2:
        return err("insufficient price history to optimize")

    config = _resolve_config(constraints)
    infeasible = _infeasible_cap_error(config, frame.shape[1])
    if infeasible is not None:
        return err(infeasible)

    returns = prices_to_returns(frame)
    model = build_mean_risk(config)
    model.fit(returns)
    portfolio = model.predict(returns)

    assets = [str(col) for col in returns.columns]
    fitted = model.weights_
    if fitted is None:  # set by the fit above; guard narrows the Optional array
        return err("optimizer produced no weights")
    weights = {a: float(w) for a, w in zip(assets, fitted, strict=True)}
    metrics = {
        "mean": float(portfolio.mean),
        "standard_deviation": float(portfolio.standard_deviation),
        "sharpe_ratio": float(portfolio.sharpe_ratio),
    }

    return ok(
        {
            "weights": weights,
            "metrics": metrics,
            "assets": assets,
            "n_observations": int(returns.shape[0]),
            "missing": missing,
        }
    )


__all__ = ["optimize_portfolio"]
