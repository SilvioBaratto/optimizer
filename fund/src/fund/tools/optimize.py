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
* no priced assets, or too few common return observations (see
  ``MIN_RETURN_OBSERVATIONS``), ⇒ ``{ok: false, error}``;
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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
)
from optimizer.preprocessing import prices_to_returns
from skfolio import Portfolio

from fund.tools._base import ToolResult, err, ok, thin_sample_error, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    import pandas as pd
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
# Interior-point solvers reach the optimum only to ~1e-9, so a dropped asset can come
# back at -1.5e-9 and one pinned at its cap a hair above it. Weights this small are
# solver noise, and ``risk_check`` reads them as bound breaches on the optimizer's
# own output.
_WEIGHT_DUST = 1e-8

# Measures in return-squared units: the scale ``a_gamma`` is already quoted in.
_SECOND_MOMENT_MEASURES: frozenset[RiskMeasureType] = frozenset(
    {RiskMeasureType.VARIANCE, RiskMeasureType.SEMI_VARIANCE}
)


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


def _calibrate_risk_aversion(
    config: MeanRiskConfig, returns: pd.DataFrame
) -> MeanRiskConfig:
    """Put ``risk_aversion`` on the scale of the config's risk measure.

    The MiFID ``a_gamma`` ladder is quoted for variance utility ``mu - a * var``,
    where per-period variance (~1e-4) and expected return are commensurate. A
    first-order measure (CVaR, CDaR, max drawdown, ...) runs ~1e-2, so the same
    coefficient makes the risk term swamp ``mu`` and the utility objective returns
    the minimum-risk portfolio for every client. The coefficient is rescaled so the
    equal-weight portfolio pays the same risk penalty as it would under variance.

    Only ``MAXIMIZE_UTILITY`` reads ``risk_aversion``; other objectives, the
    second-moment measures and a degenerate reference (no measurable risk) keep the
    config unchanged.
    """
    if (
        config.objective is not ObjectiveFunctionType.MAXIMIZE_UTILITY
        or config.risk_measure in _SECOND_MOMENT_MEASURES
    ):
        return config
    n_assets = returns.shape[1]
    reference = Portfolio(
        X=returns,
        weights=np.full(n_assets, 1.0 / n_assets),
        cvar_beta=config.cvar_beta,
        evar_beta=config.evar_beta,
        cdar_beta=config.cdar_beta,
        edar_beta=config.edar_beta,
    )
    variance = float(reference.variance)
    measure = float(getattr(reference, config.risk_measure.value))
    if not (math.isfinite(variance) and math.isfinite(measure)):
        return config
    if variance <= 0.0 or measure <= 0.0:
        return config
    return replace(config, risk_aversion=config.risk_aversion * variance / measure)


def _clean_weights(weights: np.ndarray, config: MeanRiskConfig) -> np.ndarray:
    """Strip solver noise: zero the dust, restore the budget, respect the bounds.

    Scalar bounds are re-applied after the budget rescale because that rescale can
    push an asset pinned at its cap a relative ~1e-9 over it. Non-scalar or absent
    bounds leave the corresponding step out.
    """
    cleaned = np.where(np.abs(weights) < _WEIGHT_DUST, 0.0, weights)
    total = float(cleaned.sum())
    if isinstance(config.budget, (int, float)) and total > 0.0:
        cleaned = cleaned * (config.budget / total)
    if isinstance(config.min_weights, (int, float)):
        cleaned = np.maximum(cleaned, config.min_weights)
    if isinstance(config.max_weights, (int, float)):
        cleaned = np.minimum(cleaned, config.max_weights)
    return cleaned


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
        ``assets``, ``n_observations``, ``missing`` and ``risk_aversion`` (the
        coefficient the solver ran with, after rescaling to the risk measure).
        ``err`` on an empty universe, no priced assets, too little history, or a
        weight cap too tight for the priced assets to reach the budget.
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

    returns = cast("pd.DataFrame", prices_to_returns(frame))
    if (reason := thin_sample_error(frame, returns, action="optimize")) is not None:
        return err(reason)
    config = _calibrate_risk_aversion(config, returns)
    model = build_mean_risk(config)
    model.fit(returns)
    fitted = model.weights_
    if fitted is None:  # set by the fit above; guard narrows the Optional array
        return err("optimizer produced no weights")
    # ``predict`` reads ``weights_``, so cleaning it first keeps the reported metrics
    # consistent with the weights the caller receives.
    model.weights_ = _clean_weights(np.asarray(fitted, dtype=float), config)
    portfolio = model.predict(returns)

    assets = [str(col) for col in returns.columns]
    weights = {a: float(w) for a, w in zip(assets, model.weights_, strict=True)}
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
            "risk_aversion": float(config.risk_aversion),
        }
    )


__all__ = ["optimize_portfolio"]
