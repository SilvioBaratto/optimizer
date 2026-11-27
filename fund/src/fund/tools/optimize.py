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
  ``MIN_RETURN_OBSERVATIONS``, and ``thin_sample_error`` for the share of the median
  history a young listing must reach), ⇒ ``{ok: false, error}``;
* a ``max_weights`` cap too tight for the number of priced assets to reach the
  budget, or a ``min_weights`` floor too high for them to fit in it,
  ⇒ ``{ok: false, error}`` naming the shortfall;
* a bad ``constraints`` value is caught by the envelope and returned as
  ``{ok: false}``, never raised.

``load_returns_panel`` and ``fit_mean_risk`` are shared with
``fund.tools.candidates``, so every candidate is fitted on the panel this tool would
fit, and the mean-risk candidate is this tool's own solve.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
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

from fund.tools._base import (
    ToolResult,
    annualized_metrics,
    dropped_by_returns,
    err,
    ok,
    per_period_rate,
    periods_per_year,
    thin_sample_error,
    tool_envelope,
)
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
        "prior_config",
    }
)

# An annual risk-free rate, which the tool compounds to the panel's own bar frequency
# once the return index (and so the bars per year) is known. It is not a
# ``MeanRiskConfig`` field: ``risk_free_rate`` there is per bar and is overridden by it.
ANNUAL_RISK_FREE_RATE_KEY = "annual_risk_free_rate"

_NO_PRICED_ASSETS = "no priced assets in universe"
_NO_WEIGHTS = "optimizer produced no weights"

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

# Variance is homogeneous of degree 2 in the weights, so its marginal price along the
# direction of scaling the portfolio is twice its level; a first-order measure's is
# its level (see ``_calibrate_risk_aversion``).
_VARIANCE_DEGREE = 2.0


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
    first-order measure ``R`` (CVaR, CDaR, max drawdown, ...) runs ~1e-2, so the
    same coefficient makes the risk term swamp ``mu``.

    The coefficient is chosen so the tail-measure utility settles at the risk level
    the quoted scale implies. At the variance-utility optimum ``w*`` for ``a``,
    scaling exposure by ``t`` changes utility by ``mu - 2 t a var``, so
    ``mu(w*) = 2 a var(w*)``. A utility over a degree-1 measure is indifferent to the
    same scaling when ``mu = a' R(w*)``, hence ``a' = 2 a var(w*) / R(w*)``. The
    anchor ``w*`` is solved under the config's own bounds and regularisation (not
    its cardinality cap, which would make the anchor a MIP). Equating the penalty on
    the equal-weight portfolio is not equivalent: it drops the factor 2 (variance is
    degree 2, ``R`` degree 1) and ignores that a risk-averse target is a low-variance
    portfolio, which leaves the coefficient too small and the optimum too risky.

    Only ``MAXIMIZE_UTILITY`` reads ``risk_aversion``; other objectives, the
    second-moment measures and a degenerate anchor (no measurable risk) keep the
    config unchanged.
    """
    if (
        config.objective is not ObjectiveFunctionType.MAXIMIZE_UTILITY
        or config.risk_measure in _SECOND_MOMENT_MEASURES
    ):
        return config
    anchor_model = build_mean_risk(
        replace(config, risk_measure=RiskMeasureType.VARIANCE, cardinality=None)
    )
    anchor_model.fit(returns)
    if anchor_model.weights_ is None:  # set by the fit above; narrows the Optional
        return config
    anchor = Portfolio(
        X=returns,
        weights=np.asarray(anchor_model.weights_, dtype=float),
        cvar_beta=config.cvar_beta,
        evar_beta=config.evar_beta,
        cdar_beta=config.cdar_beta,
        edar_beta=config.edar_beta,
    )
    variance = float(anchor.variance)
    measure = float(getattr(anchor, config.risk_measure.value))
    if not (math.isfinite(variance) and math.isfinite(measure)):
        return config
    if variance <= 0.0 or measure <= 0.0:
        return config
    return replace(
        config,
        risk_aversion=_VARIANCE_DEGREE * config.risk_aversion * variance / measure,
    )


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


def _infeasible_bounds_error(config: MeanRiskConfig, n_assets: int) -> str | None:
    """Explain scalar weight bounds the priced assets cannot satisfy, else ``None``.

    A per-asset cap below ``budget / n_assets`` or a floor above it leaves the QP
    with no feasible point, which the solver reports only as an opaque failure the
    allocator cannot act on.
    """
    cap, floor, budget = config.max_weights, config.min_weights, config.budget
    if budget is None:
        return None
    if cap is not None and cap > 0.0 and cap * n_assets < budget - _BUDGET_TOL:
        needed = math.ceil(budget / cap - _BUDGET_TOL)
        return (
            f"max_weights {cap:g} cannot reach budget {budget:g} with {n_assets} "
            f"priced assets; widen the universe to at least {needed} assets"
        )
    if isinstance(floor, int | float) and floor * n_assets > budget + _BUDGET_TOL:
        return (
            f"min_weights {floor:g} floor needs {floor * n_assets:g} of budget "
            f"{budget:g} across {n_assets} priced assets; narrow the universe or "
            "lower the floor"
        )
    return None


@dataclass(frozen=True)
class ReturnsPanel:
    """The returns panel and shared configuration every fit of one tool call uses.

    Attributes:
        returns: Linear returns, one column per priced ticker, on the common sample.
        missing: Requested tickers with no usable prices, including a column the
            returns step had to drop.
        periods: Bars a year of the panel holds.
        config: The long-only ``MeanRiskConfig`` resolved from the caller's
            constraints, its risk-free rate already per bar of this panel.
    """

    returns: pd.DataFrame
    missing: list[str]
    periods: int
    config: MeanRiskConfig


def _with_annual_risk_free_rate(
    config: MeanRiskConfig, constraints: dict[str, Any] | None, periods: int
) -> MeanRiskConfig:
    """Replace the per-bar risk-free rate by the compounded annual one, if given."""
    annual = (constraints or {}).get(ANNUAL_RISK_FREE_RATE_KEY)
    if annual is None:
        return config
    return replace(config, risk_free_rate=per_period_rate(float(annual), periods))


def load_returns_panel(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    constraints: dict[str, Any] | None,
    base_currency: str | None,
    action: str,
) -> ReturnsPanel | str:
    """Load the returns panel for ``universe`` as of ``asof``, or say why there is none.

    Args:
        session: A sync ``portopt_db`` session; the loader does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        universe: yfinance tickers to load.
        constraints: Overrides for the ``MeanRiskConfig`` fields in
            ``_CONSTRAINT_FIELDS`` plus ``annual_risk_free_rate``.
        base_currency: Currency prices are FX-converted to before returns are
            derived (``None`` keeps listing currency).
        action: Verb phrase naming the caller's work in an error, e.g. ``"optimize"``.

    Returns:
        The panel, or the error message for an empty universe, no priced assets, too
        little common history, or a weight cap or floor the priced assets cannot meet.
    """
    if not universe:
        return "empty universe"
    frame, missing = load_price_frame(
        session, asof, universe, base_currency=base_currency
    )
    if frame.shape[1] == 0:
        return _NO_PRICED_ASSETS
    if frame.shape[0] < 2:
        return f"insufficient price history to {action}"
    config = _resolve_config(constraints)
    returns = cast("pd.DataFrame", prices_to_returns(frame))
    # A column with no usable price vanishes from ``returns``; counting it as priced
    # would loosen the bounds check and leave it in neither ``assets`` nor ``missing``.
    missing = [*missing, *dropped_by_returns(frame, returns)]
    if returns.shape[1] == 0:
        return _NO_PRICED_ASSETS
    if (infeasible := _infeasible_bounds_error(config, returns.shape[1])) is not None:
        return infeasible
    if (reason := thin_sample_error(frame, returns, action=action)) is not None:
        return reason
    periods = periods_per_year(returns.index)
    config = _with_annual_risk_free_rate(config, constraints, periods)
    return ReturnsPanel(
        returns=returns, missing=missing, periods=periods, config=config
    )


def fit_mean_risk(returns: pd.DataFrame, config: MeanRiskConfig) -> np.ndarray:
    """Fit a ``MeanRisk`` model on ``returns`` and return its weights as solved.

    The one mean-risk solve behind ``optimize_portfolio`` and the mean-risk
    candidates of ``fund.tools.candidates``. The weights still carry solver noise;
    ``_clean_weights`` strips it.

    Args:
        returns: Linear returns, one column per asset.
        config: The configuration the model is built from.

    Returns:
        One weight per column of ``returns``, in column order.

    Raises:
        ValueError: The fit left no weights, or ``config`` pairs a cardinality with
            a solver that cannot enforce it.
    """
    model = build_mean_risk(config)
    model.fit(returns)
    if model.weights_ is None:  # set by the fit above; narrows the Optional array
        raise ValueError(_NO_WEIGHTS)
    return np.asarray(model.weights_, dtype=float)


def _in_sample_portfolio(
    returns: pd.DataFrame, weights: np.ndarray, config: MeanRiskConfig
) -> Portfolio:
    """The portfolio ``MeanRisk.predict`` would build for ``weights`` on ``returns``.

    Built from the cleaned weights, so the metrics describe exactly what the caller
    receives, with the costs and the risk-free rate the solver ran with.
    """
    return Portfolio(
        X=returns,
        weights=weights,
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        risk_free_rate=config.risk_free_rate,
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

    Under ``MAXIMIZE_UTILITY`` over a first-order risk measure the supplied
    ``risk_aversion`` (quoted on the variance-utility scale) is rescaled so the
    optimum carries the risk level that scale implies.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        universe: yfinance tickers to allocate across.
        constraints: Optional overrides for the ``MeanRiskConfig`` fields in
            ``_CONSTRAINT_FIELDS`` (objective, risk measure, risk aversion, weight
            bounds, budget, cardinality, tail betas, regularisation, the moment
            estimators of the prior); defaults to long-only, fully invested,
            minimum-variance. ``annual_risk_free_rate`` is compounded to the panel's
            own bar frequency and replaces ``risk_free_rate``.
        base_currency: Mandate base currency prices are FX-converted to before
            returns are derived (``None`` keeps listing currency; see
            :func:`fund.tools.prices.load_price_frame`).

    Returns:
        ``ok`` with ``weights`` (``{ticker: float}`` from skfolio), ``metrics``
        (in-sample, annualised: ``annualized_mean`` /
        ``annualized_standard_deviation`` / ``annualized_sharpe_ratio``),
        ``assets``, ``n_observations``, ``missing`` (tickers with no usable prices,
        including a column the returns step had to drop), ``risk_aversion`` (the
        coefficient the solver ran with, after rescaling to the risk measure),
        ``periods_per_year`` (the bars a year of the panel holds, which annualises the
        metrics) and ``risk_free_rate`` (the per-bar rate the solver ran with).
        ``err`` on an empty universe, no priced assets, too little common history
        (including a listing under half the median history, which would shrink the
        sample for every asset), or a weight cap or floor the priced assets cannot
        satisfy.
    """
    panel = load_returns_panel(
        session,
        asof,
        universe,
        constraints=constraints,
        base_currency=base_currency,
        action="optimize",
    )
    if isinstance(panel, str):
        return err(panel)
    returns = panel.returns
    config = _calibrate_risk_aversion(panel.config, returns)
    cleaned = _clean_weights(fit_mean_risk(returns, config), config)
    portfolio = _in_sample_portfolio(returns, cleaned, config)

    assets = [str(col) for col in returns.columns]
    weights = dict(zip(assets, map(float, cleaned), strict=True))
    metrics = annualized_metrics(portfolio, panel.periods)

    return ok(
        {
            "weights": weights,
            "metrics": metrics,
            "assets": assets,
            "n_observations": int(returns.shape[0]),
            "missing": panel.missing,
            "risk_aversion": float(config.risk_aversion),
            "periods_per_year": panel.periods,
            "risk_free_rate": float(config.risk_free_rate),
        }
    )


__all__ = [
    "ANNUAL_RISK_FREE_RATE_KEY",
    "ReturnsPanel",
    "fit_mean_risk",
    "load_returns_panel",
    "optimize_portfolio",
]
