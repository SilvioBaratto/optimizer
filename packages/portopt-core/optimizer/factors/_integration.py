"""Bridge factor scores to portfolio optimization inputs."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimizer.exceptions import ConfigurationError
from optimizer.factors._config import FactorIntegrationConfig
from optimizer.rebalancing._rebalancer import compute_turnover

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorExposureConstraints:
    """Enforceable linear constraints on portfolio factor exposure.

    Encodes per-factor inequalities lb_g <= sum_i w_i * z_{i,g} <= ub_g as
    a pair of matrices ready to be passed directly to
    ``skfolio.optimization.MeanRisk`` (or any optimizer that accepts
    ``left_inequality`` / ``right_inequality``).

    Attributes:
        left_inequality: Inequality matrix ``A`` in the constraint
            ``A @ w <= b`` with shape ``(2 * n_factors, n_assets)``.
            Two rows per factor: ``-z`` (lower bound) and ``+z`` (upper bound).
        right_inequality: Bound vector ``b`` in the constraint
            ``A @ w <= b`` with shape ``(2 * n_factors,)``.
        factor_names: Names of the constrained factors in the same order as
            the row pairs in ``left_inequality``.
        lower_bounds: Lower exposure bound per factor, shape ``(n_factors,)``.
        upper_bounds: Upper exposure bound per factor, shape ``(n_factors,)``.
    """

    left_inequality: np.ndarray
    right_inequality: np.ndarray
    factor_names: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


def build_factor_bl_views(
    composite_scores: pd.Series,
    selected_tickers: pd.Index,
    config: FactorIntegrationConfig,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Generate Black-Litterman absolute views from composite factor scores.

    For each selected ticker with composite score ``z_i``, generates a view::

        E[r_i] = (rf + market_premium + z_i * score_premium) / 252

    Args:
        composite_scores: Composite factor scores indexed by ticker.
        selected_tickers: Tickers in the portfolio.
        config: Integration configuration with rf, market premium, and score
            premium.

    Returns:
        ``(views, confidences)`` where views are BL-compatible strings like
        ``"AAPL == 0.00045"`` and confidences are in [0, 1].
    """
    scores = composite_scores.reindex(selected_tickers).dropna()
    if len(scores) == 0:
        return (), ()

    views: list[str] = []
    raw_confidences: list[float] = []

    for ticker, z_i in scores.items():
        daily_er = (
            config.risk_free_rate
            + config.market_risk_premium
            + float(z_i) * config.score_premium
        ) / 252.0
        views.append(f"{ticker} == {daily_er:.8f}")
        raw_confidences.append(abs(float(z_i)))

    # Scale confidences to [0, cap].  Idzorek confidence=1.0 means
    # "posterior equals view exactly" — extreme concentration.
    # Typical calibrations use 0.25–0.50 to blend view with prior.
    cap = config.view_confidence_cap
    max_abs_z = max(raw_confidences) if raw_confidences else 1.0
    if max_abs_z > 0:
        confidences = [(c / max_abs_z) * cap for c in raw_confidences]
    else:
        confidences = [cap] * len(raw_confidences)

    return tuple(views), tuple(confidences)


def build_factor_exposure_constraints(
    factor_scores: pd.DataFrame,
    bounds: tuple[float, float] | dict[str, tuple[float, float]],
) -> FactorExposureConstraints:
    """Build enforceable linear factor exposure constraints.

    For each factor ``g``, the constraint enforces::

        lb_g <= sum_i w_i * z_{i,g} <= ub_g

    The result is expressed as ``left_inequality @ w <= right_inequality``
    (two rows per factor) and can be passed directly to
    ``skfolio.optimization.MeanRisk`` via its
    ``left_inequality`` / ``right_inequality`` constructor arguments.

    Args:
        factor_scores: Tickers x factors matrix of standardised factor scores.
            The tickers must match the assets used in the optimizer ``fit``.
        bounds: Exposure bounds applied to every factor (uniform) when given as
            a single ``(lower, upper)`` tuple, or per-factor bounds when given
            as a dict mapping factor name to ``(lower, upper)``.

    Returns:
        Dataclass holding ``left_inequality``, ``right_inequality``, and
        metadata. Pass ``left_inequality`` and ``right_inequality`` as keyword
        arguments to the optimizer.

    Raises:
        ConfigurationError: If ``bounds`` is a dict and a factor name has no
            corresponding entry.

    Warns:
        UserWarning: If the equal-weight portfolio exposure lies outside
            ``[lb, ub]`` for any factor, meaning the constraint may be
            infeasible or very tight under a balanced allocation.
    """
    n_assets, n_factors = factor_scores.shape
    factor_names = list(factor_scores.columns)

    lower_arr = np.empty(n_factors)
    upper_arr = np.empty(n_factors)
    if isinstance(bounds, dict):
        for k, name in enumerate(factor_names):
            if name not in bounds:
                msg = f"Factor '{name}' has no entry in bounds dict."
                raise ConfigurationError(msg)
            lb, ub = bounds[name]
            lower_arr[k] = lb
            upper_arr[k] = ub
    else:
        lb, ub = bounds
        lower_arr[:] = lb
        upper_arr[:] = ub

    # Build inequality matrices: A @ w <= b
    # lb <= z @ w  =>  -z @ w <= -lb
    # z @ w <= ub
    scores_matrix = factor_scores.to_numpy(dtype=float)  # (n_assets, n_factors)

    A = np.empty((2 * n_factors, n_assets))
    b = np.empty(2 * n_factors)
    for k in range(n_factors):
        z = scores_matrix[:, k]
        A[2 * k] = -z
        b[2 * k] = -lower_arr[k]
        A[2 * k + 1] = z
        b[2 * k + 1] = upper_arr[k]

    equal_weight = np.ones(n_assets) / n_assets
    for k, name in enumerate(factor_names):
        z = scores_matrix[:, k]
        ew_exposure = float(np.dot(equal_weight, z))
        if not (lower_arr[k] <= ew_exposure <= upper_arr[k]):
            warnings.warn(
                f"Factor '{name}': equal-weight exposure {ew_exposure:.4f} "
                f"lies outside [{lower_arr[k]:.4f}, {upper_arr[k]:.4f}]. "
                "The constraint may be infeasible.",
                UserWarning,
                stacklevel=2,
            )

    return FactorExposureConstraints(
        left_inequality=A,
        right_inequality=b,
        factor_names=factor_names,
        lower_bounds=lower_arr,
        upper_bounds=upper_arr,
    )


def estimate_factor_premia(
    factor_mimicking_returns: pd.DataFrame,
) -> dict[str, float]:
    """Estimate annualized factor premia from long-short returns.

    Args:
        factor_mimicking_returns: Dates x factors matrix of factor-mimicking
            portfolio returns.

    Returns:
        Annualized premium per factor.
    """
    mean_daily = factor_mimicking_returns.mean()
    annualized = mean_daily * 252
    return dict(annualized)


def build_factor_integration(
    config: FactorIntegrationConfig,
    composite_scores: pd.Series,
    standardized_factors: pd.DataFrame,
    selected_tickers: pd.Index,
) -> tuple[object | None, FactorExposureConstraints | None]:
    """Build factor-to-optimizer integration objects.

    Depending on ``config.use_black_litterman``, either creates a
    Black-Litterman prior from composite scores or builds linear
    factor exposure constraints.

    Args:
        config: Integration configuration.
        composite_scores: Composite factor scores indexed by ticker.
        standardized_factors: Standardized factor scores (tickers x factors).
        selected_tickers: Tickers selected for the portfolio.

    Returns:
        ``(prior, constraints)`` — one of the two will be set, the other
        ``None``.
    """
    if config.use_black_litterman:
        views, confidences = build_factor_bl_views(
            composite_scores, selected_tickers, config
        )
        if len(views) == 0:
            logger.warning("No BL views generated from factor scores")
            return None, None

        from optimizer.views._config import (
            BlackLittermanConfig,
            ViewUncertaintyMethod,
        )
        from optimizer.views._factory import build_black_litterman

        bl_config = BlackLittermanConfig(
            views=views,
            view_confidences=confidences,
            uncertainty_method=ViewUncertaintyMethod.IDZOREK,
        )
        prior = build_black_litterman(bl_config)
        return prior, None
    else:
        selected_factors = standardized_factors.reindex(selected_tickers).dropna(
            how="all"
        )
        if selected_factors.empty:
            logger.warning("No factor data for selected tickers")
            return None, None

        constraints = build_factor_exposure_constraints(
            selected_factors,
            bounds=(config.exposure_lower_bound, config.exposure_upper_bound),
        )
        return None, constraints


@dataclass
class NetAlphaResult:
    """Result of net alpha calculation after transaction cost deduction.

    Attributes:
        gross_alpha: Annualised IC-based alpha proxy:
            ``mean(IC) * sqrt(annualisation)``.
        avg_turnover: Mean one-way turnover across consecutive rebalancing
            dates, computed via ``compute_turnover``.
        total_cost: Cost deduction: ``avg_turnover * cost_bps / 10_000``.
        net_alpha: Net annualised alpha after cost deduction:
            ``gross_alpha - total_cost``.
        net_icir: Net information coefficient information ratio:
            ``net_alpha / (std(IC) * sqrt(annualisation))``. Zero when the
            IC series has zero variance.
    """

    gross_alpha: float
    avg_turnover: float
    total_cost: float
    net_alpha: float
    net_icir: float


def compute_net_alpha(
    ic_series: pd.Series,
    weights_history: pd.DataFrame,
    cost_bps: float = 10.0,
    annualisation: int = 252,
) -> NetAlphaResult:
    """Compute factor net alpha after deducting turnover-based transaction costs.

    Combines IC-based gross alpha with the turnover cost from a weights
    history to produce a single net performance metric::

        gross_alpha  = mean(IC) * sqrt(annualisation)
        avg_turnover = mean one-way turnover across rebalancing dates
        total_cost   = avg_turnover * cost_bps / 10_000
        net_alpha    = gross_alpha - total_cost
        net_icir     = net_alpha / (std(IC) * sqrt(annualisation))

    Args:
        ic_series: Time series of period information coefficients (Spearman or
            Pearson rank correlation between factor scores and forward returns),
            one value per rebalancing period.
        weights_history: Portfolio weights at each rebalancing date: rows are
            dates, columns are assets. Turnover is computed between every pair
            of consecutive rows.
        cost_bps: Round-trip transaction cost in basis points.
        annualisation: Number of periods per year (252 for daily, 12 for
            monthly).

    Returns:
        Dataclass with ``gross_alpha``, ``avg_turnover``, ``total_cost``,
        ``net_alpha``, and ``net_icir``.
    """
    ic_values = ic_series.dropna().to_numpy(dtype=float)
    ic_mean = float(np.mean(ic_values)) if len(ic_values) > 0 else 0.0
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0

    gross_alpha = ic_mean * float(np.sqrt(annualisation))

    turnovers: list[float] = []
    for i in range(1, len(weights_history)):
        t = compute_turnover(
            weights_history.iloc[i - 1].to_numpy(dtype=float),
            weights_history.iloc[i].to_numpy(dtype=float),
        )
        turnovers.append(t)
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    total_cost = avg_turnover * cost_bps / 10_000.0
    net_alpha = gross_alpha - total_cost

    annual_ic_vol = ic_std * float(np.sqrt(annualisation))
    net_icir = net_alpha / annual_ic_vol if annual_ic_vol > 1e-12 else 0.0

    return NetAlphaResult(
        gross_alpha=gross_alpha,
        avg_turnover=avg_turnover,
        total_cost=total_cost,
        net_alpha=net_alpha,
        net_icir=net_icir,
    )
