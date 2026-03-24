"""Composite scoring from standardized factor scores."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from optimizer.exceptions import ConfigurationError
from optimizer.factors._config import (
    FACTOR_GROUP_MAPPING,
    GROUP_WEIGHT_TIER,
    CompositeMethod,
    CompositeScoringConfig,
    FactorGroupType,
    GroupWeight,
    ICFallbackStrategy,
)
from optimizer.factors._ml_scoring import (
    FittedMLModel,
    fit_gbt_composite,
    fit_ridge_composite,
    predict_composite_scores,
)
from optimizer.factors._validation import compute_icir

logger = logging.getLogger(__name__)


def _renormalized_weighted_composite(
    group_scores: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Weighted average over available (non-NaN) groups per ticker.

    For each ticker, only groups with a non-NaN score contribute.
    Weights are renormalized to sum to 1 over those available groups.
    Tickers with no available groups receive NaN.
    """
    cols = [c for c in weights if c in group_scores.columns]
    if not cols:
        return pd.Series(np.nan, index=group_scores.index)

    w_arr = np.array([weights[c] for c in cols])
    scores = group_scores[cols]
    available = scores.notna()

    # Broadcast raw weights and mask where score is NaN
    w_matrix = available.values * w_arr  # (tickers, groups)
    row_sums = w_matrix.sum(axis=1)

    # Avoid division by zero: tickers with no coverage get NaN
    safe_sums = np.where(row_sums == 0.0, np.nan, row_sums)
    normed = w_matrix / safe_sums[:, np.newaxis]

    # fillna(0) is safe because normed weights are zero where score is NaN
    composite = (scores.fillna(0.0).values * normed).sum(axis=1)
    return pd.Series(composite, index=group_scores.index)


def _handle_zero_weight_fallback(
    group_scores: pd.DataFrame,
    config: CompositeScoringConfig,
    group_weights: dict[str, float] | None,
    signal_type: str,
    n_negative: int,
) -> pd.Series:
    """Handle the zero-total-weight fallback for IC/ICIR weighting.

    Called when every factor group has non-positive IC or ICIR so that the
    pre-normalisation weight vector sums to zero.
    """
    n_groups = len(group_scores.columns)
    msg = (
        f"All {n_negative}/{n_groups} factor groups have non-positive "
        f"{signal_type}; "
        f"falling back to {config.ic_fallback_strategy.value!r} strategy. "
        "Consider increasing ic_lookback or checking factor data quality."
    )
    logger.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)

    strategy = config.ic_fallback_strategy
    if strategy == ICFallbackStrategy.EQUAL_WEIGHT:
        return compute_equal_weight_composite(group_scores, config, group_weights)

    if strategy == ICFallbackStrategy.NAN:
        return pd.Series(np.nan, index=group_scores.index)

    # ICFallbackStrategy.RAISE
    raise_msg = (
        f"All {n_negative}/{n_groups} factor groups have non-positive "
        f"{signal_type}. Cannot compute weighted composite."
    )
    raise ConfigurationError(raise_msg)


def compute_group_scores(
    standardized_factors: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Average factor scores within each group.

    Parameters
    ----------
    standardized_factors : pd.DataFrame
        Tickers x factors matrix of standardized scores.
    coverage : pd.DataFrame
        Boolean matrix of non-NaN coverage.

    Returns
    -------
    pd.DataFrame
        Tickers x groups matrix of group-level scores.
    """
    group_scores: dict[str, pd.Series] = {}

    for group in FactorGroupType:
        # Find columns belonging to this group
        group_cols = [
            ft.value
            for ft, fg in FACTOR_GROUP_MAPPING.items()
            if fg == group and ft.value in standardized_factors.columns
        ]
        if not group_cols:
            continue

        sub = standardized_factors[group_cols]
        cov = coverage[group_cols]

        # Coverage-weighted mean: sum(cov * score) / sum(cov)
        numerator = (sub * cov).sum(axis=1)
        denominator = cov.sum(axis=1)
        group_scores[group.value] = numerator / denominator.replace(0, np.nan)

    return pd.DataFrame(group_scores, index=standardized_factors.index)


def compute_equal_weight_composite(
    group_scores: pd.DataFrame,
    config: CompositeScoringConfig | None = None,
    group_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Equal-weight composite with core/supplementary tiering.

    Parameters
    ----------
    group_scores : pd.DataFrame
        Tickers x groups matrix.
    config : CompositeScoringConfig or None
        Scoring configuration.
    group_weights : dict[str, float] or None
        Pre-computed group weights (e.g. from regime tilts). When provided,
        skip tier-based derivation and use these weights directly.

    Returns
    -------
    pd.Series
        Composite score per ticker.
    """
    if config is None:
        config = CompositeScoringConfig()

    if group_weights is not None:
        weights = {k: v for k, v in group_weights.items() if k in group_scores.columns}
    else:
        weights = {}
        for group in FactorGroupType:
            if group.value not in group_scores.columns:
                continue
            tier = GROUP_WEIGHT_TIER[group]
            weights[group.value] = (
                config.core_weight
                if tier == GroupWeight.CORE
                else config.supplementary_weight
            )

    if not weights:
        return pd.Series(0.0, index=group_scores.index)

    return _renormalized_weighted_composite(group_scores, weights)


def compute_ic_weighted_composite(
    group_scores: pd.DataFrame,
    ic_history: pd.DataFrame,
    config: CompositeScoringConfig | None = None,
    group_weights: dict[str, float] | None = None,
) -> pd.Series:
    """IC-weighted composite score.

    Uses trailing information coefficient history to weight groups.

    Parameters
    ----------
    group_scores : pd.DataFrame
        Tickers x groups matrix.
    ic_history : pd.DataFrame
        Periods x groups matrix of IC values.
    config : CompositeScoringConfig or None
        Scoring configuration.
    group_weights : dict[str, float] or None
        Pre-computed group weights (e.g. from regime tilts). When provided,
        use as tier multipliers instead of config core/supplementary weights.

    Returns
    -------
    pd.Series
        Composite score per ticker.
    """
    if config is None:
        config = CompositeScoringConfig()

    # Use trailing IC mean, capped at lookback window
    lookback = min(config.ic_lookback, len(ic_history))
    recent_ic = ic_history.iloc[-lookback:].mean()

    # Apply core/supplementary tiering as a multiplier
    weights: dict[str, float] = {}
    n_negative = 0
    for group in FactorGroupType:
        if group.value not in group_scores.columns:
            continue
        ic_val = recent_ic.get(group.value, 0.0)
        if np.isnan(ic_val):
            ic_val = 0.0
        if ic_val < 0.0:
            n_negative += 1
        if group_weights is not None:
            tier_mult = group_weights.get(group.value, 0.0)
        else:
            tier = GROUP_WEIGHT_TIER[group]
            tier_mult = (
                config.core_weight
                if tier == GroupWeight.CORE
                else config.supplementary_weight
            )
        # IC-weighted: clamp negative IC to zero (theory: negative-IC groups
        # should not contribute positively to the composite)
        weights[group.value] = max(ic_val, 0.0) * tier_mult

    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return _handle_zero_weight_fallback(
            group_scores, config, group_weights, "IC", n_negative
        )

    return _renormalized_weighted_composite(group_scores, weights)


def compute_icir_weighted_composite(
    group_scores: pd.DataFrame,
    ic_series_per_group: dict[str, pd.Series],
    config: CompositeScoringConfig | None = None,
    group_weights: dict[str, float] | None = None,
) -> pd.Series:
    """ICIR-weighted composite score.

    Weights each group by ``max(ICIR, 0) = max(mean(IC) / std(IC), 0)``,
    normalised to sum to 1.  Groups with zero, negative, or undefined ICIR
    receive zero weight.  Falls back to equal-weight when all groups have
    ICIR <= 0.

    Parameters
    ----------
    group_scores : pd.DataFrame
        Tickers x groups matrix.
    ic_series_per_group : dict[str, pd.Series]
        Per-group IC time series.  Keys must match ``group_scores`` columns.
    config : CompositeScoringConfig or None
        Scoring configuration.
    group_weights : dict[str, float] or None
        Pre-computed group weights (e.g. from regime tilts). When provided,
        use as tier multipliers instead of config core/supplementary weights.

    Returns
    -------
    pd.Series
        Composite score per ticker.
    """
    if config is None:
        config = CompositeScoringConfig()

    weights: dict[str, float] = {}
    n_negative = 0
    for group in FactorGroupType:
        if group.value not in group_scores.columns:
            continue
        ic_s = ic_series_per_group.get(group.value, pd.Series(dtype=float))
        icir = compute_icir(ic_s)
        if icir < 0.0:
            n_negative += 1
        if group_weights is not None:
            tier_mult = group_weights.get(group.value, 0.0)
        else:
            tier = GROUP_WEIGHT_TIER[group]
            tier_mult = (
                config.core_weight
                if tier == GroupWeight.CORE
                else config.supplementary_weight
            )
        weights[group.value] = max(icir, 0.0) * tier_mult

    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return _handle_zero_weight_fallback(
            group_scores, config, group_weights, "ICIR", n_negative
        )

    return _renormalized_weighted_composite(group_scores, weights)


def compute_ml_composite(
    standardized_factors: pd.DataFrame,
    training_scores: pd.DataFrame,
    training_returns: pd.Series,
    config: CompositeScoringConfig,
) -> pd.Series:
    """ML composite score using ridge regression or gradient-boosted trees.

    Trains the model on historical ``(training_scores, training_returns)``
    and predicts on the current-period ``standardized_factors``.  The
    prediction is normalised to zero mean and unit variance.

    The training window must end strictly before the prediction date to
    avoid look-ahead bias; callers are responsible for this temporal split.

    Parameters
    ----------
    standardized_factors : pd.DataFrame
        Current-period tickers x factors matrix (prediction target).
    training_scores : pd.DataFrame
        Historical tickers x factors matrix aligned with
        ``training_returns``.
    training_returns : pd.Series
        Forward return per ticker for the training period.
    config : CompositeScoringConfig
        Must have ``method`` set to ``RIDGE_WEIGHTED`` or ``GBT_WEIGHTED``.

    Returns
    -------
    pd.Series
        Normalised composite score per ticker (zero mean, unit variance).
    """
    model: FittedMLModel
    if config.method == CompositeMethod.RIDGE_WEIGHTED:
        model = fit_ridge_composite(
            training_scores, training_returns, config.ridge_alpha
        )
    else:
        model = fit_gbt_composite(
            training_scores,
            training_returns,
            config.gbt_max_depth,
            config.gbt_n_estimators,
            config.gbt_random_state,
        )
    return predict_composite_scores(model, standardized_factors)


def _apply_coverage_gating(
    composite: pd.Series,
    group_scores: pd.DataFrame,
    config: CompositeScoringConfig,
) -> pd.Series | pd.DataFrame:
    """Apply minimum-coverage threshold and optional coverage_ratio output."""
    if config.min_coverage_groups > 0:
        n_available = group_scores.notna().sum(axis=1)
        composite = composite.where(n_available >= config.min_coverage_groups)

    if config.return_coverage:
        n_groups = len(group_scores.columns)
        coverage_ratio = group_scores.notna().sum(axis=1) / max(n_groups, 1)
        return pd.DataFrame(
            {"composite": composite, "coverage_ratio": coverage_ratio},
            index=composite.index,
        )

    return composite


def compute_composite_score(
    standardized_factors: pd.DataFrame,
    coverage: pd.DataFrame,
    config: CompositeScoringConfig | None = None,
    ic_history: pd.DataFrame | None = None,
    training_scores: pd.DataFrame | None = None,
    training_returns: pd.Series | None = None,
    group_weights: dict[str, float] | None = None,
) -> pd.Series | pd.DataFrame:
    """Compute composite score from standardized factors.

    Parameters
    ----------
    standardized_factors : pd.DataFrame
        Tickers x factors matrix.
    coverage : pd.DataFrame
        Boolean coverage matrix.
    config : CompositeScoringConfig or None
        Scoring configuration.
    ic_history : pd.DataFrame or None
        Required when ``config.method`` is ``IC_WEIGHTED`` or
        ``ICIR_WEIGHTED``.  Columns must match group names; each column
        is treated as the IC time series for that group.
    training_scores : pd.DataFrame or None
        Required when ``config.method`` is ``RIDGE_WEIGHTED`` or
        ``GBT_WEIGHTED``.  Historical tickers x factors matrix used to
        train the ML model (must not overlap with current-period data).
    training_returns : pd.Series or None
        Required when ``config.method`` is ``RIDGE_WEIGHTED`` or
        ``GBT_WEIGHTED``.  Forward returns aligned with ``training_scores``.
    group_weights : dict[str, float] or None
        Pre-computed group weights (e.g. from regime tilts). Threaded
        through to the inner scoring functions.

    Returns
    -------
    pd.Series or pd.DataFrame
        Composite score per ticker.  When ``config.return_coverage`` is
        True, returns a DataFrame with ``composite`` and ``coverage_ratio``
        columns.
    """
    if config is None:
        config = CompositeScoringConfig()

    group_scores = compute_group_scores(standardized_factors, coverage)

    if config.method == CompositeMethod.IC_WEIGHTED:
        if ic_history is None:
            msg = "ic_history required for IC_WEIGHTED composite method"
            raise ConfigurationError(msg)
        composite = compute_ic_weighted_composite(
            group_scores, ic_history, config, group_weights
        )
        return _apply_coverage_gating(composite, group_scores, config)

    if config.method == CompositeMethod.ICIR_WEIGHTED:
        if ic_history is None:
            msg = "ic_history required for ICIR_WEIGHTED composite method"
            raise ConfigurationError(msg)
        ic_series_per_group = {
            col: ic_history[col].dropna() for col in ic_history.columns
        }
        composite = compute_icir_weighted_composite(
            group_scores, ic_series_per_group, config, group_weights
        )
        return _apply_coverage_gating(composite, group_scores, config)

    if config.method in (CompositeMethod.RIDGE_WEIGHTED, CompositeMethod.GBT_WEIGHTED):
        if training_scores is None or training_returns is None:
            msg = (
                "training_scores and training_returns are required for "
                f"{config.method.value} composite method"
            )
            raise ConfigurationError(msg)
        composite = compute_ml_composite(
            standardized_factors, training_scores, training_returns, config
        )
        return _apply_coverage_gating(composite, group_scores, config)

    composite = compute_equal_weight_composite(group_scores, config, group_weights)
    return _apply_coverage_gating(composite, group_scores, config)
