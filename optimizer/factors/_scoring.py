"""Composite scoring from standardized factor scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer.factors._config import (
    FACTOR_GROUP_MAPPING,
    GROUP_WEIGHT_TIER,
    CompositeMethod,
    CompositeScoringConfig,
    FactorGroupType,
    GroupWeight,
)
from optimizer.factors._ml_scoring import (
    FittedMLModel,
    fit_gbt_composite,
    fit_ridge_composite,
    predict_composite_scores,
)
from optimizer.factors._validation import compute_icir


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

        # Weighted average, ignoring NaN
        group_scores[group.value] = sub.where(cov).mean(axis=1)

    return pd.DataFrame(group_scores, index=standardized_factors.index)


def compute_equal_weight_composite(
    group_scores: pd.DataFrame,
    config: CompositeScoringConfig | None = None,
) -> pd.Series:
    """Equal-weight composite with core/supplementary tiering.

    Parameters
    ----------
    group_scores : pd.DataFrame
        Tickers x groups matrix.
    config : CompositeScoringConfig or None
        Scoring configuration.

    Returns
    -------
    pd.Series
        Composite score per ticker.
    """
    if config is None:
        config = CompositeScoringConfig()

    weights: dict[str, float] = {}
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

    total_weight = sum(weights.values())
    composite = pd.Series(0.0, index=group_scores.index)
    for col, w in weights.items():
        composite = composite + (w / total_weight) * group_scores[col].fillna(0.0)

    return composite


def compute_ic_weighted_composite(
    group_scores: pd.DataFrame,
    ic_history: pd.DataFrame,
    config: CompositeScoringConfig | None = None,
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
    for group in FactorGroupType:
        if group.value not in group_scores.columns:
            continue
        ic_val = recent_ic.get(group.value, 0.0)
        if np.isnan(ic_val):
            ic_val = 0.0
        tier = GROUP_WEIGHT_TIER[group]
        tier_mult = (
            config.core_weight
            if tier == GroupWeight.CORE
            else config.supplementary_weight
        )
        # IC-weighted: use absolute IC to avoid sign issues,
        # then apply sign separately
        weights[group.value] = max(abs(ic_val), 0.001) * tier_mult

    if not weights:
        return pd.Series(0.0, index=group_scores.index)

    total_weight = sum(weights.values())
    composite = pd.Series(0.0, index=group_scores.index)
    for col, w in weights.items():
        composite = composite + (w / total_weight) * group_scores[col].fillna(0.0)

    return composite


def compute_icir_weighted_composite(
    group_scores: pd.DataFrame,
    ic_series_per_group: dict[str, pd.Series],
    config: CompositeScoringConfig | None = None,
) -> pd.Series:
    """ICIR-weighted composite score.

    Weights each group by ``|ICIR| = |mean(IC) / std(IC)|``, normalised
    to sum to 1.  Groups with zero or undefined ICIR receive zero weight.
    Falls back to equal-weight when all groups have ICIR = 0.

    Parameters
    ----------
    group_scores : pd.DataFrame
        Tickers x groups matrix.
    ic_series_per_group : dict[str, pd.Series]
        Per-group IC time series.  Keys must match ``group_scores`` columns.
    config : CompositeScoringConfig or None
        Scoring configuration.

    Returns
    -------
    pd.Series
        Composite score per ticker.
    """
    if config is None:
        config = CompositeScoringConfig()

    weights: dict[str, float] = {}
    for group in FactorGroupType:
        if group.value not in group_scores.columns:
            continue
        ic_s = ic_series_per_group.get(group.value, pd.Series(dtype=float))
        icir = compute_icir(ic_s)
        tier = GROUP_WEIGHT_TIER[group]
        tier_mult = (
            config.core_weight
            if tier == GroupWeight.CORE
            else config.supplementary_weight
        )
        weights[group.value] = abs(icir) * tier_mult

    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return compute_equal_weight_composite(group_scores, config)

    composite = pd.Series(0.0, index=group_scores.index)
    for col, w in weights.items():
        composite = composite + (w / total_weight) * group_scores[col].fillna(0.0)

    return composite


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
        )
    return predict_composite_scores(model, standardized_factors)


def compute_composite_score(
    standardized_factors: pd.DataFrame,
    coverage: pd.DataFrame,
    config: CompositeScoringConfig | None = None,
    ic_history: pd.DataFrame | None = None,
    training_scores: pd.DataFrame | None = None,
    training_returns: pd.Series | None = None,
) -> pd.Series:
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

    Returns
    -------
    pd.Series
        Composite score per ticker.
    """
    if config is None:
        config = CompositeScoringConfig()

    group_scores = compute_group_scores(standardized_factors, coverage)

    if config.method == CompositeMethod.IC_WEIGHTED:
        if ic_history is None:
            msg = "ic_history required for IC_WEIGHTED composite method"
            raise ValueError(msg)
        return compute_ic_weighted_composite(group_scores, ic_history, config)

    if config.method == CompositeMethod.ICIR_WEIGHTED:
        if ic_history is None:
            msg = "ic_history required for ICIR_WEIGHTED composite method"
            raise ValueError(msg)
        ic_series_per_group = {
            col: ic_history[col].dropna() for col in ic_history.columns
        }
        return compute_icir_weighted_composite(
            group_scores, ic_series_per_group, config
        )

    if config.method in (CompositeMethod.RIDGE_WEIGHTED, CompositeMethod.GBT_WEIGHTED):
        if training_scores is None or training_returns is None:
            msg = (
                "training_scores and training_returns are required for "
                f"{config.method.value} composite method"
            )
            raise ValueError(msg)
        return compute_ml_composite(
            standardized_factors, training_scores, training_returns, config
        )

    return compute_equal_weight_composite(group_scores, config)
