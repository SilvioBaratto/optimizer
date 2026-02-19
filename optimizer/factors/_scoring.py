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


def compute_composite_score(
    standardized_factors: pd.DataFrame,
    coverage: pd.DataFrame,
    config: CompositeScoringConfig | None = None,
    ic_history: pd.DataFrame | None = None,
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
        Required when ``config.method`` is ``IC_WEIGHTED``.

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

    return compute_equal_weight_composite(group_scores, config)
