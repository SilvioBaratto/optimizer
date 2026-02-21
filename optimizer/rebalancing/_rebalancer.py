"""Rebalancing decision logic."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.rebalancing._config import (
    HybridRebalancingConfig,
    ThresholdRebalancingConfig,
    ThresholdType,
)

logger = logging.getLogger(__name__)


def compute_drifted_weights(
    weights: npt.NDArray[np.float64],
    returns: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute portfolio weights after one period of returns.

    Parameters
    ----------
    weights : ndarray, shape (n_assets,)
        Current portfolio weights (must sum to 1).
    returns : ndarray, shape (n_assets,)
        Single-period asset returns.

    Returns
    -------
    ndarray, shape (n_assets,)
        Drifted weights after applying returns.
    """
    grown = weights * (1.0 + returns)
    total = grown.sum()
    if total == 0.0:
        return grown
    return grown / total


def compute_turnover(
    current_weights: npt.NDArray[np.float64],
    target_weights: npt.NDArray[np.float64],
) -> float:
    """Compute one-way turnover between current and target weights.

    Parameters
    ----------
    current_weights : ndarray, shape (n_assets,)
        Current portfolio weights.
    target_weights : ndarray, shape (n_assets,)
        Target portfolio weights.

    Returns
    -------
    float
        One-way turnover (sum of absolute weight changes / 2).
    """
    return float(np.abs(current_weights - target_weights).sum() / 2.0)


def compute_rebalancing_cost(
    current_weights: npt.NDArray[np.float64],
    target_weights: npt.NDArray[np.float64],
    transaction_costs: float | npt.NDArray[np.float64],
) -> float:
    """Compute the total transaction cost of rebalancing.

    Parameters
    ----------
    current_weights : ndarray, shape (n_assets,)
        Current portfolio weights.
    target_weights : ndarray, shape (n_assets,)
        Target portfolio weights.
    transaction_costs : float or ndarray
        Per-unit transaction cost (scalar for uniform costs,
        array for asset-specific costs).

    Returns
    -------
    float
        Total rebalancing cost as a fraction of portfolio value.
    """
    trades = np.abs(target_weights - current_weights)
    return float(np.sum(transaction_costs * trades))


def should_rebalance(
    current_weights: npt.NDArray[np.float64],
    target_weights: npt.NDArray[np.float64],
    config: ThresholdRebalancingConfig | None = None,
) -> bool:
    """Determine whether any asset breaches the drift threshold.

    Parameters
    ----------
    current_weights : ndarray, shape (n_assets,)
        Current (drifted) portfolio weights.
    target_weights : ndarray, shape (n_assets,)
        Target portfolio weights from the optimiser.
    config : ThresholdRebalancingConfig or None
        Threshold configuration.  Defaults to absolute 5pp threshold.

    Returns
    -------
    bool
        ``True`` if at least one asset breaches the threshold.
    """
    if config is None:
        config = ThresholdRebalancingConfig()

    drifts = np.abs(current_weights - target_weights)

    if config.threshold_type == ThresholdType.ABSOLUTE:
        return bool(np.any(drifts > config.threshold))

    # Relative threshold: drift / target (guard against zero targets)
    safe_targets = np.where(target_weights > 0, target_weights, np.inf)
    relative_drifts = drifts / safe_targets
    return bool(np.any(relative_drifts > config.threshold))


def should_rebalance_hybrid(
    current_weights: npt.NDArray[np.float64],
    target_weights: npt.NDArray[np.float64],
    config: HybridRebalancingConfig,
    current_date: pd.Timestamp,
    last_review_date: pd.Timestamp,
) -> bool:
    """Determine whether to rebalance under a hybrid calendar+threshold policy.

    Returns ``True`` only when **both** conditions are met:

    1. ``current_date`` is a calendar review date — at least
       ``config.calendar.trading_days`` business days have elapsed since
       ``last_review_date``.
    2. At least one asset's drift exceeds the threshold defined in
       ``config.threshold``.

    Between calendar review dates the function always returns ``False``
    regardless of how much drift has accumulated.

    Parameters
    ----------
    current_weights : ndarray, shape (n_assets,)
        Current (drifted) portfolio weights.
    target_weights : ndarray, shape (n_assets,)
        Target portfolio weights from the optimiser.
    config : HybridRebalancingConfig
        Hybrid configuration combining calendar and threshold rules.
    current_date : pd.Timestamp
        The date being evaluated.
    last_review_date : pd.Timestamp
        Date of the last calendar review.

    Returns
    -------
    bool
        ``True`` only if it is a calendar review date AND drift exceeds
        the threshold.
    """
    elapsed = len(pd.bdate_range(last_review_date, current_date)) - 1
    if elapsed < config.calendar.trading_days:
        return False
    return should_rebalance(current_weights, target_weights, config.threshold)
