"""Rebalancing decision logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from skfolio.model_selection import WalkForward

from optimizer.rebalancing._config import (
    CalendarRebalancingConfig,
    HybridRebalancingConfig,
    ThresholdRebalancingConfig,
    ThresholdType,
)

logger = logging.getLogger(__name__)


def _as_1d(name: str, arr: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce *arr* to a 1-D float64 array, raising on wrong dimensionality."""
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    return out


def _check_aligned(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    a_name: str,
    b_name: str,
) -> None:
    """Raise if two 1-D weight vectors differ in length (silent-broadcast guard)."""
    if a.shape != b.shape:
        raise ValueError(
            f"{a_name} and {b_name} must have the same length, "
            f"got {a.shape[0]} and {b.shape[0]}"
        )


def compute_drifted_weights(
    weights: npt.ArrayLike,
    returns: npt.ArrayLike,
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
        Drifted weights after applying returns.  If the post-drift portfolio
        value is zero or non-finite the raw grown weights are returned
        unnormalised.
    """
    w = _as_1d("weights", weights)
    r = _as_1d("returns", returns)
    _check_aligned(w, r, "weights", "returns")
    grown = w * (1.0 + r)
    total = grown.sum()
    if not np.isfinite(total) or total == 0.0:
        return grown
    return grown / total


def compute_turnover(
    current_weights: npt.ArrayLike,
    target_weights: npt.ArrayLike,
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
    cur = _as_1d("current_weights", current_weights)
    tgt = _as_1d("target_weights", target_weights)
    _check_aligned(cur, tgt, "current_weights", "target_weights")
    return float(np.abs(cur - tgt).sum() / 2.0)


def compute_rebalancing_cost(
    current_weights: npt.ArrayLike,
    target_weights: npt.ArrayLike,
    transaction_costs: float | npt.ArrayLike,
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
    cur = _as_1d("current_weights", current_weights)
    tgt = _as_1d("target_weights", target_weights)
    _check_aligned(cur, tgt, "current_weights", "target_weights")
    trades = np.abs(tgt - cur)
    costs = np.asarray(transaction_costs, dtype=np.float64)
    if costs.ndim not in (0, 1):
        raise ValueError(
            f"transaction_costs must be a scalar or 1-D, got shape {costs.shape}"
        )
    if costs.ndim == 1 and costs.shape != trades.shape:
        raise ValueError(
            "transaction_costs and weights must have the same length, "
            f"got {costs.shape[0]} and {trades.shape[0]}"
        )
    return float(np.sum(costs * trades))


def drift_breach_mask(
    current_weights: npt.ArrayLike,
    target_weights: npt.ArrayLike,
    config: ThresholdRebalancingConfig | None = None,
) -> npt.NDArray[np.bool_]:
    """Per-asset boolean mask of which positions breach the drift threshold.

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
    ndarray of bool, shape (n_assets,)
        ``True`` for each asset whose drift breaches the threshold.  Under a
        relative threshold, a zero-target position that still carries weight is
        always flagged (explicit exit intent).
    """
    if config is None:
        config = ThresholdRebalancingConfig()

    cur = _as_1d("current_weights", current_weights)
    tgt = _as_1d("target_weights", target_weights)
    _check_aligned(cur, tgt, "current_weights", "target_weights")

    drifts = np.abs(cur - tgt)

    if config.threshold_type == ThresholdType.ABSOLUTE:
        return drifts > config.threshold

    # Relative threshold: zero-target positions with non-zero current weight
    # always require rebalancing (explicit exit intent).
    exit_needed = (tgt == 0) & (cur > 0)
    safe_targets = np.where(tgt > 0, tgt, np.inf)
    relative_drifts = drifts / safe_targets
    return exit_needed | (relative_drifts > config.threshold)


def should_rebalance(
    current_weights: npt.ArrayLike,
    target_weights: npt.ArrayLike,
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
    return bool(np.any(drift_breach_mask(current_weights, target_weights, config)))


def apply_no_trade_band(
    current_weights: npt.ArrayLike,
    target_weights: npt.ArrayLike,
    config: ThresholdRebalancingConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Apply a no-trade band, trading only assets that breach the threshold.

    Turnover-reducing partial rebalance: positions whose drift breaches the
    threshold snap to their target weight; the rest keep their current
    (drifted) weight.  The resulting vector is renormalised to sum to 1 so it
    remains a valid fully-invested allocation.

    When no asset breaches the band the current weights are returned unchanged
    (renormalised), i.e. no trading occurs.

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
    ndarray, shape (n_assets,)
        Post-band weights, renormalised to sum to 1 (unless the total is zero
        or non-finite, in which case the un-normalised blend is returned).
    """
    cur = _as_1d("current_weights", current_weights)
    tgt = _as_1d("target_weights", target_weights)
    _check_aligned(cur, tgt, "current_weights", "target_weights")

    mask = drift_breach_mask(cur, tgt, config)
    blended = np.where(mask, tgt, cur)
    total = blended.sum()
    if not np.isfinite(total) or total == 0.0:
        return blended
    return blended / total


def should_rebalance_hybrid(
    current_weights: npt.NDArray[np.float64],
    target_weights: npt.NDArray[np.float64],
    config: HybridRebalancingConfig,
    current_date: pd.Timestamp,
    last_review_date: pd.Timestamp,
) -> tuple[bool, str]:
    """Determine whether to rebalance under a hybrid calendar+threshold policy.

    Returns ``(True, reason)`` only when **both** conditions are met:

    1. ``current_date`` is a calendar review date — at least
       ``config.calendar.trading_days`` business days have elapsed since
       ``last_review_date``.
    2. At least one asset's drift exceeds the threshold defined in
       ``config.threshold``.

    Between calendar review dates the function always returns
    ``(False, "between_review_dates")`` regardless of how much drift has
    accumulated.

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
    decision : bool
        ``True`` only if it is a calendar review date AND drift exceeds
        the threshold.
    reason : str
        One of ``"between_review_dates"``, ``"threshold_met"``,
        ``"threshold_not_met"`` — explains the decision branch taken.
    """
    next_review = last_review_date + pd.offsets.BDay(config.calendar.trading_days)
    if current_date < next_review:
        return False, "between_review_dates"
    if should_rebalance(current_weights, target_weights, config.threshold):
        return True, "threshold_met"
    return False, "threshold_not_met"


def build_rebalancing_walk_forward(
    config: CalendarRebalancingConfig | None = None,
    *,
    train_size: int = 12,
    test_size: int = 1,
    purged_size: int = 0,
    expand_train: bool = False,
    previous: bool = False,
    reduce_test: bool = False,
    freq_offset: str | pd.offsets.BaseOffset | None = None,
) -> WalkForward:
    """Build a calendar-driven :class:`skfolio.model_selection.WalkForward`.

    Turns a :class:`CalendarRebalancingConfig` into a skfolio walk-forward
    cross-validator whose test windows land on the real trading calendar
    (period-starts) matching the rebalancing cadence, rather than on a raw
    observation count.  ``test_size`` / ``train_size`` are counted in units of
    the calendar frequency (``config.pandas_freq``), e.g. quarters for a
    quarterly cadence.

    The returned CV feeds ``cross_val_predict`` / ``GridSearchCV`` and honours
    skfolio 1.0 calendar semantics (``freq``, ``freq_offset``, ``previous``,
    ``reduce_test``, ``expand_train``, ``purged_size``).  The input ``X`` must
    be a returns ``DataFrame`` with a ``DatetimeIndex``.

    Parameters
    ----------
    config : CalendarRebalancingConfig or None
        Rebalancing cadence supplying the calendar frequency.  Defaults to
        quarterly.
    train_size : int, default=12
        Number of ``freq`` periods in each training window (ignored beyond the
        first fold when ``expand_train`` is ``True``).
    test_size : int, default=1
        Number of ``freq`` periods in each test window (one rebalancing period
        by default).
    purged_size : int, default=0
        Observations purged between train and test to model execution latency.
        Use ``>= 1`` when execution is delayed relative to the signal.
    expand_train : bool, default=False
        Expanding (anchored) window instead of rolling.
    previous : bool, default=False
        If a period boundary is absent from the index, use the previous
        observation instead of the next.
    reduce_test : bool, default=False
        Keep the final partial test window instead of discarding it.
    freq_offset : str or pandas offset or None
        Optional offset applied to each period boundary (e.g. ``"2D"`` to
        rebalance two days after the period start).  Parsed with
        :func:`pandas.tseries.frequencies.to_offset` when a string.

    Returns
    -------
    skfolio.model_selection.WalkForward
        Configured walk-forward cross-validator.
    """
    from skfolio.model_selection import WalkForward

    if config is None:
        config = CalendarRebalancingConfig()

    offset = (
        pd.tseries.frequencies.to_offset(freq_offset)
        if isinstance(freq_offset, str)
        else freq_offset
    )

    return WalkForward(
        test_size=test_size,
        train_size=train_size,
        freq=config.pandas_freq,
        freq_offset=offset,
        previous=previous,
        expand_train=expand_train,
        reduce_test=reduce_test,
        purged_size=purged_size,
    )
