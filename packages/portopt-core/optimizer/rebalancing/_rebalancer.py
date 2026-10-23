"""Rebalancing decision logic."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.model_selection import WalkForward
from skfolio.typing import IntArray

from optimizer.rebalancing._config import (
    PANDAS_FREQ,
    CalendarRebalancingConfig,
    HybridRebalancingConfig,
    RebalancingFrequency,
    ThresholdRebalancingConfig,
    ThresholdType,
)

logger = logging.getLogger(__name__)

# Convex solvers return 1e-12..1e-6 where the true weight is zero, and up to a
# few 1e-5 on wide books (skfolio MeanRisk maximising the ratio over 30 assets
# was measured at ~5e-6). One basis point clears that with margin and is far
# below any position worth a trade. Below this magnitude a weight is noise, so
# a relative band must neither measure it against itself nor treat it as a
# position that needs exiting.
_WEIGHT_DUST = 1e-4

_SEMIANNUAL_FREQ = PANDAS_FREQ[RebalancingFrequency.SEMIANNUAL]
_SEMIANNUAL_MONTHS = (1, 7)


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

    Args:
        weights: Current portfolio weights (must sum to 1), shape (n_assets,).
        returns: Single-period asset returns, shape (n_assets,).

    Returns:
        Drifted weights after applying returns, shape (n_assets,). If the
        post-drift portfolio value is zero or non-finite the raw grown weights
        are returned unnormalised.
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

    Args:
        current_weights: Current portfolio weights, shape (n_assets,).
        target_weights: Target portfolio weights, shape (n_assets,).

    Returns:
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

    Args:
        current_weights: Current portfolio weights, shape (n_assets,).
        target_weights: Target portfolio weights, shape (n_assets,).
        transaction_costs: Per-unit transaction cost; scalar for uniform costs
            or array for asset-specific costs.

    Returns:
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
    """Return a per-asset boolean mask of which positions breach the drift threshold.

    Args:
        current_weights: Current (drifted) portfolio weights, shape (n_assets,).
        target_weights: Target portfolio weights from the optimiser, shape
            (n_assets,).
        config: Threshold configuration. Defaults to absolute 5pp threshold.

    Returns:
        Boolean array, shape (n_assets,), with ``True`` for each asset whose
        drift breaches the threshold. Under a relative threshold, a
        (near-)zero-target position that still carries a material weight is
        always flagged (explicit exit intent); weights below one basis point
        (``1e-4``) in magnitude are treated as solver noise, so dust targets
        and dust holdings never trigger a breach on their own. Short (negative) target
        weights are measured against the magnitude of the target so a short
        book is treated symmetrically with a long book.
    """
    if config is None:
        config = ThresholdRebalancingConfig()

    cur = _as_1d("current_weights", current_weights)
    tgt = _as_1d("target_weights", target_weights)
    _check_aligned(cur, tgt, "current_weights", "target_weights")

    drifts = np.abs(cur - tgt)

    if config.threshold_type == ThresholdType.ABSOLUTE:
        return drifts > config.threshold

    # Relative threshold: any zero-target position that still carries material
    # weight (long *or* short) always requires rebalancing (explicit exit
    # intent). Material targets use their absolute magnitude as the
    # denominator so negative (short) targets are measured symmetrically with
    # positive ones; dust targets get an infinite denominator and never breach.
    tgt_magnitude = np.abs(tgt)
    material_target = tgt_magnitude >= _WEIGHT_DUST
    exit_needed = ~material_target & (np.abs(cur) >= _WEIGHT_DUST)
    safe_targets = np.where(material_target, tgt_magnitude, np.inf)
    relative_drifts = drifts / safe_targets
    return exit_needed | (relative_drifts > config.threshold)


def should_rebalance(
    current_weights: npt.ArrayLike,
    target_weights: npt.ArrayLike,
    config: ThresholdRebalancingConfig | None = None,
) -> bool:
    """Determine whether any asset breaches the drift threshold.

    Args:
        current_weights: Current (drifted) portfolio weights, shape (n_assets,).
        target_weights: Target portfolio weights from the optimiser, shape
            (n_assets,).
        config: Threshold configuration. Defaults to absolute 5pp threshold.

    Returns:
        ``True`` if at least one asset breaches the threshold.
    """
    return bool(np.any(drift_breach_mask(current_weights, target_weights, config)))


# Largest factor by which the breached targets may be rescaled to fill the
# budget the held weights leave. Beyond it the targets carry too little (or too
# much) mass to stand in for that budget: scaling them would re-inflate a
# position the optimizer all but exited, so the whole vector is renormalised.
_MAX_BREACH_RESCALE = 2.0


def apply_no_trade_band(
    current_weights: npt.ArrayLike,
    target_weights: npt.ArrayLike,
    config: ThresholdRebalancingConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Apply a no-trade band, trading only assets that breach the threshold.

    Turnover-reducing partial rebalance: assets inside the band keep their
    current (drifted) weight exactly, and the assets that breach the band
    move to their target weights, rescaled pro rata so the whole vector sums
    to 1 (the breached targets share ``1 - sum(held weights)``). Held
    positions are never touched by the rescale, so no in-band asset is traded.

    When the breached targets cannot stand in for the budget the held weights
    leave (their sum is not positive, that budget is not positive, or filling
    it would rescale them by more than a factor of two), the whole blend is
    renormalised instead, so the held positions absorb the difference and the
    vector still sums to 1. This covers the exit-only case where every
    breached target is zero.

    When no asset breaches the band the current weights are returned
    (renormalised to sum to 1), i.e. no trading occurs.

    Args:
        current_weights: Current (drifted) portfolio weights, shape (n_assets,).
        target_weights: Target portfolio weights from the optimiser, shape
            (n_assets,).
        config: Threshold configuration. Defaults to absolute 5pp threshold.

    Returns:
        Post-band weights summing to 1, shape (n_assets,). If the weights that
        would be renormalised sum to zero or a non-finite value, the plain
        blend (breached targets, held current weights) is returned
        un-normalised.
    """
    cur = _as_1d("current_weights", current_weights)
    tgt = _as_1d("target_weights", target_weights)
    _check_aligned(cur, tgt, "current_weights", "target_weights")

    mask = drift_breach_mask(cur, tgt, config)
    if not mask.any():
        total = cur.sum()
        if not np.isfinite(total) or total == 0.0:
            return cur
        return cur / total

    blended = np.where(mask, tgt, cur)
    traded_total = tgt[mask].sum()
    budget = 1.0 - cur[~mask].sum()
    if traded_total > 0.0 and budget > 0.0:
        scale = budget / traded_total
        if 1.0 / _MAX_BREACH_RESCALE <= scale <= _MAX_BREACH_RESCALE:
            blended[mask] = tgt[mask] * scale
            return blended

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

    Args:
        current_weights: Current (drifted) portfolio weights, shape (n_assets,).
        target_weights: Target portfolio weights from the optimiser, shape
            (n_assets,).
        config: Hybrid configuration combining calendar and threshold rules.
        current_date: The date being evaluated.
        last_review_date: Date of the last calendar review.

    Returns:
        A ``(decision, reason)`` tuple where ``decision`` is ``True`` only if
        it is a calendar review date AND drift exceeds the threshold, and
        ``reason`` is one of ``"between_review_dates"``, ``"threshold_met"``,
        or ``"threshold_not_met"``.
    """
    next_review = last_review_date + pd.offsets.BDay(config.calendar.trading_days)
    if current_date < next_review:
        return False, "between_review_dates"
    if should_rebalance(current_weights, target_weights, config.threshold):
        return True, "threshold_met"
    return False, "threshold_not_met"


class _CalendarWalkForward(WalkForward):
    """`WalkForward` with the calendar semantics skfolio's period grid lacks.

    * A period boundary that has no observation to land on (a positive
      ``freq_offset`` past the final row with ``previous=False``, or a negative
      one before the first row with ``previous=True``) is resolved by skfolio
      to index -1. That index then starts a fold that spans the whole sample or
      ends a zero-length test window, which crashes ``cross_val_predict`` and
      is still counted by ``get_n_splits``. Such boundaries are dropped from
      the grid, whatever the cadence and ``reduce_test`` setting.
    * A multiple-period alias such as ``"2QS"`` steps from the quarter of the
      first observation, so one semiannual cadence would land on Jan/Jul or
      Apr/Oct depending on where the history starts. The semiannual grid is
      built from the January and July period starts instead.

    Only integer ``train_size`` windows are walked here; a date-offset
    ``train_size`` or an observation-count walk (``freq=None``) is delegated to
    skfolio and only stripped of folds whose test window is empty or starts
    before the first observation.
    """

    def split(
        self,
        X: npt.ArrayLike,
        y: None = None,
        groups: None = None,
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Yield the folds of the calendar grid, skipping unresolvable ones."""
        if self.freq is not None and isinstance(self.train_size, int | np.integer):
            folds = self._split_calendar(X, y, groups)
        else:
            folds = super().split(X, y, groups)
        for train_idx, test_idx in folds:
            # Test windows are ascending aranges, so a leading -1 marks a
            # boundary skfolio could not resolve to an observation.
            if len(test_idx) > 0 and test_idx[0] >= 0:
                yield train_idx, test_idx

    def get_n_splits(
        self,
        X: npt.ArrayLike | None = None,
        y: None = None,
        groups: None = None,
    ) -> int:
        """Return the number of folds `split` yields for *X*."""
        n_splits = super().get_n_splits(X, y, groups)
        if X is None or self.freq is None:
            return n_splits
        return sum(1 for _ in self.split(X))

    def _period_positions(self, ts_index: pd.DatetimeIndex) -> IntArray:
        """Return the observation index of each period boundary inside the sample."""
        start = ts_index[0]
        if self.freq_offset is not None:
            start = min(start, start - self.freq_offset)
        if self.freq == _SEMIANNUAL_FREQ:
            boundaries = pd.date_range(start=start, end=ts_index[-1], freq="QS")
            boundaries = boundaries[boundaries.month.isin(_SEMIANNUAL_MONTHS)]
        else:
            boundaries = pd.date_range(start=start, end=ts_index[-1], freq=self.freq)
        if self.freq_offset is not None:
            boundaries = boundaries + self.freq_offset
        positions = ts_index.get_indexer(
            boundaries, method="ffill" if self.previous else "bfill"
        )
        return positions[positions >= 0]

    def _split_calendar(
        self,
        X: npt.ArrayLike,
        y: None,
        groups: None,
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Walk forward over the period grid with an integer ``train_size``.

        Mirrors skfolio's period-based loop on a grid that keeps only the
        boundaries it can resolve to an observation.
        """
        # Reuses skfolio's validation of the window sizes and the datetime index.
        super().get_n_splits(X, y, groups)
        ts_index = cast("pd.DataFrame", X).index
        n_samples = len(ts_index)
        train_size = int(cast("int", self.train_size))
        test_size = int(self.test_size)

        positions = self._period_positions(ts_index)
        n = len(positions)
        i = 0
        while i + train_size < n:
            test_start = positions[i + train_size]
            if i + train_size + test_size >= n:
                if not self.reduce_test:
                    return
                test_idx = np.arange(test_start, n_samples)
            else:
                test_idx = np.arange(test_start, positions[i + train_size + test_size])
            train_start = 0 if self.expand_train else positions[i]
            yield np.arange(train_start, test_start - self.purged_size), test_idx
            i += test_size


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
    """Build a calendar-driven `WalkForward`.

    Turns a `CalendarRebalancingConfig` into a skfolio walk-forward
    cross-validator whose test windows land on the real trading calendar
    (period-starts) matching the rebalancing cadence, rather than on a raw
    observation count. ``test_size`` / ``train_size`` are counted in units of
    the calendar frequency (``config.pandas_freq``), e.g. quarters for a
    quarterly cadence.

    The returned CV feeds ``cross_val_predict`` / ``GridSearchCV`` and honours
    skfolio 1.0 calendar semantics. The input ``X`` must be a returns
    ``DataFrame`` with a ``DatetimeIndex``.

    Args:
        config: Rebalancing cadence supplying the calendar frequency. Defaults
            to quarterly.
        train_size: Number of ``freq`` periods in each training window. Ignored
            beyond the first fold when ``expand_train`` is ``True``.
        test_size: Number of ``freq`` periods in each test window (one
            rebalancing period by default).
        purged_size: Observations purged between train and test to model
            execution latency. Use ``>= 1`` when execution is delayed relative
            to the signal.
        expand_train: Expanding (anchored) window instead of rolling.
        previous: If a period boundary is absent from the index, use the
            previous observation instead of the next.
        reduce_test: Keep the final partial test window instead of discarding it.
        freq_offset: Optional offset applied to each period boundary (e.g.
            ``"2D"`` to rebalance two days after the period start). Parsed with
            `to_offset` when a string.

    Returns:
        Configured walk-forward cross-validator. A fold whose shifted test
        boundary lies beyond the last observation (possible with
        ``freq_offset``) is dropped rather than emitted with an empty test
        window.
    """
    if config is None:
        config = CalendarRebalancingConfig()

    offset = (
        pd.tseries.frequencies.to_offset(freq_offset)
        if isinstance(freq_offset, str)
        else freq_offset
    )

    return _CalendarWalkForward(
        test_size=test_size,
        train_size=train_size,
        freq=config.pandas_freq,
        freq_offset=offset,
        previous=previous,
        expand_train=expand_train,
        reduce_test=reduce_test,
        purged_size=purged_size,
    )
