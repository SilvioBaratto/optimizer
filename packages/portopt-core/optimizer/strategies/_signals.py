"""Moving-average and momentum signals.

Every signal here is a pure function of a price table (a ``DataFrame`` with one
column per asset and a sorted date index) and is decided from the close of each
date using data up to and including that date. None of them looks ahead.
Whether a signal can be traded on that same close is a question for the
backtest: ``backtest_weights`` holds a weight decided on a date only from the
following period, by default.

* The moving-average crossover goes long when a fast average of the price is
  above a slow one, as in the classic trend-following rule.
* Momentum scores are the total return over a formation window that leaves out
  the most recent rows. The 12-1 convention (twelve months back, skipping the last
  month) follows Carhart (1997), who lags the return by a month, and the short-term
  reversal of Jegadeesh (1990) that the skip avoids.
* Cross-sectional momentum ranks the assets on those scores and holds the best
  fraction (and, for a dollar-neutral book, shorts the worst), the winners-minus-
  losers construction of Jegadeesh and Titman (1993).
"""

from __future__ import annotations

import math
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.strategies._config import MovingAverageKind

# A fraction that should land exactly on an integer (0.29 of 100 assets) can sit one
# unit in the last place below it (28.999999999999996), and floor would drop a name.
_FRACTION_TOLERANCE: Final = 1e-9
# Share of the ranked assets a long-short book may hold in each leg before the legs
# would have to share an asset.
_MAX_LONG_SHORT_FRACTION: Final = 0.5
# A moving-average window, a formation window or a rebalance interval of one row is the
# shortest that means anything.
_MIN_WINDOW: Final = 1


def to_float_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return ``frame`` as float64, rejecting anything that is not a numeric table.

    Args:
        frame: The table to convert.
        name: Argument name used in error messages.

    Returns:
        A float64 copy with the same index and columns.

    Raises:
        DataError: If ``frame`` is not a ``DataFrame`` or has a non-numeric column.
    """
    if not isinstance(frame, pd.DataFrame):
        msg = (
            f"{name} must be a DataFrame with one column per asset, "
            f"got {type(frame).__name__}"
        )
        raise DataError(msg)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        msg = f"{name} must be numeric in every column"
        raise DataError(msg)
    return frame.astype(np.float64)


def _require_shape(prices: pd.DataFrame, min_rows: int) -> None:
    """Reject empty tables and indexes or columns that are not usable labels."""
    if prices.shape[1] == 0 or len(prices) < min_rows:
        msg = (
            f"prices needs at least {min_rows} rows and one column, got {prices.shape}"
        )
        raise DataError(msg)
    if not prices.columns.is_unique:
        msg = "prices has duplicate column labels"
        raise DataError(msg)
    if not (prices.index.is_unique and prices.index.is_monotonic_increasing):
        msg = "the prices index must be unique and sorted in increasing order"
        raise DataError(msg)


def require_price_frame(prices: pd.DataFrame, *, min_rows: int = 1) -> pd.DataFrame:
    """Validate a price table and return it as float64.

    Missing prices (NaN) are allowed because an asset may be listed late or have
    gaps; callers decide what a gap means for them. Zero, negative and infinite
    prices are never valid.

    Args:
        prices: One column of prices per asset, indexed by increasing dates.
        min_rows: Fewest rows the table may have.

    Returns:
        The prices as a float64 table.

    Raises:
        DataError: If the table is not a numeric ``DataFrame``, is too small, has
            a duplicate or unsorted index or duplicate columns, or holds a price
            that is infinite or not strictly positive.
    """
    frame = to_float_frame(prices, "prices")
    _require_shape(frame, min_rows)
    values = frame.to_numpy()
    if np.isinf(values).any() or (values[~np.isnan(values)] <= 0.0).any():
        msg = "prices must be finite and strictly positive wherever they are present"
        raise DataError(msg)
    return frame


def _require_window(value: int, name: str) -> None:
    """Reject a window shorter than one row."""
    if value < _MIN_WINDOW:
        msg = f"{name} must be at least {_MIN_WINDOW}, got {value}"
        raise ConfigurationError(msg)


def _as_kind(kind: MovingAverageKind | str) -> MovingAverageKind:
    """Return ``kind`` as a member of ``MovingAverageKind``."""
    try:
        return MovingAverageKind(kind)
    except ValueError as error:
        allowed = [member.value for member in MovingAverageKind]
        msg = f"kind must be one of {allowed}, got {kind!r}"
        raise ConfigurationError(msg) from error


def moving_average(
    prices: pd.DataFrame,
    window: int,
    kind: MovingAverageKind = MovingAverageKind.SMA,
) -> pd.DataFrame:
    """Return the moving average of each asset's price.

    The simple average is the mean of the last ``window`` prices. The exponential
    average has span ``window`` (a smoothing weight of ``2 / (window + 1)``) and
    starts from the first price (``adjust=False``). Both are blank (NaN) for the
    first ``window - 1`` observations of an asset, so the two kinds have the same
    warm-up. A simple average is blank while its window contains a gap in the
    prices; an exponential average carries its last value over a gap.

    Args:
        prices: One column of prices per asset, indexed by increasing dates.
        window: Window length in rows (the span for the exponential kind), at
            least 1.
        kind: Simple or exponential weighting; a string with the member's value
            is accepted.

    Returns:
        A table of the same shape as ``prices``.

    Raises:
        ConfigurationError: If ``window`` is below 1 or ``kind`` is unknown.
        DataError: If ``prices`` is not a valid price table.
    """
    _require_window(window, "window")
    averaging = _as_kind(kind)
    frame = require_price_frame(prices)
    if averaging is MovingAverageKind.EMA:
        return cast(
            "pd.DataFrame",
            frame.ewm(span=window, adjust=False, min_periods=window).mean(),
        )
    return cast("pd.DataFrame", frame.rolling(window).mean())


def moving_average_crossover_positions(
    prices: pd.DataFrame,
    fast: int,
    slow: int,
    kind: MovingAverageKind = MovingAverageKind.SMA,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Return the trend position implied by a fast and a slow moving average.

    The position of a date is decided from the close of that date: ``+1`` when
    the fast average is above the slow one, ``-1`` when it is below and
    ``allow_short`` is set, and ``0`` otherwise. Dates before the slow average
    exists, and dates where the averages are equal, are flat. The table has no
    gaps (NaN) and each asset is treated on its own; scale it into weights, for
    example by dividing by the number of assets, before backtesting.

    Args:
        prices: One column of prices per asset, indexed by increasing dates.
        fast: Window of the fast average, at least 1.
        slow: Window of the slow average, longer than ``fast``.
        kind: Simple or exponential averages.
        allow_short: Go short instead of flat when the fast average is below.

    Returns:
        A float table of positions in ``{-1, 0, 1}`` shaped like ``prices``.

    Raises:
        ConfigurationError: If ``fast`` is below 1 or not smaller than ``slow``.
        DataError: If ``prices`` is not a valid price table.
    """
    _require_window(fast, "fast")
    if fast >= slow:
        msg = f"fast must be smaller than slow, got fast={fast} and slow={slow}"
        raise ConfigurationError(msg)
    fast_average = moving_average(prices, fast, kind)
    slow_average = moving_average(prices, slow, kind)
    # A comparison with NaN is false, which leaves the rows before the slow average
    # exists flat without a separate mask.
    positions = (fast_average > slow_average).astype(np.float64)
    if allow_short:
        positions = positions - (fast_average < slow_average).astype(np.float64)
    return positions


def _require_momentum_windows(lookback: int, skip: int) -> None:
    """Reject formation windows that are empty or inverted."""
    if lookback < _MIN_WINDOW:
        msg = f"lookback must be at least {_MIN_WINDOW}, got {lookback}"
        raise ConfigurationError(msg)
    if skip < 0:
        msg = f"skip must not be negative, got {skip}"
        raise ConfigurationError(msg)
    if skip >= lookback:
        msg = (
            f"skip must be smaller than lookback, got skip={skip}, lookback={lookback}"
        )
        raise ConfigurationError(msg)


def momentum_scores(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> pd.DataFrame:
    """Return the total return from ``t - lookback`` to ``t - skip`` for each date.

    With the defaults this is the classic 12-1 momentum: the return over the last
    twelve months leaving out the most recent one. The score of a date uses
    prices up to that date only.

    Args:
        prices: One column of prices per asset, indexed by increasing dates.
        lookback: How many rows back the formation window starts, at least 1.
        skip: How many of the most recent rows are left out, from 0 to
            ``lookback - 1``.

    Returns:
        A table shaped like ``prices`` holding ``price[t - skip] /
        price[t - lookback] - 1``; NaN until both prices exist.

    Raises:
        ConfigurationError: If ``lookback`` or ``skip`` is out of range.
        DataError: If ``prices`` is not a valid price table.
    """
    _require_momentum_windows(lookback, skip)
    frame = require_price_frame(prices)
    return frame.shift(skip) / frame.shift(lookback) - 1.0


def _require_selection(
    top_fraction: float, rebalance_every: int, long_only: bool
) -> None:
    """Reject leg sizes and rebalance intervals that cannot form a book."""
    if not 0.0 < top_fraction <= 1.0:
        msg = f"top_fraction must be in (0, 1], got {top_fraction}"
        raise ConfigurationError(msg)
    if not long_only and top_fraction > _MAX_LONG_SHORT_FRACTION:
        msg = (
            f"top_fraction must be at most {_MAX_LONG_SHORT_FRACTION} for a "
            f"long-short book, because the two legs would overlap, got {top_fraction}"
        )
        raise ConfigurationError(msg)
    if rebalance_every < _MIN_WINDOW:
        msg = f"rebalance_every must be at least {_MIN_WINDOW}, got {rebalance_every}"
        raise ConfigurationError(msg)


def _rebalance_rows(
    scores: npt.NDArray[np.float64], every: int
) -> npt.NDArray[np.intp]:
    """Return the rows to rebalance on: the first scored row, then every ``every``."""
    has_score = ~np.isnan(scores).all(axis=1)
    if not has_score.any():
        return np.empty(0, dtype=np.intp)
    return np.arange(int(np.argmax(has_score)), scores.shape[0], every)


def _leg_size(n_ranked: int, top_fraction: float, long_only: bool) -> int:
    """Return how many names each leg holds, at least one when two can be formed."""
    size = max(1, math.floor(top_fraction * n_ranked + _FRACTION_TOLERANCE))
    if long_only:
        return min(size, n_ranked)
    return min(size, n_ranked // 2)


def _target_row(
    scores: npt.NDArray[np.float64], top_fraction: float, long_only: bool
) -> npt.NDArray[np.float64]:
    """Return the equal-weight target for one date's cross-section of scores."""
    ranked = np.flatnonzero(~np.isnan(scores))
    leg = _leg_size(ranked.size, top_fraction, long_only)
    target = np.zeros(scores.size)
    if leg == 0:
        return target
    # Best score first, ties by column order, so the ranking is reproducible.
    order = ranked[np.argsort(-scores[ranked], kind="stable")]
    target[order[:leg]] = 1.0 / leg
    if not long_only:
        target[order[-leg:]] = -1.0 / leg
    return target


def cross_sectional_momentum_weights(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_fraction: float = 0.2,
    rebalance_every: int = 21,
    long_only: bool = True,
) -> pd.DataFrame:
    """Return target weights that hold the strongest-momentum assets.

    On each rebalance date the assets that have a momentum score are ranked and
    the top ``top_fraction`` of them are held at equal weight, summing to 1. If
    ``long_only`` is false the bottom ``top_fraction`` is also sold short at
    equal weight, summing to -1, so the book is dollar neutral. The targets are
    held constant until the next rebalance date, and each is decided from data
    up to and including its date. The first rebalance date is the first date with
    a score and the rest follow every ``rebalance_every`` rows.

    A leg holds ``floor(top_fraction * n)`` names of the ``n`` ranked assets, and
    always at least one. Ties in score are broken by column order. If no asset is
    ranked, or a long-short book has fewer than two, the date is flat.

    Args:
        prices: One column of prices per asset, indexed by increasing dates.
        lookback: Formation window start, in rows (see ``momentum_scores``).
        skip: Most recent rows left out of the formation window.
        top_fraction: Share of the ranked assets in each leg, in ``(0, 1]``, and at
            most 0.5 for a long-short book.
        rebalance_every: Rows between rebalances, at least 1.
        long_only: Hold only the top leg, or also short the bottom leg.

    Returns:
        A float table of target weights shaped like ``prices``, with no gaps:
        rows before the first rebalance are zero.

    Raises:
        ConfigurationError: If a window, the fraction or the interval is invalid.
        DataError: If ``prices`` is not a valid price table.
    """
    _require_selection(top_fraction, rebalance_every, long_only)
    scores = momentum_scores(prices, lookback, skip)
    values = scores.to_numpy()
    targets = np.zeros(values.shape)
    starts = _rebalance_rows(values, rebalance_every)
    stops = np.append(starts[1:], values.shape[0])[: starts.size]
    for start, stop in zip(starts, stops, strict=True):
        targets[start:stop] = _target_row(values[start], top_fraction, long_only)
    return pd.DataFrame(targets, index=scores.index, columns=scores.columns)
