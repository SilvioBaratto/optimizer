"""Z-score entry and exit rules for a mean-reverting spread.

The textbook pairs rule trades a standardised spread against bands: open a trade
when the z-score is stretched, close it when it has come back near zero, and give
up if it keeps going. ``zscore_positions`` turns a z-score path into positions
using only the scores up to each date, so it can be backtested without look-ahead.

Research tools only: nothing in ``optimizer.statarb`` models transaction costs,
borrow costs or short-sale limits, and the fund mandate is long-only and
paper-only, so none of it is wired to execution.
"""

from __future__ import annotations

import math

import pandas as pd

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.statarb._inputs import as_float_array, require_finite


def _validate_levels(entry: float, exit: float, stop: float | None) -> None:
    entry = require_finite("entry", entry)
    exit = require_finite("exit", exit)
    if not 0.0 <= exit < entry:
        msg = f"exit must satisfy 0 <= exit < entry, got exit={exit}, entry={entry}"
        raise ConfigurationError(msg)
    if stop is not None and require_finite("stop", stop) <= entry:
        msg = f"stop must exceed entry, got stop={stop}, entry={entry}"
        raise ConfigurationError(msg)


def _next_state(
    position: int,
    locked: bool,
    zscore: float,
    entry: float,
    exit: float,
    stop: float | None,
) -> tuple[int, bool]:
    """Return the position and stop lock after one non-missing z-score."""
    if stop is not None and abs(zscore) > stop:
        return 0, True
    if abs(zscore) < exit:
        return 0, False
    if locked or position != 0:
        return position, locked
    if zscore > entry:
        return -1, False
    if zscore < -entry:
        return 1, False
    return 0, False


def zscore_positions(
    zscores: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = None,
) -> pd.Series:
    """Turn a z-score path into spread positions in ``{-1, 0, +1}``.

    The position is ``-1`` (short the spread) after ``z > entry``, ``+1`` (long the
    spread) after ``z < -entry``, and is held until ``|z| < exit``. With ``stop``
    set, ``|z| > stop`` closes the position and the series stays flat until
    ``|z| < exit`` again; a first reading beyond the stop is not entered either.
    Every comparison is strict. The position on a date is decided from the scores
    up to that date, it starts flat, and a NaN score keeps the previous position
    and any stop lock unchanged.

    A trade is closed before it is reversed: a score that jumps across zero
    without landing inside the exit band does not flip the position.

    This is a research tool. It ignores costs and borrow limits, and nothing here
    is wired to execution.

    Args:
        zscores: Standardised spread, for example ``KalmanResult.zscore``.
        entry: Absolute z-score that opens a position, positive.
        exit: Absolute z-score below which a position closes; ``0 <= exit <
            entry``. Zero never closes a position except by the stop.
        stop: Absolute z-score above which a position is stopped out; must exceed
            ``entry`` when given.

    Returns:
        Integer positions on the index of ``zscores``.

    Raises:
        DataError: If ``zscores`` is not a pandas Series of numbers. Bad data is a
            ``DataError`` and bad settings are a ``ConfigurationError``.
        ConfigurationError: If a level is not finite, ``exit`` is outside
            ``[0, entry)`` or ``stop`` is not above ``entry``.
    """
    if not isinstance(zscores, pd.Series):
        msg = "zscores must be a pandas Series"
        raise DataError(msg)
    _validate_levels(entry, exit, stop)
    positions = []
    position, locked = 0, False
    for zscore in as_float_array(zscores, "zscores"):
        if not math.isnan(zscore):
            position, locked = _next_state(
                position, locked, float(zscore), entry, exit, stop
            )
        positions.append(position)
    return pd.Series(positions, index=zscores.index, name="position", dtype="int64")
