"""Bars per calendar year: the one annualisation rule every fund figure shares.

Kept free of the optimizer and the agent stack so the Performance strip (read by the
TUI through ``fund.observe``) and the weighting tools scale returns the same way
without either importing the other.
"""

from __future__ import annotations

from typing import Any

# Fallback bars per year, used only when an index carries no calendar to derive from
# (see ``periods_per_year``).
DEFAULT_PERIODS_PER_YEAR = 252

# The band a derived bars-per-year is held to. A daily panel sits between a single
# exchange calendar (about 250) and a seven-day series (365); anything outside it
# means the index is not daily, and a clamped figure beats an annualisation that is
# off by a factor of five.
MIN_PERIODS_PER_YEAR = 200
MAX_PERIODS_PER_YEAR = 366

_DAYS_PER_YEAR = 365.25


def periods_per_year(index: Any) -> int:
    """Derive the bars a calendar year holds from a return index.

    The one source of the annualisation factor for every ``fund`` tool, so the
    moments the PM reads, the optimizer's scaling, the risk-free rate and the backtest
    metrics are always scaled alike. A union calendar across venues has about 260
    bars a year (a holiday on one venue is a forward-filled zero return on another),
    and a single exchange about 252, so a fixed 252 misstates both.

    Args:
        index: The return frame's index. Only a ``DatetimeIndex`` with at least two
            distinct dates carries a calendar to derive from.

    Returns:
        The number of intervals per calendar year of the index, rounded and held to
        ``[MIN_PERIODS_PER_YEAR, MAX_PERIODS_PER_YEAR]``; ``DEFAULT_PERIODS_PER_YEAR``
        when the index has no calendar. A very short sample is noisy but stays in the
        band.
    """
    import pandas as pd

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return DEFAULT_PERIODS_PER_YEAR
    span_days = (index.max() - index.min()).total_seconds() / 86400.0
    if not span_days > 0.0:
        return DEFAULT_PERIODS_PER_YEAR
    per_year = (len(index) - 1) / (span_days / _DAYS_PER_YEAR)
    return int(min(max(round(per_year), MIN_PERIODS_PER_YEAR), MAX_PERIODS_PER_YEAR))


__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "MAX_PERIODS_PER_YEAR",
    "MIN_PERIODS_PER_YEAR",
    "periods_per_year",
]
