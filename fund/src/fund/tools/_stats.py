"""Bounded per-series statistics for the read-only data tools.

``get_prices`` and ``get_macro_series`` must hand the economist numbers it can
interpret (a level, how far it moved, how volatile and how far below its peak it
is) without ever returning the raw matrix. Each series collapses to a fixed handful
of scalars computed here in-environment, so the size of a tool reply is independent
of the history length and every figure is reproducible from the same DB state.

Every statistic degrades to ``None`` when the series is too short to define it:
JSON has no NaN, and a fabricated zero would read as a real measurement.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd

# Trailing window for the risk statistics: about one trading year, so volatility and
# drawdown describe the current regime rather than a decade-old crisis.
_TRADING_DAYS = 252

_HORIZONS: tuple[tuple[str, pd.DateOffset], ...] = (
    ("change_1m", pd.DateOffset(months=1)),
    ("change_3m", pd.DateOffset(months=3)),
    ("change_12m", pd.DateOffset(years=1)),
)

# Significant digits, not decimals: a sub-cent price and a billion-share volume both
# keep their precision while the reply stays short.
_SIG_DIGITS = 8


def _scalar(value: float) -> float | None:
    """The value rounded for the model, or ``None`` when it is NaN or infinite."""
    number = float(value)
    return float(f"{number:.{_SIG_DIGITS}g}") if math.isfinite(number) else None


def _last_day(series: pd.Series) -> pd.Timestamp:
    """The series' last index label; the frames the tools build are date-indexed."""
    return cast("pd.Timestamp", series.index[-1])


def _level_at_or_before(series: pd.Series, when: pd.Timestamp) -> float | None:
    prior = series.loc[:when]
    return float(prior.iloc[-1]) if not prior.empty else None


def _horizon_changes(series: pd.Series, *, relative: bool) -> dict[str, float | None]:
    """Change of the last observation versus each calendar horizon behind it.

    Horizons are calendar offsets from the series' *own* last date and take the
    latest observation at or before the target, so a monthly series is compared with
    its previous prints rather than with a day that has no observation.
    """
    last_day = _last_day(series)
    last = float(series.iloc[-1])
    changes: dict[str, float | None] = {}
    for key, offset in _HORIZONS:
        base = _level_at_or_before(series, last_day - offset)
        if base is None or (relative and base == 0.0):
            changes[key] = None
        else:
            changes[key] = _scalar(last / base - 1.0 if relative else last - base)
    return changes


def _head(series: pd.Series, *, relative: bool) -> dict[str, Any]:
    return {
        "last": _scalar(float(series.iloc[-1])),
        "last_date": str(_last_day(series).date()),
        **_horizon_changes(series, relative=relative),
    }


def _risk(series: pd.Series) -> dict[str, float | None]:
    window = series.tail(_TRADING_DAYS + 1)
    returns = window.pct_change().dropna()
    vol = (
        float(np.std(returns.to_numpy(dtype=float), ddof=1)) * math.sqrt(_TRADING_DAYS)
        if len(returns) > 1
        else None
    )
    # Magnitudes, not signed returns: ``max_drawdown`` is a non-negative loss
    # everywhere else in ``fund`` (skfolio's ``Portfolio.max_drawdown``,
    # ``PerformanceSummary``), and one key must not flip sign between tools.
    below_peak = 1.0 - window / window.cummax()
    return {
        "realised_vol": None if vol is None else _scalar(vol),
        "drawdown": _scalar(float(below_peak.iloc[-1])),
        "max_drawdown": _scalar(float(below_peak.max())),
    }


def _zscore(series: pd.Series) -> float | None:
    if len(series) < 2:
        return None
    levels = series.to_numpy(dtype=float)
    spread = float(np.std(levels, ddof=1))
    if not math.isfinite(spread) or spread == 0.0:
        return None
    return _scalar((float(levels[-1]) - float(np.mean(levels))) / spread)


def price_stats(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarise each price column as a level, relative changes, volatility, drawdown.

    Args:
        frame: Wide price frame (sorted ``DatetimeIndex``, one column per ticker).
            Columns may carry NaN where a ticker has no bar; each is judged on its
            own observations.

    Returns:
        ``{ticker: stats}`` for every column with at least one observation. ``stats``
        holds ``last``, ``last_date``, simple-return ``change_1m``/``change_3m``/
        ``change_12m``, and over the trailing 252 observations the annualised
        ``realised_vol``, the current ``drawdown`` below the running peak and the
        ``max_drawdown`` — fractions, with both drawdowns non-negative (``0.2`` is
        20% below the peak), ``None`` where the history is too short.
    """
    stats: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        stats[str(column)] = {**_head(series, relative=True), **_risk(series)}
    return stats


def level_stats(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarise each series as a level, absolute changes and a history z-score.

    Suited to macro indicators, whose levels can be negative or near zero (a yield
    spread) so a percentage change would be meaningless, and to non-price columns.

    Args:
        frame: Wide frame (sorted ``DatetimeIndex``, one column per series). Series of
            different frequency share the union index, so each column is judged on
            its own observations only.

    Returns:
        ``{series: stats}`` for every column with at least one observation. ``stats``
        holds ``last``, ``last_date``, the level differences ``change_1m``/
        ``change_3m``/``change_12m`` and ``zscore`` of the last level against the
        series' full history to the as-of date — ``None`` where it is undefined.
    """
    stats: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        stats[str(column)] = {
            **_head(series, relative=False),
            "zscore": _zscore(series),
        }
    return stats


__all__ = ["level_stats", "price_stats"]
