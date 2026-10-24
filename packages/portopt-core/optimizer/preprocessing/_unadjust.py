"""Recover as-traded closes from dividend-adjusted closes.

The ingestion layer stores yfinance ``auto_adjust=True`` closes: every
ex-dividend date rescales *all earlier* closes by ``1 - D / C_prev``.  Ratios
of adjusted closes are therefore total returns and are safe for momentum,
volatility and beta, but the price *level* at an old date depends on dividends
paid after it.  Quantities that need the price actually paid per share, such as
dollar volume (Amihud illiquidity, ADDV screens), must start from the
as-traded close, which `unadjust_close` rebuilds from the stored dividends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer.preprocessing._coerce import _coerce_numeric

__all__ = ["unadjust_close"]


def unadjust_close(
    adjusted_close: pd.DataFrame, dividends: pd.DataFrame
) -> pd.DataFrame:
    """Undo the dividend adjustment baked into ``auto_adjust=True`` closes.

    An ex-dividend date ``d`` with cash dividend ``D_d`` multiplies every
    earlier adjusted close by ``1 - D_d / C_{d-1}``, where ``C_{d-1}`` is the
    as-traded close of the previous trading day.  Working backwards from the
    latest date (where adjusted and as-traded closes coincide) gives the
    closed form ``C_t = A_t / F_{t+1} + D_{t+1}`` with ``F`` the cumulative
    adjustment factor, so no iteration over ex-dates is needed.

    Splits need no handling: yfinance volume is split-adjusted like the close,
    so price x volume is invariant to them.

    Args:
        adjusted_close: Dividend-adjusted close matrix (dates x tickers), as
            stored by ingestion.  Missing closes stay missing; an ex-date
            whose previous day has no close uses the last available close.
            ``Decimal`` columns read back from the database are accepted.
        dividends: Cash dividend per share (dates x tickers) in the same price
            unit as the closes.  Absent dates, absent tickers and null cells
            all count as "no dividend".

    Returns:
        As-traded close matrix with the same index and columns as
        ``adjusted_close``.

    Raises:
        DataError: If ``adjusted_close`` or ``dividends`` holds values that
            cannot be cast to ``float64``.
    """
    close = _coerce_numeric(adjusted_close, "unadjust_close")
    div = (
        _coerce_numeric(dividends, "unadjust_close")
        .reindex(index=close.index, columns=close.columns)
        .fillna(0.0)
    )

    # Forward-filled so an ex-date that follows a missing close still sees the
    # previous available close, which is what the adjustment was built from.
    # A non-positive close would zero the factor and poison all earlier dates.
    last_close = close.where(close > 0).ffill().to_numpy(dtype=float)
    cash = div.to_numpy(dtype=float)

    factor = np.ones_like(last_close)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(len(last_close) - 2, -1, -1):
            later = factor[t + 1]
            factor[t] = later / (1.0 + cash[t + 1] * later / last_close[t])

    traded = close.to_numpy(dtype=float) / factor
    return pd.DataFrame(traded, index=close.index, columns=close.columns)
