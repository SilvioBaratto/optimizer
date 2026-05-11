"""Pure metric computation functions extracted from stock_selection_pipeline.py.

Zero dependencies on sibling pipeline modules. Imports only from numpy, pandas,
skfolio, yfinance, and the standard library.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOP_N_DISPLAY: int = 25  # tickers shown in the selection table

_METRICS_KEY_MAP: dict[str, str] = {
    "Ann. Return": "ann_return",
    "Ann. Vol": "ann_vol",
    "Sharpe (rf)": "sharpe",
    "Sortino": "sortino",
    "Info Ratio": "info_ratio",
    "Downside Vol": "downside_vol",
    "Max Drawdown": "max_drawdown",
}


# ---------------------------------------------------------------------------
# Pure metric computation
# ---------------------------------------------------------------------------


def _annualized_return(r: pd.Series) -> float:
    """Compound annualized return from daily returns."""
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def _daily_rf(returns: pd.Series, rf_series: pd.Series | None) -> pd.Series:
    """Forward-filled daily risk-free rate aligned to ``returns`` index."""
    if rf_series is None or rf_series.empty:
        return pd.Series(0.0, index=returns.index)
    return rf_series.reindex(returns.index, method="ffill").fillna(0.0) / 252.0


def _sharpe(
    returns: pd.Series,
    rf_series: pd.Series | None = None,
) -> float:
    """Annualized Sharpe ratio with time-varying risk-free rate.

    Fix issue #246: previous version used rf=0 (return-to-vol ratio), which
    systematically overstated Sharpe for low-volatility strategies by up to
    40-50% during 2022-2024 when Fed Funds exceeded 5%.  This version uses
    contemporaneous daily FRED DGS3MO as the risk-free benchmark.
    """
    if returns.empty:
        return float("nan")

    if rf_series is not None and not rf_series.empty:
        # Annual rate → daily; forward-fill to trading calendar
        daily_rf = rf_series.reindex(returns.index, method="ffill").fillna(0.0) / 252.0
        excess = returns - daily_rf
    else:
        excess = returns

    ann_excess = _annualized_return(excess)
    std_val = cast(float, excess.std(ddof=1))
    vol = std_val * np.sqrt(252.0)
    return ann_excess / vol if vol > 0.0 else float("nan")


def _sortino(returns: pd.Series, rf_series: pd.Series | None = None) -> float:
    """Annualised Sortino: excess return / downside vol (Cycle 4 §9.3)."""
    if returns.empty:
        return 0.0
    daily_rf = _daily_rf(returns, rf_series)
    excess = returns - daily_rf
    downside = excess[excess < 0.0]
    if downside.empty:
        return 0.0
    downside_vol = float(downside.std(ddof=1)) * np.sqrt(252.0)
    if downside_vol <= 0.0:
        return 0.0
    return _annualized_return(excess) / downside_vol


def _downside_vol(returns: pd.Series, rf_series: pd.Series | None = None) -> float:
    """Annualised std of below-rf returns (Cycle 4 §9.3)."""
    if returns.empty:
        return 0.0
    daily_rf = _daily_rf(returns, rf_series)
    downside = (returns - daily_rf)[(returns - daily_rf) < 0.0]
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=1)) * np.sqrt(252.0)


def _information_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Annualised IR = mean(active) / std(active) × √252 (Cycle 4 §9.3)."""
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) == 0:
        return float("nan")
    active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    std_val = float(active.std(ddof=1))
    if std_val <= 1e-12:
        return 0.0
    return float(active.mean()) / std_val * np.sqrt(252.0)


# ---------------------------------------------------------------------------
# JSON-safe helpers
# ---------------------------------------------------------------------------


def _to_json_safe(value: float) -> float | None:
    """Cast numpy scalars to float; replace NaN with None for strict JSON."""
    if value is None:
        return None
    f = float(value)
    return None if np.isnan(f) else f


def _project_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Convert display-key metrics dict to JSON-safe schema dict."""
    return {
        json_key: _to_json_safe(metrics.get(display_key, float("nan")))
        for display_key, json_key in _METRICS_KEY_MAP.items()
    }


# ---------------------------------------------------------------------------
# External data fetching
# ---------------------------------------------------------------------------


def _fetch_benchmark_returns(
    start: pd.Timestamp,
    end: pd.Timestamp,
    ticker: str = "SPY",
) -> pd.Series:
    """Download daily benchmark returns from yfinance."""
    import yfinance as yf
    from skfolio.preprocessing import prices_to_returns

    data: pd.DataFrame = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if data is None or data.empty:
        logger.warning("Could not download benchmark %s", ticker)
        return pd.Series(dtype=float, name=ticker)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close_series: pd.Series = close  # type: ignore[assignment]
    ret_df = prices_to_returns(close_series.to_frame(ticker))
    ret_series: pd.Series = ret_df.iloc[:, 0]
    ret_series.name = ticker
    return ret_series


def _build_country_map(db_manager: Any) -> dict[str, str]:
    """Build ticker → country mapping from ticker_profiles."""
    from sqlalchemy import text

    with db_manager.get_session() as session:
        rows = session.execute(
            text(
                "SELECT i.yfinance_ticker, tp.country "
                "FROM instruments i "
                "LEFT JOIN ticker_profiles tp ON tp.instrument_id = i.id "
                "WHERE tp.country IS NOT NULL"
            )
        ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}
