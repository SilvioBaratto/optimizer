"""Factor construction from fundamentals and price data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from optimizer.factors._config import (
    FactorConstructionConfig,
    FactorType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Point-in-time alignment
# ---------------------------------------------------------------------------


def align_to_pit(
    data: pd.DataFrame,
    period_date_col: str,
    as_of_date: pd.Timestamp | str,
    lag_days: int,
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Filter time-series data to records published before ``as_of_date``.

    A record with period end date ``D`` is considered published
    ``lag_days`` calendar days after ``D``.  A record is available as of
    ``as_of_date`` only when ``D + lag_days <= as_of_date``, equivalently
    when ``D <= as_of_date - lag_days``.

    For each ticker, the most recent record satisfying the availability
    constraint is returned so that callers receive a cross-sectional view
    as of ``as_of_date``.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data containing ``period_date_col`` and optionally
        ``ticker_col``.
    period_date_col : str
        Name of the column holding the period end date.
    as_of_date : pd.Timestamp or str
        The computation date.  Only records available on or before this
        date (after the lag has elapsed) are returned.
    lag_days : int
        Calendar days between period end and data availability.
    ticker_col : str
        Column holding the ticker identifier.  Defaults to ``"ticker"``.

    Returns
    -------
    pd.DataFrame
        Cross-sectional view: one row per ticker (the most recent
        available record), indexed by ``ticker_col`` when present.
        Returns an empty DataFrame with the same columns if no records
        pass the cutoff.
    """
    as_of = pd.Timestamp(as_of_date)
    cutoff = as_of - pd.Timedelta(days=lag_days)

    dates = pd.to_datetime(data[period_date_col])
    available = data.loc[dates <= cutoff].copy()

    if available.empty:
        return pd.DataFrame(columns=data.columns)

    if ticker_col in available.columns:
        available["_sort_date"] = pd.to_datetime(available[period_date_col])
        result = (
            available.sort_values("_sort_date")
            .groupby(ticker_col)
            .last()
            .drop(columns=["_sort_date"])
        )
        return result

    return available


# ---------------------------------------------------------------------------
# Individual factor calculators
# ---------------------------------------------------------------------------


def _compute_book_to_price(fundamentals: pd.DataFrame) -> pd.Series:
    """Book value / market cap."""
    book = fundamentals.get("book_value", fundamentals.get("total_equity"))
    if book is None:
        return pd.Series(dtype=float)
    book_s = cast(pd.Series, book)
    mcap = fundamentals["market_cap"]
    return (book_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_earnings_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Net income / market cap (inverse P/E)."""
    earnings = fundamentals.get("net_income", fundamentals.get("trailing_eps"))
    if earnings is None:
        return pd.Series(dtype=float)
    earnings_s = cast(pd.Series, earnings)
    mcap = fundamentals["market_cap"]
    return (earnings_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_cash_flow_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Operating cash flow / market cap."""
    ocf = fundamentals.get(
        "operating_cashflow", fundamentals.get("operating_cash_flow")
    )
    if ocf is None:
        return pd.Series(dtype=float)
    ocf_s = cast(pd.Series, ocf)
    mcap = fundamentals["market_cap"]
    return (ocf_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_sales_to_price(fundamentals: pd.DataFrame) -> pd.Series:
    """Total revenue / market cap (inverse P/S)."""
    revenue = fundamentals.get("total_revenue", fundamentals.get("revenue"))
    if revenue is None:
        return pd.Series(dtype=float)
    revenue_s = cast(pd.Series, revenue)
    mcap = fundamentals["market_cap"]
    return (revenue_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_ebitda_to_ev(fundamentals: pd.DataFrame) -> pd.Series:
    """EBITDA / enterprise value."""
    ebitda = fundamentals.get("ebitda")
    ev = fundamentals.get("enterprise_value")
    if ebitda is None or ev is None:
        return pd.Series(dtype=float)
    ebitda_s = cast(pd.Series, ebitda)
    ev_s = cast(pd.Series, ev)
    return (ebitda_s / ev_s).replace([np.inf, -np.inf], np.nan)


def _compute_gross_profitability(fundamentals: pd.DataFrame) -> pd.Series:
    """Gross profit / total assets (Novy-Marx)."""
    gp = fundamentals.get("gross_profit", fundamentals.get("gross_profits"))
    assets = fundamentals.get("total_assets")
    if gp is None or assets is None:
        return pd.Series(dtype=float)
    gp_s = cast(pd.Series, gp)
    assets_s = cast(pd.Series, assets)
    return (gp_s / assets_s).replace([np.inf, -np.inf], np.nan)


def _compute_roe(fundamentals: pd.DataFrame) -> pd.Series:
    """Return on equity = net income / total equity."""
    ni = fundamentals.get("net_income")
    equity = fundamentals.get("total_equity", fundamentals.get("book_value"))
    if ni is None or equity is None:
        return pd.Series(dtype=float)
    ni_s = cast(pd.Series, ni)
    equity_s = cast(pd.Series, equity)
    return (ni_s / equity_s).replace([np.inf, -np.inf], np.nan)


def _compute_operating_margin(fundamentals: pd.DataFrame) -> pd.Series:
    """Operating income / total revenue."""
    oi = fundamentals.get("operating_income", fundamentals.get("ebit"))
    revenue = fundamentals.get("total_revenue", fundamentals.get("revenue"))
    if oi is None or revenue is None:
        return pd.Series(dtype=float)
    oi_s = cast(pd.Series, oi)
    revenue_s = cast(pd.Series, revenue)
    return (oi_s / revenue_s).replace([np.inf, -np.inf], np.nan)


def _compute_profit_margin(fundamentals: pd.DataFrame) -> pd.Series:
    """Net income / total revenue."""
    ni = fundamentals.get("net_income")
    revenue = fundamentals.get("total_revenue", fundamentals.get("revenue"))
    if ni is None or revenue is None:
        return pd.Series(dtype=float)
    ni_s = cast(pd.Series, ni)
    revenue_s = cast(pd.Series, revenue)
    return (ni_s / revenue_s).replace([np.inf, -np.inf], np.nan)


def _compute_asset_growth(fundamentals: pd.DataFrame) -> pd.Series:
    """Year-over-year total asset growth.

    Negative asset growth is favorable (conservative investment).
    The sign is flipped so higher values = more conservative.
    """
    growth = fundamentals.get("asset_growth")
    if growth is None:
        return pd.Series(dtype=float)
    return pd.Series(-cast(pd.Series, growth), dtype=float)


def _compute_momentum(
    price_history: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """12-minus-1-month momentum (skip most recent month)."""
    if len(price_history) < lookback:
        return pd.Series(dtype=float, index=price_history.columns)
    end_prices: pd.Series = price_history.iloc[-skip - 1]
    start_prices: pd.Series = price_history.iloc[-lookback]
    momentum = (end_prices / start_prices) - 1.0
    return momentum.replace([np.inf, -np.inf], np.nan)


def _compute_volatility(
    price_history: pd.DataFrame,
    lookback: int = 252,
) -> pd.Series:
    """Annualized return volatility.

    Lower volatility is favorable, so sign is flipped.
    """
    returns = price_history.pct_change().dropna()
    tail = returns.iloc[-lookback:] if len(returns) >= lookback else returns
    vol: pd.Series = tail.std()
    return pd.Series(-(vol * np.sqrt(252)), dtype=float)


def _compute_beta(
    price_history: pd.DataFrame,
    market_returns: pd.Series | None = None,
    lookback: int = 252,
) -> pd.Series:
    """Market beta.

    Lower beta is favorable, so sign is flipped.
    """
    returns = price_history.pct_change().dropna()
    tail = returns.iloc[-lookback:] if len(returns) >= lookback else returns

    if market_returns is None:
        mkt: pd.Series = tail.mean(axis=1)
    else:
        mkt = market_returns.reindex(tail.index).dropna()
        tail = tail.loc[mkt.index]

    market_var = cast(float, mkt.var())
    if market_var == 0:
        return pd.Series(0.0, index=tail.columns)

    beta_values = {
        str(col): float(tail[col].cov(mkt) / market_var) for col in tail.columns
    }
    return pd.Series(
        {k: -v for k, v in beta_values.items()},
        dtype=float,
    )


def _compute_amihud_illiquidity(
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    lookback: int = 252,
) -> pd.Series:
    """Amihud illiquidity ratio (avg |return| / dollar volume).

    Higher illiquidity = higher expected return premium.
    """
    returns = price_history.pct_change().dropna()
    dollar_vol = (price_history * volume_history).reindex(returns.index)

    tail_ret = returns.iloc[-lookback:] if len(returns) >= lookback else returns
    tail_dv = dollar_vol.iloc[-lookback:] if len(dollar_vol) >= lookback else dollar_vol

    safe_dv = tail_dv.replace(0, np.nan)
    ratio = tail_ret.abs() / safe_dv
    result: pd.Series = ratio.mean()
    return result


def _compute_dividend_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Trailing dividend yield."""
    dy = fundamentals.get(
        "dividend_yield",
        fundamentals.get("trailing_annual_dividend_yield"),
    )
    if dy is None:
        return pd.Series(dtype=float)
    return cast(pd.Series, dy).fillna(0.0)


def _compute_recommendation_change(
    analyst_data: pd.DataFrame | None,
) -> pd.Series:
    """Net recommendation upgrades - downgrades."""
    if analyst_data is None or len(analyst_data) == 0:
        return pd.Series(dtype=float)
    if "recommendation_change" in analyst_data.columns:
        result: pd.Series = analyst_data.groupby("ticker")[
            "recommendation_change"
        ].mean()
        return result
    if "strong_buy" in analyst_data.columns:
        # Compute from raw counts: positive = bullish
        bull = analyst_data.get("strong_buy")
        buy = analyst_data.get("buy")
        strong_sell = analyst_data.get("strong_sell")
        sell = analyst_data.get("sell")
        bull_total = (cast(pd.Series, bull) if bull is not None else 0) + (
            cast(pd.Series, buy) if buy is not None else 0
        )
        bear_total = (
            cast(pd.Series, strong_sell) if strong_sell is not None else 0
        ) + (cast(pd.Series, sell) if sell is not None else 0)
        score = pd.Series(bull_total - bear_total, dtype=float)
        if "ticker" in analyst_data.columns:
            grouped: pd.Series = score.groupby(analyst_data["ticker"]).mean()
            return grouped
        return score
    return pd.Series(dtype=float)


def _compute_net_insider_buying(
    insider_data: pd.DataFrame | None,
) -> pd.Series:
    """Net insider buying volume (purchases - sales)."""
    if insider_data is None or len(insider_data) == 0:
        return pd.Series(dtype=float)
    if "shares" not in insider_data.columns or "ticker" not in insider_data.columns:
        return pd.Series(dtype=float)
    if "transaction_type" in insider_data.columns:
        insider = insider_data.copy()
        is_purchase = (
            insider["transaction_type"]
            .str.lower()
            .str.contains("purchase|buy", na=False)
        )
        insider["signed_shares"] = insider["shares"].where(
            is_purchase, -insider["shares"]
        )
        result: pd.Series = insider.groupby("ticker")["signed_shares"].sum()
        return result
    result_sum: pd.Series = insider_data.groupby("ticker")["shares"].sum()
    return result_sum


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_FUNDAMENTAL_FACTORS: dict[FactorType, Callable[[pd.DataFrame], pd.Series]] = {
    FactorType.BOOK_TO_PRICE: _compute_book_to_price,
    FactorType.EARNINGS_YIELD: _compute_earnings_yield,
    FactorType.CASH_FLOW_YIELD: _compute_cash_flow_yield,
    FactorType.SALES_TO_PRICE: _compute_sales_to_price,
    FactorType.EBITDA_TO_EV: _compute_ebitda_to_ev,
    FactorType.GROSS_PROFITABILITY: _compute_gross_profitability,
    FactorType.ROE: _compute_roe,
    FactorType.OPERATING_MARGIN: _compute_operating_margin,
    FactorType.PROFIT_MARGIN: _compute_profit_margin,
    FactorType.ASSET_GROWTH: _compute_asset_growth,
    FactorType.DIVIDEND_YIELD: _compute_dividend_yield,
}


def compute_factor(
    factor_type: FactorType,
    fundamentals: pd.DataFrame,
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame | None = None,
    analyst_data: pd.DataFrame | None = None,
    insider_data: pd.DataFrame | None = None,
    config: FactorConstructionConfig | None = None,
) -> pd.Series:
    """Compute a single factor.

    Parameters
    ----------
    factor_type : FactorType
        Which factor to compute.
    fundamentals : pd.DataFrame
        Cross-sectional data indexed by ticker.
    price_history : pd.DataFrame
        Price matrix (dates x tickers).
    volume_history : pd.DataFrame or None
        Volume matrix (dates x tickers).
    analyst_data : pd.DataFrame or None
        Analyst recommendation data.
    insider_data : pd.DataFrame or None
        Insider transaction data.
    config : FactorConstructionConfig or None
        Construction parameters.

    Returns
    -------
    pd.Series
        Factor values indexed by ticker.
    """
    if config is None:
        config = FactorConstructionConfig()

    # Fundamental factors
    if factor_type in _FUNDAMENTAL_FACTORS:
        return _FUNDAMENTAL_FACTORS[factor_type](fundamentals)

    # Price-based factors
    match factor_type:
        case FactorType.MOMENTUM_12_1:
            return _compute_momentum(
                price_history,
                lookback=config.momentum_lookback,
                skip=config.momentum_skip,
            )
        case FactorType.VOLATILITY:
            return _compute_volatility(
                price_history,
                lookback=config.volatility_lookback,
            )
        case FactorType.BETA:
            return _compute_beta(
                price_history,
                lookback=config.beta_lookback,
            )
        case FactorType.AMIHUD_ILLIQUIDITY:
            if volume_history is None:
                return pd.Series(dtype=float)
            return _compute_amihud_illiquidity(
                price_history,
                volume_history,
                lookback=config.amihud_lookback,
            )
        case FactorType.RECOMMENDATION_CHANGE:
            return _compute_recommendation_change(analyst_data)
        case FactorType.NET_INSIDER_BUYING:
            return _compute_net_insider_buying(insider_data)
        case _:
            return pd.Series(dtype=float)


def compute_all_factors(
    fundamentals: pd.DataFrame,
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame | None = None,
    analyst_data: pd.DataFrame | None = None,
    insider_data: pd.DataFrame | None = None,
    config: FactorConstructionConfig | None = None,
) -> pd.DataFrame:
    """Compute all configured factors.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Cross-sectional data indexed by ticker.
    price_history : pd.DataFrame
        Price matrix (dates x tickers).
    volume_history : pd.DataFrame or None
        Volume matrix.
    analyst_data : pd.DataFrame or None
        Analyst recommendation data.
    insider_data : pd.DataFrame or None
        Insider transaction data.
    config : FactorConstructionConfig or None
        Construction parameters.

    Returns
    -------
    pd.DataFrame
        Tickers x factors matrix.
    """
    if config is None:
        config = FactorConstructionConfig()

    results: dict[str, pd.Series] = {}
    for factor_type in config.factors:
        series = compute_factor(
            factor_type=factor_type,
            fundamentals=fundamentals,
            price_history=price_history,
            volume_history=volume_history,
            analyst_data=analyst_data,
            insider_data=insider_data,
            config=config,
        )
        if len(series) > 0:
            results[factor_type.value] = series

    if not results:
        return pd.DataFrame(index=fundamentals.index)

    return pd.DataFrame(results).reindex(fundamentals.index)
