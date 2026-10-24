"""Factor construction from fundamentals and price data."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import clone

from optimizer.factors._config import (
    DEFAULT_MIN_OBSERVATIONS,
    FactorConstructionConfig,
    FactorType,
)
from optimizer.fx import FxPriceConverter

logger = logging.getLogger(__name__)

# Longest run of missing rows bridged when reading a momentum reference price;
# covers multi-day exchange closures (Golden Week, Lunar New Year).
_MAX_STALE_PRICE_ROWS = 10


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
    as of ``as_of_date``.  ``PublicationLagConfig`` holds the conventional
    per-source values for ``lag_days``.

    Because only one record per ticker survives, this suits sources with one
    record per reporting period, such as fundamentals.  It cannot be applied
    to a multi-row ledger (insider transactions would collapse to a single
    trade) nor to analyst snapshots (no period end date to filter on).

    Args:
        data: Time-series data containing ``period_date_col`` and optionally
            ``ticker_col``.
        period_date_col: Name of the column holding the period end date.
        as_of_date: The computation date.  Only records available on or before
            this date (after the lag has elapsed) are returned.
        lag_days: Calendar days between period end and data availability.
        ticker_col: Column holding the ticker identifier.  Defaults to
            ``"ticker"``.

    Returns:
        Cross-sectional view: one row per ticker (the most recent available
        record), indexed by ``ticker_col`` when present.  Returns an empty
        DataFrame with the same columns if no records pass the cutoff.
    """
    as_of = pd.Timestamp(as_of_date)
    cutoff = as_of - pd.Timedelta(days=lag_days)

    dates = pd.to_datetime(data[period_date_col])
    available = data.loc[dates <= cutoff].copy()

    if available.empty:
        return pd.DataFrame(columns=data.columns)

    if ticker_col in available.columns:
        available["_sort_date"] = pd.to_datetime(available[period_date_col])
        # tail(1) keeps the newest row intact; groupby().last() would skip NaN
        # per column and splice a stale field into a newer period's row.
        result = (
            available.sort_values("_sort_date", kind="stable")
            .groupby(ticker_col)
            .tail(1)
            .set_index(ticker_col)
            .sort_index()
            .drop(columns=["_sort_date"])
        )
        return result

    return available


def _market_cap(fundamentals: pd.DataFrame) -> pd.Series | None:
    """Return the ``market_cap`` column, or ``None`` when it is absent."""
    mcap = fundamentals.get("market_cap")
    return None if mcap is None else cast(pd.Series, mcap)


def _compute_book_to_price(fundamentals: pd.DataFrame) -> pd.Series:
    """Book value / market cap."""
    book = fundamentals.get("book_value", fundamentals.get("total_equity"))
    mcap = _market_cap(fundamentals)
    if book is None or mcap is None:
        return pd.Series(dtype=float)
    book_s = cast(pd.Series, book)
    return (book_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_earnings_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Net income / market cap (inverse P/E).

    ``net_income`` is a *total* (financial-statement) quantity in the same
    currency unit as ``market_cap``.  There is intentionally no per-share
    fallback (e.g. yfinance ``trailingEps``): dividing a per-share figure by
    a total market cap is dimensionally inconsistent — the result is smaller
    by a factor of shares-outstanding and silently pollutes the value group.
    When ``net_income`` is absent the factor is left uncomputed, matching the
    other statement-based value calculators (roe, profit_margin, accruals).
    """
    earnings = fundamentals.get("net_income")
    mcap = _market_cap(fundamentals)
    if earnings is None or mcap is None:
        return pd.Series(dtype=float)
    earnings_s = cast(pd.Series, earnings)
    return (earnings_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_cash_flow_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Operating cash flow / market cap."""
    ocf = fundamentals.get(
        "operating_cashflow", fundamentals.get("operating_cash_flow")
    )
    mcap = _market_cap(fundamentals)
    if ocf is None or mcap is None:
        return pd.Series(dtype=float)
    ocf_s = cast(pd.Series, ocf)
    return (ocf_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_sales_to_price(fundamentals: pd.DataFrame) -> pd.Series:
    """Total revenue / market cap (inverse P/S)."""
    revenue = fundamentals.get("total_revenue", fundamentals.get("revenue"))
    mcap = _market_cap(fundamentals)
    if revenue is None or mcap is None:
        return pd.Series(dtype=float)
    revenue_s = cast(pd.Series, revenue)
    return (revenue_s / mcap).replace([np.inf, -np.inf], np.nan)


def _compute_ebitda_to_ev(fundamentals: pd.DataFrame) -> pd.Series:
    """EBITDA / enterprise value.

    EV multiples are undefined for non-positive enterprise value: a profitable
    net-cash firm (cash above market cap plus debt) would otherwise get a
    negative yield and rank below ordinary expensive stocks.  Non-positive EV
    yields NaN.
    """
    ebitda = fundamentals.get("ebitda")
    ev = fundamentals.get("enterprise_value")
    if ebitda is None or ev is None:
        return pd.Series(dtype=float)
    ebitda_s = cast(pd.Series, ebitda)
    ev_s = cast(pd.Series, ev)
    positive_ev = ev_s.where(ev_s > 0)
    return (ebitda_s / positive_ev).replace([np.inf, -np.inf], np.nan)


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
    """Return on equity = net income / total equity.

    ROE is only meaningful for positive book equity: a loss-making firm with
    negative equity would otherwise get a positive ratio (negative / negative)
    and rank alongside the most profitable names.  Non-positive equity yields
    NaN, the convention used by Fama-French RMW and AQR QMJ.
    """
    ni = fundamentals.get("net_income")
    equity = fundamentals.get("total_equity", fundamentals.get("book_value"))
    if ni is None or equity is None:
        return pd.Series(dtype=float)
    ni_s = cast(pd.Series, ni)
    equity_s = cast(pd.Series, equity)
    positive_equity = equity_s.where(equity_s > 0)
    return (ni_s / positive_equity).replace([np.inf, -np.inf], np.nan)


def _compute_accruals(fundamentals: pd.DataFrame) -> pd.Series:
    """Sloan accruals = (net_income - operating_cashflow) / total_assets.

    Lower accruals = higher earnings quality (less reliance on non-cash items).
    Direction flip handled in standardization via FACTOR_DIRECTION.
    """
    ni = fundamentals.get("net_income")
    ocf = fundamentals.get(
        "operating_cashflow", fundamentals.get("operating_cash_flow")
    )
    assets = fundamentals.get("total_assets")
    if ni is None or ocf is None or assets is None:
        return pd.Series(dtype=float)
    ni_s = cast(pd.Series, ni)
    ocf_s = cast(pd.Series, ocf)
    assets_s = cast(pd.Series, assets)
    return ((ni_s - ocf_s) / assets_s).replace([np.inf, -np.inf], np.nan)


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
    """Year-over-year total asset growth (natural units).

    Negative asset growth is favorable (conservative investment).
    The sign is NOT flipped here; direction is handled in standardization
    via FACTOR_DIRECTION.
    """
    growth = fundamentals.get("asset_growth")
    if growth is None:
        return pd.Series(dtype=float)
    return pd.Series(cast(pd.Series, growth), dtype=float)


def _compute_momentum(
    price_history: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """12-minus-1-month momentum (skip most recent month).

    A reference date on which a ticker has no print (exchange holiday on a
    union trading calendar, missing bar) falls back to its last valid price,
    for at most ``_MAX_STALE_PRICE_ROWS`` rows so that a ticker that stopped
    trading long ago is not revived from a stale quote.
    """
    if len(price_history) < lookback:
        return pd.Series(dtype=float, index=price_history.columns)
    filled = price_history.ffill(limit=_MAX_STALE_PRICE_ROWS)
    end_prices: pd.Series = filled.iloc[-skip - 1]
    start_prices: pd.Series = filled.iloc[-lookback]
    momentum = (end_prices / start_prices) - 1.0
    return momentum.replace([np.inf, -np.inf], np.nan)


def _simple_returns(price_history: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker simple returns that keep NaN gaps in place.

    Only rows where *every* ticker is NaN (the leading row) are dropped.
    Dropping a row because a single ticker is missing would let one dead
    ticker, young listing or exchange holiday shrink or wipe out the window
    of every other ticker; downstream reductions skip NaN per column instead.
    """
    return price_history.pct_change().dropna(how="all")


def _required_observations(min_obs: int, lookback: int) -> int:
    """Observations a ticker needs, never more than the window can hold."""
    return min(min_obs, lookback)


def _compute_volatility(
    price_history: pd.DataFrame,
    lookback: int = 252,
    min_obs: int = DEFAULT_MIN_OBSERVATIONS,
) -> pd.Series:
    """Annualized return volatility (natural units, always non-negative).

    Lower volatility is favorable for LOW_RISK factor scoring.
    The sign is NOT flipped here; direction is handled in standardization
    via FACTOR_DIRECTION. Tickers with fewer than ``min_obs`` returns in the
    window are NaN.
    """
    tail = _simple_returns(price_history).iloc[-lookback:]
    vol: pd.Series = tail.std()
    enough = tail.count() >= _required_observations(min_obs, lookback)
    return pd.Series(vol.where(enough) * np.sqrt(252), dtype=float)


def _compute_beta(
    price_history: pd.DataFrame,
    market_returns: pd.Series | None = None,
    lookback: int = 252,
    min_obs: int = DEFAULT_MIN_OBSERVATIONS,
) -> pd.Series:
    """Market beta (natural units).

    Lower beta is favorable for LOW_RISK factor scoring.
    The sign is NOT flipped here; direction is handled in standardization
    via FACTOR_DIRECTION. Tickers with fewer than ``min_obs`` jointly observed
    returns are NaN.
    """
    tail = _simple_returns(price_history).iloc[-lookback:]

    if market_returns is None:
        mkt: pd.Series = tail.mean(axis=1)
    else:
        mkt = market_returns.reindex(tail.index).dropna()
        tail = tail.loc[mkt.index]

    # Covariance and market variance must come from the same sample, so each
    # ticker is regressed on the dates where both it and the market exist.
    required = _required_observations(min_obs, lookback)
    betas = [_pairwise_beta(tail[col], mkt, required) for col in tail.columns]
    return pd.Series(betas, index=tail.columns, dtype=float)


def _pairwise_beta(asset: pd.Series, market: pd.Series, min_obs: int = 2) -> float:
    """Beta of ``asset`` on ``market`` over their jointly observed dates."""
    observed = asset.notna() & market.notna()
    asset_obs, market_obs = asset.loc[observed], market.loc[observed]
    if len(asset_obs) < max(2, min_obs):
        return float("nan")
    market_var = cast(float, market_obs.var())
    if market_var < 1e-12:
        # Near-zero market variance (over-diversified proxy) produces
        # numerically unstable betas — return NaN rather than huge values.
        return float("nan")
    return float(asset_obs.cov(market_obs) / market_var)


def _compute_amihud_illiquidity(
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    lookback: int = 252,
    min_obs: int = DEFAULT_MIN_OBSERVATIONS,
    fx_converter: FxPriceConverter | None = None,
    traded_price_history: pd.DataFrame | None = None,
) -> pd.Series:
    """Amihud illiquidity ratio (avg |return| / dollar volume).

    Higher illiquidity = higher expected return premium. Only days with a
    return and non-zero dollar volume count towards ``min_obs``; tickers
    below it are NaN.

    The ratio is not scale-invariant: dollar volume is price x volume, so the
    price's currency unit leaks straight into the cross-sectional ranking.
    ``fx_converter`` expresses the prices used for dollar volume in one base
    currency (rescaling GBp/ZAc/ILA minor units first); returns are always
    taken from the local-currency prices, where the unit cancels.

    Dollar volume also needs the price actually paid.  Dividend-adjusted
    closes understate it at old dates, so ``traded_price_history`` supplies
    as-traded closes for the turnover while returns still come from
    ``price_history`` (total return).
    """
    returns = _simple_returns(price_history)
    traded_price = price_history
    if traded_price_history is not None:
        traded_price = traded_price_history.reindex(columns=price_history.columns)
    if fx_converter is not None:
        # Cloned so the caller's converter is never fitted in place.
        traded_price = clone(fx_converter).fit_transform(traded_price)
    dollar_vol = (traded_price * volume_history).reindex(returns.index)

    tail_ret = returns.iloc[-lookback:]
    tail_dv = dollar_vol.iloc[-lookback:]

    safe_dv = tail_dv.replace(0, np.nan)
    ratio = tail_ret.abs() / safe_dv
    enough = ratio.count() >= _required_observations(min_obs, lookback)
    result: pd.Series = ratio.mean().where(enough)
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


_RATING_COUNT_COLUMNS = ("strong_buy", "buy", "hold", "sell", "strong_sell")
_SNAPSHOT_PERIOD = re.compile(r"^\s*([+-]?\d+)\s*m\s*$", re.IGNORECASE)


def _compute_recommendation_change(
    analyst_data: pd.DataFrame | None,
) -> pd.Series:
    """Revision in analyst sentiment between the two latest snapshots.

    Each snapshot is scored as its net buy-rating share,
    ``(strong_buy + buy - sell - strong_sell) / all ratings``, so a name with
    40 analysts is comparable to one with 4 and a steady consensus scores zero
    however bullish it is.  The factor is the latest snapshot's share minus the
    prior one's, with snapshots ordered by ``period`` (``0m`` is the latest,
    ``-1m`` a month earlier, as stored by ingestion).  Tickers with fewer than
    two snapshots are not scored, and rating counts without ``ticker`` and
    ``period`` columns cannot form a revision and yield an empty Series.

    A precomputed ``recommendation_change`` column takes precedence and is
    averaged per ticker.
    """
    if analyst_data is None or len(analyst_data) == 0:
        return pd.Series(dtype=float)
    if "recommendation_change" in analyst_data.columns:
        result: pd.Series = cast(
            pd.Series, analyst_data.groupby("ticker")["recommendation_change"].mean()
        )
        return result
    if not {"ticker", "period", "strong_buy"} <= set(analyst_data.columns):
        return pd.Series(dtype=float)

    counts = (
        analyst_data.reindex(columns=list(_RATING_COUNT_COLUMNS))
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    net = counts["strong_buy"] + counts["buy"] - counts["sell"] - counts["strong_sell"]
    total = counts.sum(axis=1)
    months = pd.to_numeric(
        analyst_data["period"]
        .astype("string")
        .str.extract(_SNAPSHOT_PERIOD, expand=False),
        errors="coerce",
    )
    snapshots = (
        pd.DataFrame(
            {
                "ticker": analyst_data["ticker"],
                "months": months,
                "share": net / total.where(total > 0),
            }
        )
        .dropna()
        .drop_duplicates(["ticker", "months"], keep="last")
        .sort_values(["ticker", "months"], kind="stable")
    )
    snapshots["change"] = snapshots.groupby("ticker")["share"].diff()
    latest = snapshots.groupby("ticker").tail(1).set_index("ticker")["change"]
    return latest.dropna().astype(float)


# (ledger amount column, fundamentals column that scales it), in preference
# order: traded value over market cap, else shares over shares outstanding.
_INSIDER_AMOUNT_SCALES: tuple[tuple[str, str], ...] = (
    ("value", "market_cap"),
    ("shares", "shares_outstanding"),
)


def _compute_net_insider_buying(
    insider_data: pd.DataFrame | None,
    fundamentals: pd.DataFrame,
) -> pd.Series:
    """Net insider buying as a fraction of the company (purchases - sales).

    Only open-market purchases (+) and sales (-) carry a signal.  Equity
    awards, gifts, option exercises and rows with a blank type are not
    discretionary trades and contribute zero, so a company that grants stock
    regularly is not read as one whose insiders are selling.

    The net amount is divided by the company's size so tickers are comparable:
    signed ``value`` over ``market_cap`` when the ledger carries a value, else
    signed ``shares`` over ``shares_outstanding``.  A raw share count would
    rank a penny stock above a large cap for the same dollar conviction.
    Direction comes only from ``transaction_type``: the ledger stores sales as
    positive numbers, so without that column (or without a positive size in
    ``fundamentals``) the factor is left uncomputed rather than reporting
    traded volume as net buying.

    Unknown data is left unscored rather than neutral.  A ticker with a
    purchase or sale that has no usable amount is NaN, since dropping that
    trade would misstate the net; a ticker none of whose rows has a transaction
    type is NaN; and a ledger with no transaction type at all yields an empty
    Series.  A missing amount on a non-trade row (an award, say) is harmless.
    """
    if insider_data is None or len(insider_data) == 0:
        return pd.Series(dtype=float)
    if not {"ticker", "transaction_type"} <= set(insider_data.columns):
        return pd.Series(dtype=float)
    for amount_col, scale_col in _INSIDER_AMOUNT_SCALES:
        if (
            amount_col in insider_data.columns
            and scale_col in fundamentals.columns
            and insider_data[amount_col].notna().any()
        ):
            break
    else:
        return pd.Series(dtype=float)

    # astype("string") keeps the .str accessor valid for an all-NaN float
    # column, which is what a blank source column becomes after loading.
    tx_type = insider_data["transaction_type"].astype("string").str.lower()
    classified = tx_type.str.strip().fillna("").ne("").to_numpy(bool)
    if not classified.any():
        return pd.Series(dtype=float)
    is_purchase = tx_type.str.contains("purchase|buy", na=False).to_numpy(bool)
    is_sale = tx_type.str.contains("sale|sell", na=False).to_numpy(bool)
    direction = np.select(
        [is_purchase & ~is_sale, is_sale & ~is_purchase], [1.0, -1.0], default=0.0
    )
    amount = pd.to_numeric(insider_data[amount_col], errors="coerce").to_numpy(
        "float64"
    )
    unusable = np.isnan(amount)
    by_ticker = (
        insider_data[["ticker"]]
        .assign(
            signed=np.where(unusable, 0.0, amount) * direction,
            classified=classified,
            unknown_trade=(direction != 0) & unusable,
        )
        .groupby("ticker")
    )
    scored = by_ticker["classified"].any() & ~by_ticker["unknown_trade"].any()
    net = cast(pd.Series, by_ticker["signed"].sum().where(scored))
    scale = pd.to_numeric(fundamentals[scale_col], errors="coerce")
    return net / scale.where(scale > 0).reindex(net.index)


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
    FactorType.ACCRUALS: _compute_accruals,
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
    market_returns: pd.Series | None = None,
    fx_converter: FxPriceConverter | None = None,
    traded_price_history: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute a single factor.

    Args:
        factor_type: Which factor to compute.
        fundamentals: Cross-sectional data indexed by ticker.
        price_history: Price matrix (dates x tickers).
        volume_history: Volume matrix (dates x tickers).
        analyst_data: Analyst recommendation data.
        insider_data: Insider transaction ledger with ``ticker``,
            ``transaction_type`` and ``value`` (scaled by ``market_cap``) or
            ``shares`` (scaled by ``shares_outstanding``) columns, used only
            by ``NET_INSIDER_BUYING``.
        config: Construction parameters.
        market_returns: Pre-computed market return series for beta estimation.
            When provided, used as the benchmark instead of the equal-weight
            cross-sectional mean.  Pass a currency-consistent broad index (e.g.
            SPY daily returns) when ``price_history`` spans multiple currency
            zones.
        fx_converter: Converter that expresses ``price_history`` in a single
            base currency (``currency_map`` keyed on the DB ``price_unit``,
            so minor units such as GBp are rescaled too).  Used only by
            ``AMIHUD_ILLIQUIDITY``, whose dollar volume depends on the price
            unit; without it ``price_history`` must already share one major
            currency or the factor ranks by quotation currency.
        traded_price_history: As-traded (not dividend-adjusted) closes,
            dates x tickers in the same unit as ``price_history``.  Used only
            by ``AMIHUD_ILLIQUIDITY`` for dollar volume.  The stored
            ``price_history`` closes are yfinance ``auto_adjust=True``, which
            shrinks old closes by every later dividend; build this panel with
            ``optimizer.preprocessing.unadjust_close``.  Defaults to
            ``price_history``.

    Returns:
        Factor values indexed by ticker.
    """
    if config is None:
        config = FactorConstructionConfig()

    if factor_type in _FUNDAMENTAL_FACTORS:
        return _FUNDAMENTAL_FACTORS[factor_type](fundamentals)

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
                min_obs=config.min_observations,
            )
        case FactorType.BETA:
            return _compute_beta(
                price_history,
                market_returns=market_returns,
                lookback=config.beta_lookback,
                min_obs=config.min_observations,
            )
        case FactorType.AMIHUD_ILLIQUIDITY:
            if volume_history is None:
                return pd.Series(dtype=float)
            return _compute_amihud_illiquidity(
                price_history,
                volume_history,
                lookback=config.amihud_lookback,
                min_obs=config.min_observations,
                fx_converter=fx_converter,
                traded_price_history=traded_price_history,
            )
        case FactorType.RECOMMENDATION_CHANGE:
            return _compute_recommendation_change(analyst_data)
        case FactorType.NET_INSIDER_BUYING:
            return _compute_net_insider_buying(insider_data, fundamentals)
        case _:
            return pd.Series(dtype=float)


def compute_all_factors(
    fundamentals: pd.DataFrame,
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame | None = None,
    analyst_data: pd.DataFrame | None = None,
    insider_data: pd.DataFrame | None = None,
    config: FactorConstructionConfig | None = None,
    market_returns: pd.Series | None = None,
    fx_converter: FxPriceConverter | None = None,
    traded_price_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all configured factors.

    No as-of filtering or publication lag is applied: every row of every
    input is used, so a backtest must hand in data that was already knowable
    on the decision date.

    - Fundamentals: one record per reporting period, so slice them upstream
      with `align_to_pit` and the lags in `PublicationLagConfig`.
    - Insider ledger: several rows per ticker, which `align_to_pit` would
      collapse to one.  Filter it by its transaction date column yourself,
      keeping rows dated on or before the decision date less the lag.
    - Analyst snapshots: dated only by the relative ``period`` label, so there
      is no column to filter on.  Pass the snapshots as they stood on the
      decision date.

    Args:
        fundamentals: Cross-sectional data indexed by ticker.
        price_history: Price matrix (dates x tickers).
        volume_history: Volume matrix.
        analyst_data: Analyst recommendation data.
        insider_data: Insider transaction ledger used only by
            ``NET_INSIDER_BUYING``, with ``ticker``, ``transaction_type`` and
            ``value`` (scaled by ``market_cap`` in ``fundamentals``) or
            ``shares`` (scaled by ``shares_outstanding``) columns.  The factor
            is dropped from the result when these columns are missing.
        config: Construction parameters.
        market_returns: Pre-computed market return series for beta estimation.
            See `compute_factor` for details.
        fx_converter: Base-currency price converter for the dollar-volume
            based factors.  See `compute_factor` for details.
        traded_price_history: As-traded closes for the dollar-volume based
            factors.  See `compute_factor` for details.

    Returns:
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
            market_returns=market_returns,
            fx_converter=fx_converter,
            traded_price_history=traded_price_history,
        )
        if len(series) > 0:
            results[factor_type.value] = series

    if not results:
        return pd.DataFrame(index=fundamentals.index)

    return pd.DataFrame(results).reindex(fundamentals.index)
