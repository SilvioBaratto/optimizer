"""Universe screening service: assembles DB data and runs investability screens.

Stateless functions following Single Responsibility Principle:
  - _assemble_fundamentals   — query ticker_profiles + exchanges
  - _assemble_price_volume   — pivot price_history to (dates × tickers) matrices
  - _assemble_statements     — query financial_statements as ticker/period DataFrame
  - _build_config            — resolve InvestabilityScreenConfig from request
  - _run_screen_diagnostics  — run each screen individually for per-ticker pass/fail
  - run_universe_screen      — orchestrate and return response dict
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.universe import Exchange, Instrument
from app.models.yfinance_data import FinancialStatement, PriceHistory, TickerProfile
from app.schemas.universe_screen import ScreenPreset, UniverseScreenRequest
from optimizer.universe._config import (
    ExchangeRegion,
    HysteresisConfig,
    InvestabilityScreenConfig,
)
from optimizer.universe._screener import (
    apply_screen,
    compute_addv,
    compute_listing_age,
    compute_trading_frequency,
    count_financial_statements,
)

logger = logging.getLogger(__name__)

_SCREEN_NAMES = (
    "market_cap",
    "mcap_percentile",
    "addv_12m",
    "addv_3m",
    "trading_frequency",
    "price",
    "trading_history",
    "ipo_seasoning",
    "financial_statements",
)

_PRESET_FACTORIES: dict[ScreenPreset, Any] = {
    ScreenPreset.DEVELOPED_MARKETS: InvestabilityScreenConfig.for_developed_markets,
    ScreenPreset.BROAD_UNIVERSE: InvestabilityScreenConfig.for_broad_universe,
    ScreenPreset.SMALL_CAP: InvestabilityScreenConfig.for_small_cap,
    ScreenPreset.LARGE_CAP: InvestabilityScreenConfig.for_large_cap,
}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def _assemble_fundamentals(session: Session) -> pd.DataFrame:
    """Return DataFrame indexed by yfinance_ticker with market_cap, current_price, exchange."""
    stmt = (
        select(
            Instrument.yfinance_ticker,
            TickerProfile.market_cap,
            TickerProfile.current_price,
            Exchange.name.label("exchange"),
        )
        .join(TickerProfile, TickerProfile.instrument_id == Instrument.id)
        .join(Exchange, Exchange.id == Instrument.exchange_id)
        .where(Instrument.delisted_at.is_(None))
        .where(Instrument.yfinance_ticker.isnot(None))
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["market_cap", "current_price", "exchange"])
    df = pd.DataFrame(rows, columns=["yfinance_ticker", "market_cap", "current_price", "exchange"])
    return df.set_index("yfinance_ticker")


def _assemble_price_volume(session: Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (close_df, volume_df) both shaped (dates × tickers)."""
    stmt = (
        select(
            Instrument.yfinance_ticker,
            PriceHistory.date,
            PriceHistory.close,
            PriceHistory.volume,
        )
        .join(PriceHistory, PriceHistory.instrument_id == Instrument.id)
        .where(Instrument.delisted_at.is_(None))
        .where(Instrument.yfinance_ticker.isnot(None))
        .order_by(PriceHistory.date)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume"])
    close_df = df.pivot(index="date", columns="ticker", values="close").astype(float)
    volume_df = df.pivot(index="date", columns="ticker", values="volume").astype(float)
    return close_df, volume_df


def _assemble_statements(session: Session) -> pd.DataFrame:
    """Return DataFrame with columns [ticker, period_type, period_date]."""
    stmt = (
        select(
            Instrument.yfinance_ticker.label("ticker"),
            FinancialStatement.period_type,
            FinancialStatement.period_date,
        )
        .join(FinancialStatement, FinancialStatement.instrument_id == Instrument.id)
        .where(Instrument.yfinance_ticker.isnot(None))
        .distinct()
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["ticker", "period_type", "period_date"])
    return pd.DataFrame(rows, columns=["ticker", "period_type", "period_date"])


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _build_config(request: UniverseScreenRequest) -> InvestabilityScreenConfig:
    """Resolve InvestabilityScreenConfig from request: preset > custom > default."""
    if request.preset is not None:
        return _PRESET_FACTORIES[request.preset]()
    return _apply_overrides(InvestabilityScreenConfig(), request)


def _apply_overrides(
    base: InvestabilityScreenConfig,
    request: UniverseScreenRequest,
) -> InvestabilityScreenConfig:
    """Return new config with only the fields explicitly set in request overridden."""

    def _hysteresis(req_field: Any, default: HysteresisConfig) -> HysteresisConfig:
        if req_field is None:
            return default
        return HysteresisConfig(entry=req_field.entry, exit_=req_field.exit_)

    def _region(value: str | None) -> ExchangeRegion:
        if value is None:
            return base.exchange_region
        return ExchangeRegion(value)

    return InvestabilityScreenConfig(
        market_cap=_hysteresis(request.market_cap, base.market_cap),
        addv_12m=_hysteresis(request.addv_12m, base.addv_12m),
        addv_3m=_hysteresis(request.addv_3m, base.addv_3m),
        trading_frequency=_hysteresis(request.trading_frequency, base.trading_frequency),
        price_us=_hysteresis(request.price_us, base.price_us),
        price_europe=_hysteresis(request.price_europe, base.price_europe),
        min_trading_history=request.min_trading_history or base.min_trading_history,
        min_ipo_seasoning=request.min_ipo_seasoning or base.min_ipo_seasoning,
        min_annual_reports=request.min_annual_reports or base.min_annual_reports,
        min_quarterly_reports=request.min_quarterly_reports or base.min_quarterly_reports,
        exchange_region=_region(request.exchange_region),
        mcap_percentile_entry=request.mcap_percentile_entry or base.mcap_percentile_entry,
        mcap_percentile_exit=request.mcap_percentile_exit or base.mcap_percentile_exit,
    )


# ---------------------------------------------------------------------------
# Screen diagnostics
# ---------------------------------------------------------------------------


def _passing_set(passing: pd.Index) -> set[str]:
    return set(passing.tolist())


def _run_screen_diagnostics(
    fundamentals: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    statements: pd.DataFrame,
    config: InvestabilityScreenConfig,
    current_members: pd.Index | None,
) -> tuple[pd.Index, dict[str, dict[str, bool]]]:
    """Run each screen individually to produce passing tickers and per-screen diagnostics.

    Returns
    -------
    passing : pd.Index
        Tickers passing all screens.
    diagnostics : dict[str, dict[str, bool]]
        {ticker: {screen_name: bool}} for every screened ticker.
    """
    all_tickers = fundamentals.index
    screen_results: dict[str, set[str]] = {name: set() for name in _SCREEN_NAMES}

    _run_market_cap_screens(fundamentals, config, current_members, screen_results)
    _run_liquidity_screens(close_df, volume_df, config, current_members, screen_results)
    _run_price_screen(fundamentals, config, current_members, screen_results)
    _run_history_screens(close_df, config, screen_results)
    _run_statements_screen(statements, config, screen_results)

    passing = _intersect_all_screens(all_tickers, screen_results)
    diagnostics = _build_diagnostics(all_tickers, screen_results)
    return passing, diagnostics


def _run_market_cap_screens(
    fundamentals: pd.DataFrame,
    config: InvestabilityScreenConfig,
    current_members: pd.Index | None,
    results: dict[str, set[str]],
) -> None:
    if "market_cap" not in fundamentals.columns:
        return
    mcap = fundamentals["market_cap"].dropna()
    results["market_cap"] = _passing_set(apply_screen(mcap, config.market_cap, current_members))

    if "exchange" in fundamentals.columns:
        exchange_mapping = fundamentals["exchange"].dropna()
        results["mcap_percentile"] = _passing_set(
            _compute_mcap_percentile_pass(mcap, exchange_mapping, config, current_members)
        )
    else:
        results["mcap_percentile"] = _passing_set(mcap.index)


def _compute_mcap_percentile_pass(
    mcap: pd.Series,
    exchange_mapping: pd.Series,
    config: InvestabilityScreenConfig,
    current_members: pd.Index | None,
) -> pd.Index:
    """Compute exchange-percentile market-cap screen — mirrors _apply_mcap_percentile_screen."""
    import numpy as np

    common = mcap.index.intersection(exchange_mapping.index)
    mcaps = mcap.loc[common]
    exchanges = exchange_mapping.loc[common]

    entry_thresholds = pd.Series(0.0, index=common)
    exit_thresholds = pd.Series(0.0, index=common)
    min_size = 10

    for _, group_mcaps in mcaps.groupby(exchanges):
        if len(group_mcaps) >= min_size:
            entry_thresh = float(np.percentile(group_mcaps.values, config.mcap_percentile_entry * 100))
            exit_thresh = float(np.percentile(group_mcaps.values, config.mcap_percentile_exit * 100))
        else:
            entry_thresh = exit_thresh = 0.0
        entry_thresholds.loc[group_mcaps.index] = entry_thresh
        exit_thresholds.loc[group_mcaps.index] = exit_thresh

    new_entrants = common[mcaps >= entry_thresholds]
    if current_members is None or len(current_members) == 0:
        return new_entrants

    surviving = current_members.intersection(common)
    exit_aligned = exit_thresholds.reindex(surviving, fill_value=0.0)
    surviving = surviving[mcaps.loc[surviving] >= exit_aligned.values]
    return surviving.union(new_entrants)


def _run_liquidity_screens(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    config: InvestabilityScreenConfig,
    current_members: pd.Index | None,
    results: dict[str, set[str]],
) -> None:
    if close_df.empty or volume_df.empty:
        return
    addv_12 = compute_addv(close_df, volume_df, window=252)
    results["addv_12m"] = _passing_set(apply_screen(addv_12.dropna(), config.addv_12m, current_members))
    addv_3 = compute_addv(close_df, volume_df, window=63)
    results["addv_3m"] = _passing_set(apply_screen(addv_3.dropna(), config.addv_3m, current_members))
    freq = compute_trading_frequency(volume_df, window=252)
    results["trading_frequency"] = _passing_set(apply_screen(freq.dropna(), config.trading_frequency, current_members))


def _run_price_screen(
    fundamentals: pd.DataFrame,
    config: InvestabilityScreenConfig,
    current_members: pd.Index | None,
    results: dict[str, set[str]],
) -> None:
    if "current_price" not in fundamentals.columns:
        return
    prices = fundamentals["current_price"].dropna()
    price_thresh = (
        config.price_us
        if config.exchange_region == ExchangeRegion.US
        else config.price_europe
    )
    results["price"] = _passing_set(apply_screen(prices, price_thresh, current_members))


def _run_history_screens(
    close_df: pd.DataFrame,
    config: InvestabilityScreenConfig,
    results: dict[str, set[str]],
) -> None:
    if close_df.empty:
        return
    listing_age = compute_listing_age(close_df)
    results["trading_history"] = _passing_set(
        listing_age.index[listing_age >= config.min_trading_history]
    )
    results["ipo_seasoning"] = _passing_set(
        listing_age.index[listing_age >= config.min_ipo_seasoning]
    )


def _run_statements_screen(
    statements: pd.DataFrame,
    config: InvestabilityScreenConfig,
    results: dict[str, set[str]],
) -> None:
    if statements.empty:
        return
    annual_counts = count_financial_statements(statements, period_type="annual")
    quarterly_counts = count_financial_statements(statements, period_type="quarterly")
    all_tickers = statements["ticker"].dropna().unique()
    annual_ok = annual_counts.reindex(all_tickers, fill_value=0)
    quarterly_ok = quarterly_counts.reindex(all_tickers, fill_value=0)
    has_enough = (annual_ok >= config.min_annual_reports) | (
        quarterly_ok >= config.min_quarterly_reports
    )
    results["financial_statements"] = _passing_set(has_enough.index[has_enough])


def _intersect_all_screens(
    all_tickers: pd.Index,
    screen_results: dict[str, set[str]],
) -> pd.Index:
    """Return tickers passing every non-empty screen."""
    passing = set(all_tickers.tolist())
    for name, passed in screen_results.items():
        if passed:
            passing &= passed
    return pd.Index(sorted(passing))


def _build_diagnostics(
    all_tickers: pd.Index,
    screen_results: dict[str, set[str]],
) -> dict[str, dict[str, bool]]:
    return {
        ticker: {name: ticker in passed for name, passed in screen_results.items()}
        for ticker in all_tickers
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_universe_screen(
    request: UniverseScreenRequest,
    session: Session,
) -> dict[str, Any]:
    """Run universe investability screening and return response dict.

    Returns
    -------
    dict with keys: passing_tickers, total_screened, diagnostics
    """
    fundamentals = _assemble_fundamentals(session)
    if fundamentals.empty:
        return {"passing_tickers": [], "total_screened": 0, "diagnostics": {}}

    close_df, volume_df = _assemble_price_volume(session)
    statements = _assemble_statements(session)
    config = _build_config(request)
    current_members = _resolve_current_members(request.current_members)

    passing, diagnostics = _run_screen_diagnostics(
        fundamentals, close_df, volume_df, statements, config, current_members
    )
    return {
        "passing_tickers": passing.tolist(),
        "total_screened": len(fundamentals),
        "diagnostics": diagnostics,
    }


def _resolve_current_members(tickers: list[str] | None) -> pd.Index | None:
    if not tickers:
        return None
    return pd.Index(tickers)
