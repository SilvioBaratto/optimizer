"""Investability screening logic."""

from __future__ import annotations

import pandas as pd

from optimizer.universe._config import (
    ExchangeRegion,
    HysteresisConfig,
    InvestabilityScreenConfig,
)


def apply_screen(
    values: pd.Series,
    hysteresis: HysteresisConfig,
    current_members: pd.Index | None = None,
) -> pd.Index:
    """Apply a single screen with hysteresis.

    New stocks must exceed ``hysteresis.entry``; existing members
    are retained until they fall below ``hysteresis.exit_``.

    Parameters
    ----------
    values : pd.Series
        Metric values indexed by ticker.
    hysteresis : HysteresisConfig
        Entry/exit thresholds.
    current_members : pd.Index or None
        Tickers currently in the universe.  If ``None``, entry
        thresholds are applied to all stocks.

    Returns
    -------
    pd.Index
        Tickers passing the screen.
    """
    new_entrants = values.index[values >= hysteresis.entry]

    if current_members is None or len(current_members) == 0:
        return new_entrants

    # Current members survive if they remain above exit threshold
    surviving = current_members.intersection(values.index)
    surviving = surviving[values.loc[surviving] >= hysteresis.exit_]

    return surviving.union(new_entrants)


def compute_addv(
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    window: int,
) -> pd.Series:
    """Compute average daily dollar volume over a trailing window.

    Parameters
    ----------
    price_history : pd.DataFrame
        Price matrix (dates x tickers).
    volume_history : pd.DataFrame
        Volume matrix (dates x tickers), aligned with price_history.
    window : int
        Number of trailing trading days.

    Returns
    -------
    pd.Series
        Average daily dollar volume per ticker.
    """
    dollar_volume = price_history * volume_history
    if len(dollar_volume) >= window:
        tail = dollar_volume.iloc[-window:]
    else:
        tail = dollar_volume
    return tail.mean()


def compute_trading_frequency(
    volume_history: pd.DataFrame,
    window: int,
) -> pd.Series:
    """Compute fraction of trading days with nonzero volume.

    Parameters
    ----------
    volume_history : pd.DataFrame
        Volume matrix (dates x tickers).
    window : int
        Number of trailing trading days.

    Returns
    -------
    pd.Series
        Trading frequency per ticker (0 to 1).
    """
    if len(volume_history) >= window:
        tail = volume_history.iloc[-window:]
    else:
        tail = volume_history
    return (tail > 0).mean()


def compute_listing_age(price_history: pd.DataFrame) -> pd.Series:
    """Compute listing age in trading days for each ticker.

    Parameters
    ----------
    price_history : pd.DataFrame
        Price matrix (dates x tickers).

    Returns
    -------
    pd.Series
        Number of non-NaN trading days per ticker.
    """
    return price_history.notna().sum()


def count_financial_statements(
    statements: pd.DataFrame,
    period_type: str,
    min_lookback_days: int | None = None,
) -> pd.Series:
    """Count financial statements per ticker.

    Parameters
    ----------
    statements : pd.DataFrame
        Must contain columns ``ticker``, ``period_type``, and
        optionally ``period_date``.
    period_type : str
        Filter to this period type (e.g. ``"annual"`` or
        ``"quarterly"``).
    min_lookback_days : int or None
        If provided, only count statements with ``period_date``
        within this many calendar days from the latest date.

    Returns
    -------
    pd.Series
        Statement count indexed by ticker.
    """
    filtered = statements[statements["period_type"] == period_type]

    if min_lookback_days is not None and "period_date" in filtered.columns:
        filtered = filtered.copy()
        filtered["period_date"] = pd.to_datetime(filtered["period_date"])
        latest = filtered["period_date"].max()
        if pd.notna(latest):
            cutoff = latest - pd.Timedelta(days=min_lookback_days)
            filtered = filtered[filtered["period_date"] >= cutoff]

    return filtered.groupby("ticker").size()


def apply_investability_screens(
    fundamentals: pd.DataFrame,
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    financial_statements: pd.DataFrame | None = None,
    config: InvestabilityScreenConfig | None = None,
    current_members: pd.Index | None = None,
) -> pd.Index:
    """Apply all investability screens to produce a universe.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Cross-sectional data with one row per ticker.  Required
        columns: ``market_cap``, ``current_price``.  Index is ticker.
    price_history : pd.DataFrame
        Price matrix (dates x tickers).
    volume_history : pd.DataFrame
        Volume matrix (dates x tickers).
    financial_statements : pd.DataFrame or None
        Statement-level data with ``ticker``, ``period_type``,
        and optionally ``period_date`` columns.
    config : InvestabilityScreenConfig or None
        Screening configuration.  Defaults to developed-market
        thresholds.
    current_members : pd.Index or None
        Tickers currently in the universe for hysteresis.

    Returns
    -------
    pd.Index
        Tickers passing all investability screens.
    """
    if config is None:
        config = InvestabilityScreenConfig()

    # Start with all tickers present in fundamentals
    candidates = fundamentals.index

    # 1. Market capitalization
    if "market_cap" in fundamentals.columns:
        mcap = fundamentals["market_cap"].dropna()
        passed = apply_screen(mcap, config.market_cap, current_members)
        candidates = candidates.intersection(passed)

    # 2. ADDV 12-month
    if len(price_history) > 0 and len(volume_history) > 0:
        addv_12 = compute_addv(price_history, volume_history, window=252)
        addv_12 = addv_12.reindex(candidates).dropna()
        passed = apply_screen(addv_12, config.addv_12m, current_members)
        candidates = candidates.intersection(passed)

    # 3. ADDV 3-month
    if len(price_history) > 0 and len(volume_history) > 0:
        addv_3 = compute_addv(price_history, volume_history, window=63)
        addv_3 = addv_3.reindex(candidates).dropna()
        passed = apply_screen(addv_3, config.addv_3m, current_members)
        candidates = candidates.intersection(passed)

    # 4. Trading frequency
    if len(volume_history) > 0:
        freq = compute_trading_frequency(volume_history, window=252)
        freq = freq.reindex(candidates).dropna()
        passed = apply_screen(freq, config.trading_frequency, current_members)
        candidates = candidates.intersection(passed)

    # 5. Price filter (region-dependent)
    if "current_price" in fundamentals.columns:
        prices = fundamentals["current_price"].reindex(candidates).dropna()
        price_thresh = (
            config.price_us
            if config.exchange_region == ExchangeRegion.US
            else config.price_europe
        )
        passed = apply_screen(prices, price_thresh, current_members)
        candidates = candidates.intersection(passed)

    # 6. Trading history
    listing_age = compute_listing_age(price_history)
    listing_age = listing_age.reindex(candidates).dropna()
    candidates = candidates.intersection(
        listing_age.index[listing_age >= config.min_trading_history]
    )

    # 7. IPO seasoning
    candidates = candidates.intersection(
        listing_age.index[listing_age >= config.min_ipo_seasoning]
    )

    # 8. Financial statement availability
    if financial_statements is not None and len(financial_statements) > 0:
        annual_counts = count_financial_statements(
            financial_statements, period_type="annual"
        )
        quarterly_counts = count_financial_statements(
            financial_statements, period_type="quarterly"
        )

        annual_ok = annual_counts.reindex(candidates, fill_value=0)
        quarterly_ok = quarterly_counts.reindex(candidates, fill_value=0)

        # Pass if either annual OR quarterly threshold is met
        has_enough = (
            (annual_ok >= config.min_annual_reports)
            | (quarterly_ok >= config.min_quarterly_reports)
        )
        candidates = candidates.intersection(
            has_enough.index[has_enough]
        )

    return candidates
