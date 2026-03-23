"""Currency normalization for minor-unit listing currencies.

yfinance returns market data in the **listing currency**, which for many
exchanges is a minor unit (e.g. GBX = pence for LSE, ILA = agorot for
TASE, ZAC = cents for JSE).  This module provides pure functions to
detect these currencies and convert to their major-unit equivalents so
that screening thresholds and factor computations receive consistent
values.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from app.models.yfinance_data import TickerProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minor-currency divisor map
# ---------------------------------------------------------------------------

#: Maps minor-unit ISO / yfinance currency codes to the divisor that
#: converts them to their major-unit equivalent.
#:
#: Examples:
#:   GBX (pence)  → GBP: ÷ 100
#:   GBp (pence)  → GBP: ÷ 100   (yfinance sometimes uses "GBp")
#:   ILA (agorot) → ILS: ÷ 100
#:   ZAC (cents)  → ZAR: ÷ 100
MINOR_CURRENCY_DIVISORS: dict[str, float] = {
    "GBX": 100.0,
    "GBp": 100.0,
    "ILA": 100.0,
    "ZAC": 100.0,
}

#: Maps minor-unit codes to their major-unit ISO equivalent.
_MINOR_TO_MAJOR: dict[str, str] = {
    "GBX": "GBP",
    "GBp": "GBP",
    "ILA": "ILS",
    "ZAC": "ZAR",
}

#: Currency priority for cross-listed ticker deduplication.
#: Lower rank = higher priority.  When a ticker appears on multiple
#: exchanges, the listing with the lowest rank "wins" deduplication.
CURRENCY_DEDUP_PRIORITY: dict[str, int] = {
    "USD": 0,
    "GBP": 1,
    "EUR": 2,
    "GBX": 3,
    "GBp": 3,  # yfinance variant of GBX — same rank
    "CHF": 4,
    "JPY": 5,
    "CAD": 6,
    "AUD": 7,
    "HKD": 8,
}

# Columns in a fundamentals DataFrame that are denominated in local
# currency and must be divided by the minor-unit divisor.
_CURRENCY_DENOMINATED_COLUMNS: list[str] = [
    "market_cap",
    "enterprise_value",
    "current_price",
    "book_value",
    "total_cash",
    "total_debt",
    "operating_cashflow",
    "free_cashflow",
    "total_revenue",
    "ebitda",
    "gross_profits",
    "trailing_eps",
    "forward_eps",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_to_major_currency(currency_code: str) -> str:
    """Convert a minor-unit currency code to its major-unit equivalent.

    >>> normalize_to_major_currency("GBX")
    'GBP'
    >>> normalize_to_major_currency("USD")
    'USD'
    """
    return _MINOR_TO_MAJOR.get(currency_code, currency_code)


def currency_dedup_rank(currency_code: str | None) -> int:
    """Return the deduplication priority rank for a currency code.

    Lower rank is preferred (USD=0 outranks EUR=2).  Unknown or missing
    currencies receive rank 99 so that a known listing always wins.
    """
    if currency_code is None:
        return 99
    return CURRENCY_DEDUP_PRIORITY.get(currency_code, 99)


def build_currency_map(profiles: list[TickerProfile]) -> dict[str, str]:
    """Build ``{yfinance_ticker: currency_code}`` from loaded profiles.

    Uses ``Instrument.currency_code`` when available, falling back to
    ``TickerProfile.currency``.  Returns only entries where a currency
    could be determined.

    Parameters
    ----------
    profiles : list[TickerProfile]
        TickerProfile rows with ``.instrument`` eagerly loaded
        (``joinedload``).
    """
    currency_map: dict[str, str] = {}
    for profile in profiles:
        instrument = profile.instrument
        if instrument is None or not instrument.yfinance_ticker:
            continue
        ticker = instrument.yfinance_ticker
        # Prefer Instrument.currency_code; fall back to TickerProfile.currency
        ccy = instrument.currency_code or profile.currency
        if ccy:
            currency_map[ticker] = ccy
    return currency_map


def normalize_fundamentals(
    df: pd.DataFrame,
    currency_map: dict[str, str],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Divide minor-unit columns by their divisor and normalize the currency map.

    Parameters
    ----------
    df : pd.DataFrame
        Fundamentals DataFrame indexed by yfinance ticker.
    currency_map : dict[str, str]
        ``{ticker: raw_currency_code}`` (e.g. ``{"BARC.L": "GBX"}``).
    columns : list[str] or None
        Columns to normalize.  Defaults to ``_CURRENCY_DENOMINATED_COLUMNS``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        - Normalized DataFrame (minor-unit values divided by divisor).
        - Normalized currency map (minor codes replaced with major codes,
          e.g. ``"GBX" → "GBP"``).
    """
    if columns is None:
        columns = _CURRENCY_DENOMINATED_COLUMNS

    # Group tickers by divisor to batch-apply
    divisor_groups: dict[float, list[str]] = {}
    for ticker, ccy in currency_map.items():
        divisor = MINOR_CURRENCY_DIVISORS.get(ccy)
        if divisor is not None and ticker in df.index:
            divisor_groups.setdefault(divisor, []).append(ticker)

    if not divisor_groups:
        # Nothing to normalize — still normalize the currency map
        normalized_map = {
            t: normalize_to_major_currency(c) for t, c in currency_map.items()
        }
        return df, normalized_map

    df = df.copy()
    total_normalized = 0
    for divisor, tickers in divisor_groups.items():
        mask = df.index.isin(tickers)
        for col in columns:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] / divisor
        total_normalized += int(mask.sum())

    logger.info(
        "Normalized %d tickers from minor-unit currencies (divisors: %s).",
        total_normalized,
        {ccy: div for ccy, div in MINOR_CURRENCY_DIVISORS.items()
         if any(currency_map.get(t) == ccy for t in df.index)},
    )

    normalized_map = {
        t: normalize_to_major_currency(c) for t, c in currency_map.items()
    }
    return df, normalized_map


def normalize_prices(
    prices: pd.DataFrame,
    currency_map: dict[str, str],
) -> pd.DataFrame:
    """Divide minor-unit price columns by their divisor.

    Parameters
    ----------
    prices : pd.DataFrame
        ``dates × tickers`` close-price DataFrame.
    currency_map : dict[str, str]
        ``{ticker: raw_currency_code}``.

    Returns
    -------
    pd.DataFrame
        Copy with minor-unit columns divided.
    """
    divisor_groups: dict[float, list[str]] = {}
    for ticker, ccy in currency_map.items():
        divisor = MINOR_CURRENCY_DIVISORS.get(ccy)
        if divisor is not None and ticker in prices.columns:
            divisor_groups.setdefault(divisor, []).append(ticker)

    if not divisor_groups:
        return prices

    prices = prices.copy()
    for divisor, tickers in divisor_groups.items():
        prices[tickers] = prices[tickers] / divisor

    total = sum(len(t) for t in divisor_groups.values())
    logger.info("Normalized %d price columns from minor-unit currencies.", total)
    return prices
