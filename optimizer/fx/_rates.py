"""FX rate pair construction and alignment utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError

logger = logging.getLogger(__name__)


def build_fx_pair_ticker(
    from_ccy: str,
    to_ccy: str,
    *,
    cross_via_usd: bool = True,
) -> str | tuple[str, str] | None:
    """Build yfinance FX pair ticker(s) for converting *from_ccy* to *to_ccy*.

    Parameters
    ----------
    from_ccy : str
        Source currency ISO code (e.g. ``"GBP"``).
    to_ccy : str
        Target currency ISO code (e.g. ``"EUR"``).
    cross_via_usd : bool
        When ``True`` and neither currency is USD, return a tuple of
        two USD-based tickers for cross-rate computation.

    Returns
    -------
    str or tuple[str, str] or None
        ``None`` when ``from_ccy == to_ccy``.
        A single ticker string when one side is USD.
        A tuple ``(FROM/USD, TO/USD)`` when crossing via USD.
    """
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()

    if from_ccy == to_ccy:
        return None

    if from_ccy == "USD":
        return f"{to_ccy}USD=X"
    if to_ccy == "USD":
        return f"{from_ccy}USD=X"

    if cross_via_usd:
        return (f"{from_ccy}USD=X", f"{to_ccy}USD=X")

    return f"{from_ccy}{to_ccy}=X"


def compute_cross_rate(
    fx_from_usd: pd.Series,
    fx_to_usd: pd.Series,
) -> pd.Series:
    """Compute cross rate FROM/TO via USD.

    Given FROM/USD and TO/USD rates, the cross rate FROM/TO is::

        FROM/TO = FROM/USD / TO/USD

    One unit of FROM currency buys ``FROM/TO`` units of TO currency.

    Parameters
    ----------
    fx_from_usd : pd.Series
        FROM/USD exchange rate series.
    fx_to_usd : pd.Series
        TO/USD exchange rate series.

    Returns
    -------
    pd.Series
        Cross rate series (FROM per 1 unit of TO is the reciprocal;
        this returns units-of-TO per 1 unit of FROM).
    """
    aligned_from, aligned_to = fx_from_usd.align(fx_to_usd, join="inner")
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = aligned_from / aligned_to
    return cross


def align_fx_rates(
    fx_rates: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """Align FX rate DataFrame to a price calendar.

    Reindexes FX rates to the ``price_index`` and forward-fills gaps
    (weekends, holidays) up to ``fill_limit`` days.

    Parameters
    ----------
    fx_rates : pd.DataFrame
        FX rates indexed by date with currency codes as columns.
        Each column holds the rate from that currency to the base
        currency (units of base per one unit of foreign).
    price_index : pd.DatetimeIndex
        Target date index (from the price DataFrame).
    fill_limit : int
        Maximum consecutive NaN days to forward-fill.

    Returns
    -------
    pd.DataFrame
        Aligned FX rates on the ``price_index``.

    Raises
    ------
    DataError
        If ``fx_rates`` is empty.
    """
    if fx_rates.empty:
        raise DataError("fx_rates DataFrame is empty.")

    aligned = fx_rates.reindex(price_index)
    aligned = aligned.ffill(limit=fill_limit)
    return aligned


def required_fx_currencies(
    currency_map: dict[str, str],
    base_currency: str,
) -> set[str]:
    """Return the set of foreign currencies that need FX data.

    Parameters
    ----------
    currency_map : dict[str, str]
        Ticker → ISO currency code mapping.
    base_currency : str
        The portfolio base currency.

    Returns
    -------
    set[str]
        Currencies in ``currency_map`` that differ from
        ``base_currency``.
    """
    base = base_currency.upper()
    return {ccy.upper() for ccy in currency_map.values() if ccy.upper() != base}
