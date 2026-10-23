"""FX rate pair construction and alignment utilities."""

from __future__ import annotations

import logging
from typing import TypeVar, cast

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError
from optimizer.fx._minor_units import normalize_currency_code

logger = logging.getLogger(__name__)

_RatesT = TypeVar("_RatesT", pd.Series, pd.DataFrame)


def _mask_invalid_rates(rates: _RatesT) -> _RatesT:
    """Replace non-positive and non-finite FX quotes with ``NaN``.

    An exchange rate is a price ratio, so a zero, negative or infinite print is
    bad data.  Left in place it prices assets at zero or inf without tripping
    the converter's NaN warning.
    """
    return rates.where((rates > 0) & (rates < np.inf))


def build_fx_pair_ticker(
    from_ccy: str,
    to_ccy: str,
    *,
    cross_via_usd: bool = True,
) -> str | tuple[str, str] | None:
    """Build yfinance FX pair ticker(s) for converting *from_ccy* to *to_ccy*.

    Args:
        from_ccy: Source currency ISO code (e.g. ``"GBP"``).
        to_ccy: Target currency ISO code (e.g. ``"EUR"``).
        cross_via_usd: When ``True`` and neither currency is USD, return a tuple
            of two USD-based tickers for cross-rate computation.

    Returns:
        ``None`` when ``from_ccy == to_ccy``.
        A single ticker string when one side is USD, or when
        ``cross_via_usd`` is ``False``.  It is always ``<FROM><TO>=X``, which
        Yahoo quotes as units of TO per one unit of FROM, so the series needs
        no inversion before use as a base-per-foreign rate.
        A tuple ``(FROM/USD, TO/USD)`` when crossing via USD; both legs are
        USD-per-unit quotes to feed `compute_cross_rate`.

    Note:
        Minor-unit codes (``GBp``, ``ZAc``, ``ILA``, ...) are resolved to their
        major currency first, so ``build_fx_pair_ticker("GBp", "EUR")`` builds
        the GBP pair — there is no ``GBp`` FX market to quote.
    """
    from_ccy, _ = normalize_currency_code(from_ccy)
    to_ccy, _ = normalize_currency_code(to_ccy)

    if from_ccy == to_ccy:
        return None

    if "USD" in (from_ccy, to_ccy) or not cross_via_usd:
        return f"{from_ccy}{to_ccy}=X"

    return (f"{from_ccy}USD=X", f"{to_ccy}USD=X")


def _wall_clock(index: pd.Index) -> pd.Index:
    """Return ``index`` without its timezone, keeping each label's local date."""
    if getattr(index, "tz", None) is None:
        return index
    return cast(pd.DatetimeIndex, index).tz_localize(None)


def compute_cross_rate(
    fx_from_usd: pd.Series,
    fx_to_usd: pd.Series,
) -> pd.Series:
    """Compute cross rate FROM/TO via USD.

    Given FROM/USD and TO/USD rates, the cross rate FROM/TO is::

        FROM/TO = FROM/USD / TO/USD

    One unit of FROM currency buys ``FROM/TO`` units of TO currency.

    Args:
        fx_from_usd: FROM/USD exchange rate series.
        fx_to_usd: TO/USD exchange rate series.

    Returns:
        Cross rate series (FROM per 1 unit of TO is the reciprocal;
        this returns units-of-TO per 1 unit of FROM).  Dates where either leg
        is zero, negative or non-finite are ``NaN``.
    """
    aligned_from, aligned_to = fx_from_usd.align(fx_to_usd, join="inner")
    return _mask_invalid_rates(aligned_from) / _mask_invalid_rates(aligned_to)


def align_fx_rates(
    fx_rates: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """Align FX rate DataFrame to a price calendar.

    Reindexes FX rates to the ``price_index`` and forward-fills gaps
    (weekends, holidays) up to ``fill_limit`` days.  When exactly one of the
    two indexes is tz-aware, rates are matched on local calendar date.

    Args:
        fx_rates: FX rates indexed by date with currency codes as columns.
            Each column holds the rate from that currency to the base
            currency (units of base per one unit of foreign).
        price_index: Target date index (from the price DataFrame).
        fill_limit: Maximum consecutive NaN days to forward-fill; ``0``
            disables filling so only exact-date rates are used.

    Returns:
        Aligned FX rates on the ``price_index``.  Zero, negative and
        non-finite rates are treated as missing (``NaN``) before filling.

    Raises:
        DataError: If ``fx_rates`` is empty.
    """
    if fx_rates.empty:
        raise DataError("fx_rates DataFrame is empty.")

    rates = _mask_invalid_rates(fx_rates)
    lookup = price_index
    # A tz-aware index never matches a tz-naive one label-for-label, so the whole
    # frame would come back NaN.  Daily bars are identified by their local date,
    # which is also what a naive rate frame holds.
    if (getattr(price_index, "tz", None) is None) != (
        getattr(rates.index, "tz", None) is None
    ):
        rates = rates.set_axis(_wall_clock(rates.index), axis=0)
        lookup = cast(pd.DatetimeIndex, _wall_clock(price_index))

    aligned = rates.reindex(lookup)
    # pandas rejects limit=0, which here is the valid "exact-date rates only".
    if fill_limit > 0:
        aligned = aligned.ffill(limit=fill_limit)
    # Callers multiply against the price frame, which aligns on labels.
    aligned.index = price_index
    return aligned


def invert_fx_rates(
    fx_rates: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Invert FX-rate quote direction (reciprocal).

    Converts a *units-of-base per one unit-of-foreign* quote into a
    *units-of-foreign per one unit-of-base* quote (or vice versa) by
    taking the element-wise reciprocal.  Zeros map to ``NaN`` rather than
    raising or producing ``inf``.

    Args:
        fx_rates: FX rate quotes.

    Returns:
        Reciprocal of ``fx_rates`` with the same shape/labels.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inverted = fx_rates.rdiv(1.0)
    return inverted.replace([np.inf, -np.inf], np.nan)


def required_fx_currencies(
    currency_map: dict[str, str],
    base_currency: str,
) -> set[str]:
    """Return the set of foreign currencies that need FX data.

    Args:
        currency_map: Ticker → ISO currency code mapping.
        base_currency: The portfolio base currency.

    Returns:
        The set of *major* currencies in ``currency_map`` that differ from
        ``base_currency``.  Minor-unit codes are resolved to their major
        currency (``GBp``/``ZAc``/``ILA`` -> ``GBP``/``ZAR``/``ILS``) so the
        caller fetches the FX pairs that actually trade.
    """
    base, _ = normalize_currency_code(base_currency)
    majors = {normalize_currency_code(ccy)[0] for ccy in currency_map.values()}
    return {ccy for ccy in majors if ccy != base}
