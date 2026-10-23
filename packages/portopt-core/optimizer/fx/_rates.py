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


def _as_datetime_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    """Coerce a date axis to a ``DatetimeIndex``, rejecting anything else.

    The as-of fill orders the union of the rate and price axes.  An integer or
    mixed-type axis cannot be ordered, and an unordered union would let a future
    print fill earlier rows, so it is rejected instead of guessed at.  Integers
    are refused outright because pandas would read them as epoch nanoseconds.
    """
    if isinstance(index, pd.DatetimeIndex):
        converted = index
    elif isinstance(index, pd.MultiIndex) or pd.api.types.is_numeric_dtype(index):
        raise DataError(
            f"{name} must be a date-like index, got {type(index).__name__} "
            f"with dtype {index.dtype}."
        )
    else:
        try:
            converted = pd.DatetimeIndex(index)
        except (TypeError, ValueError) as exc:
            raise DataError(f"{name} must be a date-like index: {exc}") from exc
    if converted.hasnans:
        raise DataError(f"{name} contains missing dates.")
    return converted


def _ffill_within_business_days(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Forward-fill ``frame`` while the last observation is at most ``limit`` old.

    Age is counted in business days between row labels rather than in rows, so
    it means the same thing on a daily, weekly or month-end calendar.  ``None``
    leaves the age uncapped.
    """
    if limit is None:
        return frame.ffill()
    days = _wall_clock(frame.index).to_numpy().astype("datetime64[D]")
    ordinal = np.busday_count(np.datetime64("1970-01-01"), days)[:, np.newaxis]
    observed = np.where(frame.notna().to_numpy(), ordinal, np.nan)
    last_seen = pd.DataFrame(observed).ffill().to_numpy()
    return frame.ffill().where(ordinal - last_seen <= limit)


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
    fill_limit: int | None = 5,
) -> pd.DataFrame:
    """Align FX rate DataFrame to a price calendar.

    Each price date takes the latest rate printed on or before it, as long as
    that print is at most ``fill_limit`` business days old (weekends, holidays).
    When the two indexes do not share a timezone (one naive, or two different
    zones), rates are matched on each index's local calendar date.

    Args:
        fx_rates: FX rates indexed by date with currency codes as columns.
            Each column holds the rate from that currency to the base
            currency (units of base per one unit of foreign).
        price_index: Target date index (from the price DataFrame).  Date
            strings are parsed; the returned frame keeps the labels as given.
        fill_limit: Maximum age, in business days, of a rate that may be
            carried forward; ``0`` uses same-day rates only and ``None`` puts
            no cap on the age.

    Returns:
        Aligned FX rates on the ``price_index``.  Zero, negative and
        non-finite rates are treated as missing (``NaN``) before filling.

    Raises:
        DataError: If ``fx_rates`` is empty, ``fill_limit`` is negative, or
            either index is not date-like (for example integers) or holds
            missing dates.
    """
    if fx_rates.empty:
        raise DataError("fx_rates DataFrame is empty.")
    if fill_limit is not None and fill_limit < 0:
        raise DataError(f"fill_limit must be non-negative or None, got {fill_limit}.")

    rates = _mask_invalid_rates(fx_rates)
    rates = rates.set_axis(_as_datetime_index(rates.index, "fx_rates index"), axis=0)
    lookup = _as_datetime_index(price_index, "price_index")
    # Labels in different timezones never match one-for-one, and in the as-of
    # union an offset zone would silently pair a bar with the previous day's
    # print.  Daily bars are identified by their local date, which is also what
    # a naive rate frame holds.
    if lookup.tz != rates.index.tz:
        rates = rates.set_axis(_wall_clock(rates.index), axis=0)
        lookup = cast(pd.DatetimeIndex, _wall_clock(lookup))

    # Rates printed on dates the price calendar lacks (holidays, the day before
    # the first bar) must still seed the fill, so the fill runs on the union.
    on_union = rates.reindex(rates.index.union(lookup))
    aligned = _ffill_within_business_days(on_union, fill_limit).reindex(lookup)
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
