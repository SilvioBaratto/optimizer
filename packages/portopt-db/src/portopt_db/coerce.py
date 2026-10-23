"""Value-coercion utilities shared by the repositories."""

import logging
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_MONTH_ABBR = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Sources stamp quotes in their own timezone, so a date a day or so ahead of
# ours is still "now" rather than a quote from the future.
_FUTURE_SKEW = timedelta(days=2)
# Wide enough to reach the previous 29 February across a skipped leap year
# (e.g. 2096 -> 2104).
_LOOKBACK_YEARS = 8


def _infer_year(month: int, day: int) -> date | None:
    """Resolve a year-less month/day to the latest such date not after today.

    Returns None when no calendar year in the lookback window has that
    month/day (e.g. day 99).
    """
    today = date.today()
    latest = today + _FUTURE_SKEW
    for year in range(today.year + 1, today.year - _LOOKBACK_YEARS, -1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate <= latest:
            return candidate
    return None


def parse_reference_date(value: Any) -> date | None:
    """Parse a multi-format reference date value into a date.

    Accepts date objects, month-year strings ("Dec 2024"), IlSole compact
    dates ("12/25" or "12/ 25"), abbreviated month-day ("Mon/DD"), ISO dates
    ("2024-12-01"), and US/EU slash formats ("M/D/YYYY", "D/M/YYYY").

    "Mon/DD" carries no year, so it resolves to the latest occurrence that is
    not after today (allowing a couple of days of source-timezone skew):
    "Dec/31" read in early January is last year's, not a date eleven months
    ahead.

    Args:
        value: Raw value from an external data source; may be a date, a
            string in any of the supported formats, or None/non-string.

    Returns:
        Parsed date, or None if value is None, empty, or unparseable.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()

    parts = text.split()
    if len(parts) == 2:
        month_str, year_str = parts
        month = _MONTH_ABBR.get(month_str[:3].lower())
        if month is not None:
            try:
                return date(int(year_str), month, 1)
            except (ValueError, TypeError):
                pass

    # IlSole publishes compact MM/YY dates with an optional space after the slash.
    normalized = text.replace(" ", "")
    if "/" in normalized:
        slash_parts = normalized.split("/")
        if len(slash_parts) == 2:
            left, right = slash_parts[0].strip(), slash_parts[1].strip()
            if left.isdigit() and right.isdigit() and len(right) == 2:
                try:
                    month = int(left)
                    year = 2000 + int(right)
                    return date(year, month, 1)
                except (ValueError, TypeError):
                    pass
            month = _MONTH_ABBR.get(left[:3].lower())
            if month is not None:
                try:
                    resolved = _infer_year(month, int(right))
                except (ValueError, TypeError):
                    resolved = None
                if resolved is not None:
                    return resolved

    try:
        return date.fromisoformat(text)
    except (ValueError, TypeError):
        pass

    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except (ValueError, TypeError):
            continue

    logger.debug("Could not parse reference_date: %r", value)
    return None


__all__ = ["parse_reference_date"]
