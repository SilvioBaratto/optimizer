"""Value-coercion utilities shared by the repositories."""

import logging
from datetime import date, datetime
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


def parse_reference_date(value: Any) -> date | None:
    """Parse a multi-format reference date value into a date.

    Accepts date objects, month-year strings ("Dec 2024"), IlSole compact
    dates ("12/25" or "12/ 25"), abbreviated month-day ("Mon/DD"), ISO dates
    ("2024-12-01"), and US/EU slash formats ("M/D/YYYY", "D/M/YYYY").

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
                    day = int(right)
                    return date(date.today().year, month, day)
                except (ValueError, TypeError):
                    pass

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
