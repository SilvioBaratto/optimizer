"""Multi-format date parsing utilities."""

import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Month abbreviation mapping for parsing reference dates like "Dec 2024"
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
    """Parse a reference date string into a :class:`date` object.

    Supported formats:
    - ``date`` objects (returned as-is)
    - ``"Mon YYYY"`` (e.g. ``"Dec 2024"``) → ``date(2024, 12, 1)``
    - ``"MM/YY"`` or ``"MM/ YY"`` IlSole-style → ``date(20YY, MM, 1)``
    - ``"Mon/DD"`` → current year
    - ISO format ``"YYYY-MM-DD"``
    - US/EU date formats ``"M/D/YYYY"``, ``"D/M/YYYY"``

    Returns:
        Parsed date or ``None`` if parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()

    # Try "Mon YYYY" format (e.g. "Dec 2024", "Jan 2025")
    parts = text.split()
    if len(parts) == 2:
        month_str, year_str = parts
        month = _MONTH_ABBR.get(month_str[:3].lower())
        if month is not None:
            try:
                return date(int(year_str), month, 1)
            except (ValueError, TypeError):
                pass

    # Try "MM/ YY" or "MM/YY" format (e.g. "12/ 25", "01/26") — IlSole style
    # Normalize by removing internal spaces: "12/ 25" → "12/25"
    normalized = text.replace(" ", "")
    if "/" in normalized:
        slash_parts = normalized.split("/")
        if len(slash_parts) == 2:
            left, right = slash_parts[0].strip(), slash_parts[1].strip()
            # MM/YY — both sides are numeric, right side is 2-digit year
            if left.isdigit() and right.isdigit() and len(right) == 2:
                try:
                    month = int(left)
                    year = 2000 + int(right)
                    return date(year, month, 1)
                except (ValueError, TypeError):
                    pass
            # Mon/DD — left is month abbreviation, right is day
            month = _MONTH_ABBR.get(left[:3].lower())
            if month is not None:
                try:
                    day = int(right)
                    return date(date.today().year, month, day)
                except (ValueError, TypeError):
                    pass

    # Try ISO format (e.g. "2024-12-01")
    try:
        return date.fromisoformat(text)
    except (ValueError, TypeError):
        pass

    # Try parsing via datetime for other formats (e.g. "1/15/2025")
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except (ValueError, TypeError):
            continue

    logger.debug("Could not parse reference_date: %r", value)
    return None
