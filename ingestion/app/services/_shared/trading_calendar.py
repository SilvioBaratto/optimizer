"""Exchange-specific trading calendar utilities for data validation."""

import logging
import re
from datetime import date

import exchange_calendars as xcals

logger = logging.getLogger(__name__)

# Map exchange names (as stored in DB) to ISO 10383 MIC codes
EXCHANGE_NAME_TO_MIC = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS",
    "London Stock Exchange": "XLON",
    "Euronext Paris": "XPAR",
    "Deutsche Börse Xetra": "XFRA",
}


def parse_period_years(period: str) -> int | None:
    """Extract number of years from a yfinance period string.

    Returns None for non-year periods ("6mo", "max", "1d", etc.).
    """
    match = re.fullmatch(r"(\d+)y", period)
    return int(match.group(1)) if match else None


def get_expected_trading_sessions(
    exchange_name: str,
    period: str,
    reference_date: date | None = None,
) -> int | None:
    """Compute expected trading sessions for an exchange over a period.

    Returns None when validation should be skipped (unknown exchange,
    non-year period, etc.).
    """
    years = parse_period_years(period)
    if years is None:
        return None

    mic = EXCHANGE_NAME_TO_MIC.get(exchange_name)
    if mic is None:
        return None

    if reference_date is None:
        reference_date = date.today()

    # Compute start date, handling Feb 29 edge case
    try:
        start = reference_date.replace(year=reference_date.year - years)
    except ValueError:
        # Feb 29 in a leap year → fall back to Feb 28
        start = reference_date.replace(year=reference_date.year - years, day=28)

    try:
        cal = xcals.get_calendar(mic)
        # Clamp to calendar bounds
        cal_start = (
            cal.first_session.date()
            if hasattr(cal.first_session, "date")
            else cal.first_session
        )
        cal_end = (
            cal.last_session.date()
            if hasattr(cal.last_session, "date")
            else cal.last_session
        )
        start = max(start, cal_start)
        end = min(reference_date, cal_end)
        if start >= end:
            return None
        sessions = cal.sessions_in_range(start.isoformat(), end.isoformat())
        return len(sessions)
    except Exception:
        logger.warning(
            "Failed to compute sessions for %s (MIC=%s)",
            exchange_name,
            mic,
            exc_info=True,
        )
        return None


def has_sufficient_history(
    row_count: int,
    exchange_name: str | None,
    period: str,
    tolerance: float = 0.95,
) -> tuple[bool, int | None, int | None]:
    """Check whether fetched row count meets expected trading sessions.

    Returns:
        (sufficient, expected, minimum) where:
        - sufficient: True if data passes validation or validation was skipped
        - expected: expected number of sessions (None when skipped)
        - minimum: minimum required rows (None when skipped)
    """
    if exchange_name is None:
        return (True, None, None)

    expected = get_expected_trading_sessions(exchange_name, period)
    if expected is None:
        return (True, None, None)

    minimum = int(expected * tolerance)
    return (row_count >= minimum, expected, minimum)
