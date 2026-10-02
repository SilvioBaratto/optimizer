"""Exchange-specific trading calendar utilities for data validation."""

import logging
import re
from datetime import date

import exchange_calendars as xcals

logger = logging.getLogger(__name__)

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

    Args:
        exchange_name: Exchange name as stored in the database (must appear in
            ``EXCHANGE_NAME_TO_MIC`` to resolve a calendar).
        period: yfinance period string (e.g. ``"5y"``).
        reference_date: End of the period window; defaults to today.

    Returns:
        Number of scheduled trading sessions in the window, or ``None``
        when validation should be skipped (unknown exchange, non-year
        period, calendar bounds issue, etc.).
    """
    years = parse_period_years(period)
    if years is None:
        return None

    mic = EXCHANGE_NAME_TO_MIC.get(exchange_name)
    if mic is None:
        return None

    if reference_date is None:
        reference_date = date.today()

    try:
        start = reference_date.replace(year=reference_date.year - years)
    except ValueError:
        # Feb 29 in a leap year → fall back to Feb 28
        start = reference_date.replace(year=reference_date.year - years, day=28)

    try:
        cal = xcals.get_calendar(mic)
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


def iter_trading_days(
    start: date,
    end: date,
    exchange_name: str = "NYSE",
) -> list[date]:
    """Return exchange trading-session dates in ``[start, end]`` inclusive.

    Backs the daily-events backfill: one digest per trading session rather than
    per calendar day. Weekends and exchange holidays are excluded by the
    ``exchange_calendars`` schedule.

    Args:
        start: Inclusive first calendar date to consider.
        end: Inclusive last calendar date to consider.
        exchange_name: Exchange whose calendar defines the sessions; must appear
            in ``EXCHANGE_NAME_TO_MIC``.

    Returns:
        Session dates in ascending order. Empty when the exchange is unknown,
        the range is reversed (``start > end``), or the calendar lookup fails.
    """
    mic = EXCHANGE_NAME_TO_MIC.get(exchange_name)
    if mic is None or start > end:
        return []

    try:
        cal = xcals.get_calendar(mic)
        sessions = cal.sessions_in_range(start.isoformat(), end.isoformat())
    except Exception:
        logger.warning(
            "Failed to list sessions for %s (MIC=%s) in [%s, %s]",
            exchange_name,
            mic,
            start,
            end,
            exc_info=True,
        )
        return []

    return [s.date() if hasattr(s, "date") else s for s in sessions]


def has_sufficient_history(
    row_count: int,
    exchange_name: str | None,
    period: str,
    tolerance: float = 0.95,
) -> tuple[bool, int | None, int | None]:
    """Check whether fetched row count meets expected trading sessions.

    Args:
        row_count: Number of price rows actually stored for this ticker.
        exchange_name: Exchange name as stored in the database; ``None``
            skips validation entirely (returns sufficient=True).
        period: yfinance period string (e.g. ``"5y"``).
        tolerance: Fraction of expected sessions required to pass
            (default 0.95 allows for minor calendar gaps).

    Returns:
        ``(sufficient, expected, minimum)`` where *sufficient* is ``True``
        when the data passes validation or validation was skipped,
        *expected* is the scheduled session count (``None`` when skipped),
        and *minimum* is the required row threshold (``None`` when skipped).
    """
    if exchange_name is None:
        return (True, None, None)

    expected = get_expected_trading_sessions(exchange_name, period)
    if expected is None:
        return (True, None, None)

    minimum = int(expected * tolerance)
    return (row_count >= minimum, expected, minimum)
