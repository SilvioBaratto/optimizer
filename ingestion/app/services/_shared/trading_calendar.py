"""Exchange-specific trading calendar utilities for data validation."""

import logging
import re
from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd

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


_PERIOD_PATTERN = re.compile(r"([1-9]\d*)(d|wk|mo|y)")


def period_start_date(period: str, reference_date: date) -> date | None:
    """Return the first calendar date covered by a yfinance period string.

    Args:
        period: yfinance period such as ``"5y"``, ``"6mo"``, ``"2wk"``,
            ``"10d"`` or ``"ytd"``.
        reference_date: End of the window, normally today.

    Returns:
        The window's start date. Month and year offsets clamp to the end of a
        shorter month (one year before 2024-02-29 is 2023-02-28). ``None`` when the
        period has no finite start (``"max"``) or is not a recognised period.
    """
    if period == "ytd":
        return date(reference_date.year, 1, 1)
    match = _PERIOD_PATTERN.fullmatch(period)
    if match is None:
        return None
    count = int(match.group(1))
    offsets = {
        "d": pd.DateOffset(days=count),
        "wk": pd.DateOffset(weeks=count),
        "mo": pd.DateOffset(months=count),
        "y": pd.DateOffset(years=count),
    }
    offset = offsets[match.group(2)]
    return (pd.Timestamp(reference_date) - offset).date()


def get_expected_trading_sessions(
    exchange_name: str,
    period: str,
    reference_date: date | None = None,
    since: date | None = None,
) -> int | None:
    """Compute expected trading sessions for an exchange over a period.

    Args:
        exchange_name: Exchange name as stored in the database (must appear in
            ``EXCHANGE_NAME_TO_MIC`` to resolve a calendar).
        period: yfinance period string (e.g. ``"5y"``).
        reference_date: End of the period window; defaults to today.
        since: Earliest date the window may start, so an instrument listed
            after the period start is measured only over its own lifetime.
            ``None`` keeps the full period.

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

    if since is not None:
        start = max(start, since)

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
        Session dates in ascending order. Out-of-range requests are clamped to the
        calendar's own bounds (a warning is logged). Empty when the exchange is
        unknown, the range is reversed (``start > end``), the clamped range has no
        sessions, or the calendar lookup fails.
    """
    mic = EXCHANGE_NAME_TO_MIC.get(exchange_name)
    if mic is None or start > end:
        return []

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
        # Clamp to the calendar's own bounds. Without this an out-of-range date
        # makes ``sessions_in_range`` raise; the bare ``except`` would then read as
        # "no sessions" and the backfill caller would report a false success.
        clamped_start = max(start, cal_start)
        clamped_end = min(end, cal_end)
        if (clamped_start, clamped_end) != (start, end):
            logger.warning(
                "Requested range [%s, %s] trimmed to %s calendar bounds [%s, %s]",
                start,
                end,
                exchange_name,
                clamped_start,
                clamped_end,
            )
        if clamped_start > clamped_end:
            return []
        sessions = cal.sessions_in_range(
            clamped_start.isoformat(), clamped_end.isoformat()
        )
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


# Wider than any real run of consecutive closures (the longest exchange
# shutdowns are a few days), so the window always contains a session.
_LAST_SESSION_LOOKBACK_DAYS = 14


def last_completed_session(exchange_name: str | None, as_of: date) -> date:
    """Return the newest trading session strictly before *as_of*.

    Args:
        exchange_name: Exchange name as stored in the database. ``None`` or an
            exchange without a calendar falls back to the previous weekday,
            which ignores holidays and so can only err towards "a newer
            session exists".
        as_of: Reference date; its own session is excluded because it has not
            completed (and a date-ranged fetch ending on it excludes it too).

    Returns:
        The date of the most recent session before *as_of*.
    """
    day_before = as_of - timedelta(days=1)
    if exchange_name is not None:
        sessions = iter_trading_days(
            as_of - timedelta(days=_LAST_SESSION_LOOKBACK_DAYS),
            day_before,
            exchange_name,
        )
        if sessions:
            return sessions[-1]

    while day_before.weekday() >= 5:
        day_before -= timedelta(days=1)
    return day_before


def has_sufficient_history(
    row_count: int,
    exchange_name: str | None,
    period: str,
    tolerance: float = 0.95,
    since: date | None = None,
) -> tuple[bool, int | None, int | None]:
    """Check whether fetched row count meets expected trading sessions.

    Args:
        row_count: Number of price rows actually stored for this ticker.
        exchange_name: Exchange name as stored in the database; ``None``
            skips validation entirely (returns sufficient=True).
        period: yfinance period string (e.g. ``"5y"``).
        tolerance: Fraction of expected sessions required to pass
            (default 0.95 allows for minor calendar gaps).
        since: Earliest date the expected window may start, typically the
            earliest bar already stored for the instrument. When it falls
            inside the period, sessions are counted from that date instead of
            the period start, so a young listing's complete history is not
            judged against sessions that predate its IPO. Anchor it on stored
            data, never on the fetched frame: a fetch that lost its head would
            otherwise be measured only against its own surviving bars.

    Returns:
        ``(sufficient, expected, minimum)`` where *sufficient* is ``True``
        when the data passes validation or validation was skipped,
        *expected* is the scheduled session count (``None`` when skipped),
        and *minimum* is the required row threshold (``None`` when skipped).
    """
    if exchange_name is None:
        return (True, None, None)

    expected = get_expected_trading_sessions(exchange_name, period, since=since)
    if expected is None:
        return (True, None, None)

    minimum = int(expected * tolerance)
    return (row_count >= minimum, expected, minimum)
