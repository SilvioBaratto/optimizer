"""Database pre-flight health checks for the research pipeline (issue #519).

The orchestrator :func:`run_db_preflight` runs four independent checks against
the optimizer database and aborts with a single aggregated :class:`RuntimeError`
when any of them fail.  It is invoked from
:func:`research.stock_selection_pipeline.load_data` immediately after
``DatabaseManager.initialize()`` and BEFORE any data assembly so that
silent empty-frame fallbacks (Bug #237 regression class) are made
impossible to ship.

Each check is a pure helper accepting an open SQLAlchemy ``Session``
and returning ``str | None`` — ``None`` on pass, an actionable failure
message on fail.  Checks do NOT short-circuit; the orchestrator collects
every failure so the operator can fix all DB issues in a single session.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_MIN_INSTRUMENTS: int = 2_500
_MAX_PRICE_GAP_DAYS: int = 7
_MAX_FRED_GAP_DAYS: int = 30
_MIN_COUNTRY_COVERAGE: float = 0.95
_REQUIRED_FRED_SERIES: tuple[str, ...] = (
    "DGS3MO",
    "DGS2",
    "DGS10",
    "BAMLH0A0HYM2",
    "VIXCLS",
    "USRECDM",
)


# ---------------------------------------------------------------------------
# DB-manager protocol — keeps this module decoupled from app.database
# ---------------------------------------------------------------------------


class _DbManagerLike(Protocol):
    def get_session(self) -> AbstractContextManager[Session]:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_date(value: object) -> datetime.date | None:
    """Coerce a SQL MAX(date) result to ``datetime.date`` (None on empty)."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_universe_coverage(session: Session) -> str | None:
    """Verify ``instruments`` table holds at least the required floor."""
    result = session.execute(text("SELECT COUNT(*) FROM instruments")).scalar()
    count = int(result or 0)
    if count >= _MIN_INSTRUMENTS:
        return None
    return f"Universe coverage: {count} instruments found (need ≥ {_MIN_INSTRUMENTS})."


def _check_price_staleness(session: Session, today: datetime.date) -> str | None:
    """Verify ``MAX(price_history.date)`` is within the freshness window."""
    raw = session.execute(text("SELECT MAX(date) FROM price_history")).scalar()
    max_date = _coerce_date(raw)
    if max_date is None:
        return "Price history: table is empty."
    gap = (today - max_date).days
    if gap <= _MAX_PRICE_GAP_DAYS:
        return None
    return (
        f"Price history: max date {max_date.isoformat()} is {gap} days "
        f"stale (need ≤ {_MAX_PRICE_GAP_DAYS})."
    )


def _check_fred_freshness(session: Session, today: datetime.date) -> str | None:
    """Verify every required FRED series has a recent observation."""
    stmt = text(
        "SELECT series_id, MAX(date) FROM fred_observations "
        "WHERE series_id IN :ids GROUP BY series_id"
    ).bindparams(bindparam("ids", expanding=True))
    rows = session.execute(stmt, {"ids": list(_REQUIRED_FRED_SERIES)}).fetchall()

    last_seen: dict[str, datetime.date | None] = {
        str(sid): _coerce_date(d) for sid, d in rows
    }

    issues: list[str] = []
    for sid in _REQUIRED_FRED_SERIES:
        max_date = last_seen.get(sid)
        if max_date is None:
            issues.append(f"  - {sid}: missing")
            continue
        gap = (today - max_date).days
        if gap > _MAX_FRED_GAP_DAYS:
            issues.append(f"  - {sid}: last={max_date.isoformat()} ({gap} days stale)")

    if not issues:
        return None
    return f"FRED freshness (need ≤ {_MAX_FRED_GAP_DAYS} days):\n" + "\n".join(issues)


def _check_country_coverage(session: Session) -> str | None:
    """Verify country profile coverage of active instruments ≥ floor."""
    row = session.execute(
        text(
            "SELECT "
            "  SUM(CASE WHEN tp.country IS NOT NULL THEN 1 ELSE 0 END) "
            "    AS with_country, "
            "  COUNT(*) AS total "
            "FROM instruments i "
            "LEFT JOIN ticker_profiles tp ON tp.instrument_id = i.id "
            "WHERE i.delisted_at IS NULL"
        )
    ).first()
    if row is None:
        return "Country coverage: query returned no rows."

    with_country = int(row[0] or 0)
    total = int(row[1] or 0)
    if total == 0:
        return "Country coverage: no active instruments to evaluate."

    ratio = with_country / total
    if ratio >= _MIN_COUNTRY_COVERAGE:
        return None
    return (
        f"Country coverage: {ratio:.2f} "
        f"({with_country}/{total} active instruments) "
        f"(need ≥ {_MIN_COUNTRY_COVERAGE:.2f})."
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _iter_check_results(session: Session, today: datetime.date) -> Iterator[str]:
    """Yield non-None failure messages from every check."""
    candidates: list[str | None] = [
        _check_universe_coverage(session),
        _check_price_staleness(session, today=today),
        _check_fred_freshness(session, today=today),
        _check_country_coverage(session),
    ]
    yield from (msg for msg in candidates if msg)


def run_db_preflight(
    db_manager: _DbManagerLike,
    today: datetime.date | None = None,
) -> None:
    """Run all DB pre-flight checks; raise on any failure.

    Parameters
    ----------
    db_manager : DatabaseManager-like
        Anything exposing a ``get_session()`` context manager yielding a
        synchronous SQLAlchemy ``Session``.
    today : datetime.date | None
        Reference "today" used by staleness checks.  Defaults to
        ``datetime.date.today()``.

    Raises
    ------
    RuntimeError
        With every failure joined on newlines.  All checks always run —
        no short-circuit — so a single error message can guide the
        operator through the full set of fixes.
    """
    reference_date = today if today is not None else datetime.date.today()
    with db_manager.get_session() as session:
        failures = list(_iter_check_results(session, reference_date))

    if not failures:
        return

    report = "DB pre-flight failed:\n" + "\n".join(failures)
    logger.error(report)
    raise RuntimeError(report)
