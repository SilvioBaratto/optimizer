"""Market-wide calendar ingestion (SPEC B2).

Fetches the four ``yf.Calendars`` rollups (earnings / IPO / splits / economic
events) and upserts them. Market-wide + forward-looking → its own scheduler
step. Each calendar is fetched and persisted independently so one failure does
not sink the others.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import YFinanceClient

logger = logging.getLogger(__name__)


def run_calendars_fetch(
    yf_client: YFinanceClient,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Fetch + persist all four market-wide calendars."""
    from portopt_db.repositories.market_data.calendars_repository import (
        CalendarsRepository,
    )

    from app.database import database_manager

    now = datetime.now(timezone.utc)

    with database_manager.get_session() as session:
        repo = CalendarsRepository(session)
        counts: dict[str, int] = {}
        errors: list[str] = []

        steps = (
            ("earnings", yf_client.calendars.fetch_earnings, repo.upsert_earnings),
            ("ipos", yf_client.calendars.fetch_ipos, repo.upsert_ipos),
            ("splits", yf_client.calendars.fetch_splits, repo.upsert_splits),
            (
                "economic_events",
                yf_client.calendars.fetch_economic_events,
                repo.upsert_economic_events,
            ),
        )
        on_progress(total=len(steps))

        for idx, (name, fetch, upsert) in enumerate(steps, 1):
            on_progress(current=idx, current_calendar=name)
            try:
                rows = fetch()
                counts[name] = upsert(rows) if rows else 0
                session.commit()
            except Exception as e:  # one calendar failure must not sink the rest
                logger.warning("Failed calendar %s: %s", name, e)
                errors.append(f"{name}: {e}")
                session.rollback()

        result = {"counts": counts, "error_count": len(errors)}
        logger.info("Calendars fetch done: %s", counts)
        on_progress(
            status="completed",
            finished_at=now.isoformat(),
            errors=errors,
            result=result,
        )

    return result
