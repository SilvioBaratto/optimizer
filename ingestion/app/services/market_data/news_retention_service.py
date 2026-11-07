"""Retention of the per-ticker news store (``ticker_news``).

Every sweep upserts the scraped body of each ticker's articles and nothing else
removes them, so the table grows with the universe size forever. This service
deletes the articles that aged out of ``settings.news_retention_days``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository

from app.config import settings
from app.database import database_manager
from app.services._shared import ProgressCallback, _noop

logger = logging.getLogger(__name__)

# Rows per committed delete. Each row carries a full article body, so this bounds
# how long one transaction holds its row locks.
NEWS_PRUNE_BATCH_SIZE = 1000


def run_news_retention(
    *,
    retention_days: int | None = None,
    batch_size: int = NEWS_PRUNE_BATCH_SIZE,
    now: datetime | None = None,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Delete the news articles older than the retention window.

    Rows go in batches of ``batch_size``, each in its own session and committed
    before the next starts, so a large backlog never becomes one long
    transaction and a failure keeps what earlier batches already removed.
    Progress is reported after every batch, which also renews the step's lease
    and aborts the loop if the orphan reaper has cancelled the job.

    Args:
        retention_days: Window to keep; defaults to
            ``settings.news_retention_days``. Zero or less keeps everything and
            opens no database session.
        batch_size: Maximum rows per committed delete.
        now: Reference instant the window is measured back from; defaults to the
            current UTC time.
        on_progress: Callback receiving progress and terminal-status updates.

    Returns:
        Dict with ``deleted`` (rows removed), ``batches`` (delete statements
        issued) and ``cutoff`` (ISO timestamp, ``None`` when disabled).

    Raises:
        sqlalchemy.exc.SQLAlchemyError: A batch failed; earlier batches stay
            committed.
        concurrent.futures.CancelledError: The job was reaped mid-run.
    """
    days = settings.news_retention_days if retention_days is None else retention_days
    started = datetime.now(timezone.utc)
    if days <= 0:
        result: dict[str, Any] = {"deleted": 0, "batches": 0, "cutoff": None}
        logger.info("news_retention: disabled (retention_days=%d)", days)
        on_progress(status="completed", finished_at=started.isoformat(), result=result)
        return result

    cutoff = (now or started) - timedelta(days=days)
    deleted = 0
    batches = 0
    while True:
        with database_manager.get_session() as session:
            removed = YFinanceRepository(session).prune_ticker_news(
                cutoff, batch_size=batch_size
            )
            session.commit()
        batches += 1
        deleted += removed
        on_progress(current=deleted)
        if removed < batch_size:
            break

    result = {"deleted": deleted, "batches": batches, "cutoff": cutoff.isoformat()}
    logger.info(
        "news_retention: deleted %d article(s) older than %s in %d batch(es)",
        deleted,
        cutoff.date().isoformat(),
        batches,
    )
    on_progress(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        result=result,
    )
    return result


__all__ = ["NEWS_PRUNE_BATCH_SIZE", "run_news_retention"]
