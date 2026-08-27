"""Regional market-summary ingestion (SPEC B3).

Fetches ``yf.Market(id).summary`` for the eight documented market identifiers
and upserts the flattened index/quote rows. Market-wide → its own scheduler
step. Each market is fetched + committed independently.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import YFinanceClient
from app.services.market_data.yfinance.market.market_summary import MARKET_IDENTIFIERS

logger = logging.getLogger(__name__)


def run_market_summary_fetch(
    yf_client: YFinanceClient,
    *,
    markets: tuple[str, ...] = MARKET_IDENTIFIERS,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Fetch + persist regional market summaries for every identifier."""
    from app.database import database_manager
    from app.repositories.market_data.market_summary_repository import (
        MarketSummaryRepository,
    )

    now = datetime.now(timezone.utc)
    as_of = now.date()

    with database_manager.get_session() as session:
        repo = MarketSummaryRepository(session)
        on_progress(total=len(markets))
        counts: dict[str, int] = {}
        errors: list[str] = []

        for idx, market in enumerate(markets, 1):
            on_progress(current=idx, current_market=market)
            try:
                rows = yf_client.market.fetch_summary(market)
                counts[market] = repo.upsert_summaries(market, as_of, rows or [])
                session.commit()
            except Exception as e:  # one market failure must not sink the rest
                logger.warning("Failed market summary %s: %s", market, e)
                errors.append(f"{market}: {e}")
                session.rollback()

        result = {
            "counts": counts,
            "rows_total": sum(counts.values()),
            "error_count": len(errors),
        }
        logger.info("Market-summary fetch done: %s", counts)
        on_progress(
            status="completed",
            finished_at=now.isoformat(),
            errors=errors,
            result=result,
        )

    return result
