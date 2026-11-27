"""Instrument-universe build, driven by the scheduler and the CLI.

Discovers the tradable instrument universe from the yfinance Screener
(:class:`YFinanceUniverseSource`) and persists it to ``exchanges`` /
``instruments``.

Every other ingestion step iterates the ``instruments`` table, so this is the
head of the pipeline — a stale universe silently caps what yfinance fetches.
"""

from __future__ import annotations

import logging
from typing import Any

from portopt_db.repositories.universe.universe_repository import UniverseRepository

from app.database import database_manager
from app.schemas.universe.build_request import UniverseBuildRequest
from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import get_yfinance_client
from app.services.universe.builder.builder import BuildProgress, UniverseBuilder
from app.services.universe.builder.config import UniverseBuilderConfig
from app.services.universe.yfinance_source import (
    PassThroughTickerMapper,
    YFinanceUniverseSource,
)

logger = logging.getLogger(__name__)


def run_universe_build(
    request: UniverseBuildRequest,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Discover and persist the instrument universe from the yfinance Screener.

    Instruments come from ``yf.screen`` (see :class:`YFinanceUniverseSource`).
    No investability filtering: every
    classified STOCK/ETF is admitted (screening is a downstream fund-layer
    concern). ISIN is not populated here (D14).

    Args:
        request: ``UniverseBuildRequest`` with ``exchanges``, ``max_workers``.
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``exchanges_saved``, ``instruments_saved``,
        ``total_processed``, ``filter_stats``, and ``errors``.
    """
    config = UniverseBuilderConfig()

    def _forward(p: BuildProgress) -> None:
        on_progress(
            current=p.current,
            total=p.total,
            current_exchange=p.current_exchange,
            current_stock=p.current_stock,
        )

    with database_manager.get_session() as session:
        repository = UniverseRepository(session)
        source = YFinanceUniverseSource(
            screener=get_yfinance_client().screener,
            config=config,
            member_lookup=repository.get_active_listings,
        )
        builder = UniverseBuilder(
            config=config,
            api_client=source,
            ticker_mapper=PassThroughTickerMapper(),
            repository=repository,
            max_workers=request.max_workers,
            only_exchanges=request.exchanges,
            progress_callback=_forward,
        )
        result = builder.build()
        session.commit()

    logger.info(
        "universe_build: %d exchanges, %d instruments (%d processed, %d errors)",
        result.exchanges_saved,
        result.instruments_saved,
        result.total_processed,
        len(result.errors),
    )

    summary = {
        "exchanges_saved": result.exchanges_saved,
        "instruments_saved": result.instruments_saved,
        "total_processed": result.total_processed,
        "filter_stats": result.filter_stats,
        "errors": result.errors,
    }
    on_progress(status="completed", result=summary, errors=result.errors)
    return summary
