"""Trading 212 universe build, driven by the scheduler and the CLI.

Discovers the tradable instrument universe from the Trading 212 API, maps each
instrument to a yfinance ticker, runs it through the quality filter pipeline,
and persists the survivors to ``exchanges`` / ``instruments``.

Every other ingestion step iterates the ``instruments`` table, so this is the
head of the pipeline — a stale universe silently caps what yfinance fetches.
"""

from __future__ import annotations

import logging
from typing import Any

from app.database import database_manager
from app.repositories.universe.universe_repository import UniverseRepository
from app.schemas.universe.trading212 import UniverseBuildRequest
from app.services._shared import ProgressCallback, _noop
from app.services.universe.trading212.builder import BuildProgress, UniverseBuilder
from app.services.universe.trading212.cache.ticker_cache import TickerMappingCache
from app.services.universe.trading212.client import Trading212Client
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.filters import (
    DataCoverageFilter,
    FilterPipelineImpl,
    HistoricalDataFilter,
    LiquidityFilter,
    MarketCapFilter,
    PriceFilter,
)
from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper

logger = logging.getLogger(__name__)


class Trading212NotConfiguredError(RuntimeError):
    """Raised when a universe build is requested without a Trading 212 API key."""


def build_trading212_client() -> Trading212Client:
    """Construct a Trading 212 client from settings.

    Raises:
        Trading212NotConfiguredError: when ``TRADING_212_API_KEY`` is unset.
    """
    from app.config import settings

    if not settings.trading_212_api_key:
        raise Trading212NotConfiguredError(
            "TRADING_212_API_KEY is not set — cannot build the instrument universe"
        )
    return Trading212Client(
        api_key=settings.trading_212_api_key,
        api_secret=settings.trading_212_secret_key,
        mode=settings.trading_212_mode,
    )


def run_universe_build(
    request: UniverseBuildRequest,
    client: Trading212Client,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Discover, filter, and persist the Trading 212 instrument universe.

    Args:
        request: ``UniverseBuildRequest`` with ``exchanges``, ``skip_filters``,
            ``max_workers``.
        client: Configured Trading 212 client (see :func:`build_trading212_client`).
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``exchanges_saved``, ``instruments_saved``,
        ``total_processed``, ``filter_stats``, and ``errors``.
    """
    config = UniverseBuilderConfig()
    cache = TickerMappingCache()

    pipeline = FilterPipelineImpl()
    if not request.skip_filters:
        pipeline.add_filter(MarketCapFilter(config=config))
        pipeline.add_filter(PriceFilter(config=config))
        pipeline.add_filter(LiquidityFilter(config=config))
        pipeline.add_filter(DataCoverageFilter(config=config))
        pipeline.add_filter(HistoricalDataFilter(config=config))

    def _forward(p: BuildProgress) -> None:
        on_progress(
            current=p.current,
            total=p.total,
            current_exchange=p.current_exchange,
            current_stock=p.current_stock,
        )

    with database_manager.get_session() as session:
        builder = UniverseBuilder(
            config=config,
            api_client=client,
            ticker_mapper=YFinanceTickerMapper(config=config, cache=cache),
            filter_pipeline=pipeline,
            repository=UniverseRepository(session),
            max_workers=request.max_workers,
            skip_filters=request.skip_filters,
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
