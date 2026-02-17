"""FastAPI router for Trading212 universe endpoints."""

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, database_manager
from app.repositories.universe_repository import UniverseRepository
from app.services.trading212.config import UniverseBuilderConfig
from app.services.trading212.client import Trading212Client
from app.services.trading212.cache.ticker_cache import TickerMappingCache
from app.services.trading212.ticker_mapper import YFinanceTickerMapper
from app.services.trading212.filters import (
    FilterPipelineImpl,
    MarketCapFilter,
    PriceFilter,
    LiquidityFilter,
    DataCoverageFilter,
    HistoricalDataFilter,
)
from app.services.trading212.builder import UniverseBuilder, BuildProgress
from app.schemas.trading212 import (
    UniverseBuildRequest,
    ExchangeResponse,
    InstrumentResponse,
    InstrumentListResponse,
    BuildResultResponse,
    BuildJobResponse,
    BuildProgressResponse,
    CacheStatsResponse,
    UniverseStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/universe", tags=["Universe"])


# ---------------------------------------------------------------------------
# In-memory build job store
# ---------------------------------------------------------------------------

_build_jobs: Dict[str, Dict[str, Any]] = {}
_build_jobs_lock = threading.Lock()


def _set_job(build_id: str, data: Dict[str, Any]) -> None:
    with _build_jobs_lock:
        _build_jobs[build_id] = data


def _get_job(build_id: str) -> Dict[str, Any] | None:
    with _build_jobs_lock:
        return _build_jobs.get(build_id, {}).copy() if build_id in _build_jobs else None


def _update_job(build_id: str, **kwargs: Any) -> None:
    with _build_jobs_lock:
        if build_id in _build_jobs:
            _build_jobs[build_id].update(kwargs)


# ---------------------------------------------------------------------------
# Background build worker
# ---------------------------------------------------------------------------

def _run_build(
    build_id: str,
    request: UniverseBuildRequest,
    config: UniverseBuilderConfig,
    client: Trading212Client,
    cache: TickerMappingCache,
) -> None:
    """Execute universe build in a background thread with its own DB session."""
    _update_job(build_id, status="running")

    try:
        with database_manager.get_session() as session:
            repo = UniverseRepository(session)
            mapper = YFinanceTickerMapper(config=config, cache=cache)

            pipeline = FilterPipelineImpl()
            if not request.skip_filters:
                pipeline.add_filter(MarketCapFilter(config=config))
                pipeline.add_filter(PriceFilter(config=config))
                pipeline.add_filter(LiquidityFilter(config=config))
                pipeline.add_filter(DataCoverageFilter(config=config))
                pipeline.add_filter(HistoricalDataFilter(config=config))

            def on_progress(p: BuildProgress) -> None:
                _update_job(
                    build_id,
                    current=p.current,
                    total=p.total,
                    current_exchange=p.current_exchange,
                    current_stock=p.current_stock,
                )

            builder = UniverseBuilder(
                config=config,
                api_client=client,
                ticker_mapper=mapper,
                filter_pipeline=pipeline,
                repository=repo,
                max_workers=request.max_workers,
                skip_filters=request.skip_filters,
                only_exchanges=request.exchanges,
                progress_callback=on_progress,
            )

            result = builder.build()
            session.commit()

            _update_job(
                build_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                result={
                    "exchanges_saved": result.exchanges_saved,
                    "instruments_saved": result.instruments_saved,
                    "total_processed": result.total_processed,
                    "filter_stats": result.filter_stats,
                    "errors": result.errors,
                },
            )
            logger.info(
                "Build %s completed: %d exchanges, %d instruments",
                build_id,
                result.exchanges_saved,
                result.instruments_saved,
            )

    except Exception as e:
        logger.error("Build %s failed: %s", build_id, e)
        _update_job(
            build_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------

def get_universe_config() -> UniverseBuilderConfig:
    return UniverseBuilderConfig()


def get_trading212_client() -> Trading212Client:
    if not settings.trading_212_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trading212 API key not configured",
        )
    return Trading212Client(
        api_key=settings.trading_212_api_key,
        mode=settings.trading_212_mode,
    )


def get_ticker_cache() -> TickerMappingCache:
    return TickerMappingCache()


def get_universe_repository(db: Session = Depends(get_db)) -> UniverseRepository:
    return UniverseRepository(db)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=UniverseStatsResponse)
def get_stats(repo: UniverseRepository = Depends(get_universe_repository)):
    """Get exchange and instrument counts."""
    return UniverseStatsResponse(
        exchange_count=repo.get_exchange_count(),
        instrument_count=repo.get_instrument_count(),
    )


@router.get("/exchanges", response_model=list[ExchangeResponse])
def list_exchanges(repo: UniverseRepository = Depends(get_universe_repository)):
    """List all exchanges."""
    return repo.get_exchanges()


@router.get("/instruments", response_model=InstrumentListResponse)
def list_instruments(
    exchange: str | None = Query(default=None, description="Filter by exchange name"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    repo: UniverseRepository = Depends(get_universe_repository),
):
    """List instruments with optional exchange filter and pagination."""
    items = repo.get_instruments(exchange_name=exchange, skip=skip, limit=limit)
    total = repo.get_instrument_count()
    return InstrumentListResponse(items=items, total=total)


@router.post(
    "/build",
    response_model=BuildJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def build_universe(
    request: UniverseBuildRequest,
    config: UniverseBuilderConfig = Depends(get_universe_config),
    client: Trading212Client = Depends(get_trading212_client),
    cache: TickerMappingCache = Depends(get_ticker_cache),
):
    """Start a universe build in the background. Returns a build ID to poll for progress."""
    # Reject if another build is already running
    with _build_jobs_lock:
        for job in _build_jobs.values():
            if job.get("status") in ("pending", "running"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A build is already in progress (id={job['build_id']})",
                )

    build_id = str(uuid.uuid4())
    _set_job(build_id, {
        "build_id": build_id,
        "status": "pending",
        "current": 0,
        "total": 0,
        "current_exchange": "",
        "current_stock": "",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "result": None,
        "error": None,
    })

    thread = threading.Thread(
        target=_run_build,
        args=(build_id, request, config, client, cache),
        daemon=True,
    )
    thread.start()

    return BuildJobResponse(
        build_id=build_id,
        status="pending",
        message="Build started. Poll GET /universe/build/{build_id} for progress.",
    )


@router.get("/build/{build_id}", response_model=BuildProgressResponse)
def get_build_status(build_id: str):
    """Poll the status and progress of a universe build."""
    job = _get_job(build_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Build {build_id} not found",
        )

    result = None
    if job.get("result"):
        result = BuildResultResponse(**job["result"])

    return BuildProgressResponse(
        build_id=job["build_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        current_exchange=job.get("current_exchange", ""),
        current_stock=job.get("current_stock", ""),
        result=result,
        error=job.get("error"),
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
def get_cache_stats(cache: TickerMappingCache = Depends(get_ticker_cache)):
    """Get ticker mapping cache statistics."""
    stats = cache.get_stats()
    return CacheStatsResponse(**stats)


@router.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
def clear_cache(cache: TickerMappingCache = Depends(get_ticker_cache)):
    """Clear the ticker mapping cache."""
    cache.clear()
