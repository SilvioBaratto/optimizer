"""FastAPI router for portfolio CRUD and broker sync endpoints."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import database_manager, get_db
from app.repositories.portfolio_repository import PortfolioRepository
from app.schemas.portfolio import (
    BrokerAccountResponse,
    BrokerPositionResponse,
    PortfolioCreate,
    PortfolioListResponse,
    PortfolioResponse,
    SnapshotCreate,
    SnapshotListResponse,
    SnapshotResponse,
    SyncJobResponse,
    SyncProgressResponse,
)
from app.services._progress import make_progress
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services.broker_sync_service import sync_portfolio
from app.services.trading212.client import Trading212Client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

_sync_job_service = BackgroundJobService(
    job_type="portfolio_sync",
    session_factory=database_manager.get_session,
)


# ---------------------------------------------------------------------------
# Background sync worker
# ---------------------------------------------------------------------------


def _run_sync(job_id: str, portfolio_id: str, mode: str) -> None:
    """Execute T212 sync in a background thread with its own DB session."""
    _sync_job_service.update_job(job_id, status="running")

    try:
        client = Trading212Client(
            api_key=settings.trading_212_api_key or "",
            api_secret=settings.trading_212_secret_key or "",
            mode=mode,
        )
        on_progress = make_progress(job_id, _sync_job_service)

        with database_manager.get_session() as session:
            result = sync_portfolio(
                client,
                uuid.UUID(portfolio_id),
                session,
                on_progress=on_progress,
            )

        _sync_job_service.update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result={
                "positions_synced": result.positions_synced,
                "positions_removed": result.positions_removed,
                "account_synced": result.account_synced,
                "orders_fetched": result.orders_fetched,
                "dividends_fetched": result.dividends_fetched,
                "errors": result.errors or [],
            },
        )
        logger.info("Sync job %s completed for portfolio %s", job_id, portfolio_id)

    except Exception as e:
        logger.error("Sync job %s failed: %s", job_id, e)
        _sync_job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_portfolio_repository(db: Session = Depends(get_db)) -> PortfolioRepository:
    return PortfolioRepository(db)


def get_t212_client() -> Trading212Client:
    if not settings.trading_212_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trading212 API key not configured",
        )
    return Trading212Client(
        api_key=settings.trading_212_api_key,
        api_secret=settings.trading_212_secret_key,
        mode=settings.trading_212_mode,
    )


# ---------------------------------------------------------------------------
# Portfolio CRUD
# ---------------------------------------------------------------------------


@router.get("/", response_model=PortfolioListResponse)
def list_portfolios(
    repo: PortfolioRepository = Depends(get_portfolio_repository),
) -> PortfolioListResponse:
    """List all active portfolios."""
    portfolios = repo.get_all_active()
    items = [PortfolioResponse.model_validate(p) for p in portfolios]
    return PortfolioListResponse(items=items, total=len(items))


@router.post(
    "/",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_portfolio(
    body: PortfolioCreate,
    db: Session = Depends(get_db),
) -> PortfolioResponse:
    """Create a new portfolio (idempotent by name via get_or_create)."""
    repo = PortfolioRepository(db)
    portfolio = repo.get_or_create(
        name=body.name,
        description=body.description,
        currency=body.currency,
        benchmark_ticker=body.benchmark_ticker,
    )
    db.commit()
    return PortfolioResponse.model_validate(portfolio)


@router.get("/{name}", response_model=PortfolioResponse)
def get_portfolio(
    name: str,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
) -> PortfolioResponse:
    """Get a portfolio by name."""
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )
    return PortfolioResponse.model_validate(portfolio)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@router.post(
    "/{name}/snapshots",
    response_model=SnapshotResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_snapshot(
    name: str,
    body: SnapshotCreate,
    db: Session = Depends(get_db),
) -> SnapshotResponse:
    """Create a portfolio snapshot from optimizer result or manual input."""
    repo = PortfolioRepository(db)
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snap = repo.create_snapshot(
        portfolio_id=portfolio.id,
        snapshot_date=body.snapshot_date,
        snapshot_type=body.snapshot_type,
        weights=body.weights,
        sector_mapping=body.sector_mapping,
        summary=body.summary,
        optimizer_config=body.optimizer_config,
        turnover=body.turnover,
    )
    db.commit()
    return SnapshotResponse.model_validate(snap)


@router.get("/{name}/snapshots", response_model=SnapshotListResponse)
def list_snapshots(
    name: str,
    limit: int = Query(default=100, ge=1, le=500),
    repo: PortfolioRepository = Depends(get_portfolio_repository),
) -> SnapshotListResponse:
    """List portfolio snapshots ordered by date descending."""
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snaps = repo.get_snapshots(portfolio.id, limit=limit)
    items = [SnapshotResponse.model_validate(s) for s in snaps]
    return SnapshotListResponse(items=items, total=len(items))


@router.get("/{name}/snapshots/latest", response_model=SnapshotResponse)
def get_latest_snapshot(
    name: str,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
) -> SnapshotResponse:
    """Get the most recent snapshot for a portfolio."""
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    snap = repo.get_latest_snapshot(portfolio.id)
    if snap is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No snapshots found for this portfolio",
        )
    return SnapshotResponse.model_validate(snap)


# ---------------------------------------------------------------------------
# Broker sync
# ---------------------------------------------------------------------------


@router.post(
    "/{name}/sync",
    response_model=SyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_sync(
    name: str,
    db: Session = Depends(get_db),
    _client: Trading212Client = Depends(get_t212_client),
) -> SyncJobResponse:
    """Start a Trading212 sync job for a portfolio. Returns 202 + job_id."""
    repo = PortfolioRepository(db)
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    try:
        job_id = _sync_job_service.create_job(portfolio_name=name)
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    _sync_job_service.start_background(
        target=_run_sync,
        args=(job_id, str(portfolio.id), settings.trading_212_mode or "live"),
    )

    return SyncJobResponse(
        job_id=job_id,
        status="pending",
        message=f"Sync started for portfolio '{name}'.",
    )


@router.get("/{name}/sync/{job_id}", response_model=SyncProgressResponse)
def get_sync_status(name: str, job_id: str) -> SyncProgressResponse:
    """Poll a sync job for progress and result."""
    job = _sync_job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sync job '{job_id}' not found",
        )

    return SyncProgressResponse(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        result=job.get("result"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# Positions and account
# ---------------------------------------------------------------------------


@router.get("/{name}/positions", response_model=list[BrokerPositionResponse])
def get_positions(
    name: str,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
) -> list[BrokerPositionResponse]:
    """List synced broker positions for a portfolio."""
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    positions = repo.get_positions(portfolio.id)
    return [BrokerPositionResponse.model_validate(p) for p in positions]


@router.get("/{name}/account", response_model=BrokerAccountResponse)
def get_account(
    name: str,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
) -> BrokerAccountResponse:
    """Get the latest broker account cash snapshot for a portfolio."""
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )

    account = repo.get_latest_account_snapshot(portfolio.id)
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account snapshot found; run sync first",
        )
    return BrokerAccountResponse.model_validate(account)
