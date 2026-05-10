"""FastAPI router for reference-index seeding endpoints.

Reference indices (e.g. SPY, QQQ, IWM, sector ETFs) are benchmark instruments
used by the dashboard, portfolio attribution, and HMM regime detection.  They
are *not* members of the T212 tradable universe, so they are ingested here via
a dedicated background job rather than through the universe builder.
"""

import logging
import threading
from concurrent.futures import CancelledError
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import database_manager, get_db
from app.repositories.dashboard_repository import DashboardRepository
from app.repositories.portfolio_repository import PortfolioRepository
from app.schemas.reference_index import (
    ReferenceIndexConfiguredResponse,
    ReferenceIndexSeedJobResponse,
    ReferenceIndexSeedProgress,
    ReferenceIndexSeedRequest,
    ReferenceIndexStatusItem,
    ReferenceIndexStatusResponse,
)
from app.services._progress import make_cancellable_progress
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services.reference_index_seeder import seed_reference_indices
from app.services.yfinance import YFinanceClient, get_yfinance_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reference-indices", tags=["Reference Indices"])

# Shared job service instance for this router.
_job_service = BackgroundJobService(
    job_type="reference_index_seed",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)


# ---------------------------------------------------------------------------
# Background seed worker
# ---------------------------------------------------------------------------


def _run_seed(
    job_id: str,
    request: ReferenceIndexSeedRequest,
    yf_client: YFinanceClient,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    """Thin wrapper managing job lifecycle around the seeder service."""
    if cancel_event is None:
        cancel_event = threading.Event()
    on_progress = make_cancellable_progress(job_id, _job_service, cancel_event)
    _job_service.update_job(job_id, status="running")
    try:
        seed_reference_indices(request.tickers, yf_client, on_progress=on_progress)
    except CancelledError:
        return
    except Exception as exc:
        logger.error("Reference-index seed job %s failed: %s", job_id, exc)
        _job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_yf_client() -> YFinanceClient:
    return get_yfinance_client()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/seed",
    response_model=ReferenceIndexSeedJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_seed(
    request: ReferenceIndexSeedRequest,
    yf_client: YFinanceClient = Depends(_get_yf_client),
) -> ReferenceIndexSeedJobResponse:
    """Start a background job that seeds reference-index instruments."""
    try:
        job_id = _job_service.create_job(current_ticker="")
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    _job_service.start_background(
        target=_run_seed,
        args=(job_id, request, yf_client),
    )

    return ReferenceIndexSeedJobResponse(
        job_id=job_id,
        status="pending",
        message=(
            "Reference-index seed started. "
            "Poll GET /reference-indices/seed/{job_id} for progress."
        ),
    )


@router.get("/seed/{job_id}", response_model=ReferenceIndexSeedProgress)
def get_seed_status(job_id: str) -> ReferenceIndexSeedProgress:
    """Poll the status and progress of a reference-index seed job."""
    job = _job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return ReferenceIndexSeedProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get(
    "/status",
    response_model=ReferenceIndexStatusResponse,
    response_model_by_alias=True,
    summary="Coverage diagnostics for every configured benchmark",
)
def get_reference_index_status(
    db: Session = Depends(get_db),
) -> ReferenceIndexStatusResponse:
    """Return per-ticker coverage without triggering a fetch.

    A ticker is *healthy* when it has an instrument row, at least one price
    row, and a latest price within ``settings.scheduler_benchmark_stale_days``
    of today.
    """
    from datetime import date, timedelta

    tickers = list(_configured_tickers(db))
    repo = DashboardRepository(db)
    coverage = repo.get_benchmark_coverage(tickers)

    today = date.today()
    stale_days = settings.scheduler_benchmark_stale_days
    stale_cutoff = today - timedelta(days=stale_days)

    items: list[ReferenceIndexStatusItem] = []
    healthy = missing = stale = 0
    for ticker in tickers:
        rows, latest = coverage.get(ticker, (0, None))
        instrument_exists = ticker in coverage and (rows > 0 or latest is not None)
        is_stale = latest is None or latest < stale_cutoff
        if rows == 0:
            missing += 1
        elif is_stale:
            stale += 1
        else:
            healthy += 1
        items.append(
            ReferenceIndexStatusItem(
                ticker=ticker,
                instrument_exists=instrument_exists or rows > 0,
                price_rows=rows,
                latest_price_date=latest,
                is_stale=is_stale,
            )
        )

    return ReferenceIndexStatusResponse(
        items=items,
        total=len(items),
        healthy_count=healthy,
        missing_count=missing,
        stale_count=stale,
        stale_days=stale_days,
    )


@router.get(
    "/configured",
    response_model=ReferenceIndexConfiguredResponse,
    response_model_by_alias=True,
    summary="Union of settings.benchmark_tickers and active portfolio benchmarks",
)
def get_configured_tickers(
    db: Session = Depends(get_db),
) -> ReferenceIndexConfiguredResponse:
    """Return every benchmark the scheduler should keep current.

    Union of:
      * ``settings.benchmark_tickers`` (operator-controlled list)
      * every distinct ``portfolios.benchmark_ticker`` among active portfolios
    """
    portfolio_repo = PortfolioRepository(db)
    portfolio_tickers = portfolio_repo.get_distinct_benchmark_tickers()
    settings_tickers = list(settings.benchmark_tickers)
    union = sorted({*settings_tickers, *portfolio_tickers})

    return ReferenceIndexConfiguredResponse(
        tickers=union,
        settings_tickers=sorted(settings_tickers),
        portfolio_tickers=portfolio_tickers,
    )


def _configured_tickers(db: Session) -> list[str]:
    """Return the union used by ``/status`` (mirrors ``/configured``)."""
    portfolio_repo = PortfolioRepository(db)
    portfolio_tickers = portfolio_repo.get_distinct_benchmark_tickers()
    return sorted({*settings.benchmark_tickers, *portfolio_tickers})
