"""FastAPI router for backtest endpoints.

Always runs as a background job — POST returns 202 + job_id.
Follows the macro_regime.py / yfinance_data.py background job pattern.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import database_manager, get_db
from app.metrics import time_endpoint
from app.repositories.execution.execution_repository import ExecutionRepository
from app.schemas.backtest.backtest import (
    BacktestProgressResponse,
    BacktestRequest,
    BacktestRunResponse,
)
from app.services.backtest.backtest_service import run_and_persist
from app.services.jobs.background_job import (
    BackgroundJobService,
    JobAlreadyRunningError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtest"])

# Module-level job service — follows macro_regime.py / portfolio.py pattern.
_job_service = BackgroundJobService(
    job_type="backtest",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_backtest_bg(
    job_id: str,
    run_id: str,
    request: BacktestRequest,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    """Execute backtest in a daemon thread with its own DB session."""
    _job_service.update_job(job_id, status="running")
    try:
        with database_manager.get_session() as session:
            run_and_persist(
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                pipeline_config=request.pipeline_config,
                session=session,
                run_id=uuid.UUID(run_id),
            )
        _job_service.update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Backtest job %s completed (run=%s)", job_id, run_id)
    except Exception as exc:
        logger.error("Backtest job %s failed: %s", job_id, exc)
        _job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        _fail_run(run_id, str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
)
def start_backtest(
    request: BacktestRequest,
    db: Session = Depends(get_db),
):
    """Start a backtest as a background job. Returns 202 + job_id + run_id."""
    with time_endpoint("backtest"):
        repo = ExecutionRepository(db)
        pending_run = repo.create_backtest_run(
            {
                "status": "pending",
                "config": request.pipeline_config,
                "equity_curve": {},
                "drawdowns": {},
                "monthly_returns": {},
                "yearly_returns": {},
                "rolling_metrics": {},
                "turnover_history": {},
                "summary_stats": {},
            }
        )
        db.commit()

        run_id = str(pending_run.id)

        try:
            job_id = _job_service.create_job()
        except JobAlreadyRunningError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        _job_service.start_background(
            target=_run_backtest_bg,
            args=(job_id, run_id, request),
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "jobId": job_id,
                "runId": run_id,
                "status": "pending",
                "message": f"Backtest started. Poll GET /backtest/{job_id} for progress.",
            },
        )


@router.get(
    "/runs/{run_id}",
    response_model=BacktestRunResponse,
    response_model_by_alias=True,
    summary="Fetch a persisted backtest run by its primary-key UUID",
)
def get_backtest_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> BacktestRunResponse:
    """Return the :class:`BacktestRun` row with matching ``run_id`` (issue #464).

    Serves ``backtest_runs`` data directly — distinct from
    ``GET /backtest/{job_id}`` which polls the in-memory background job
    progress. The frontend uses the ``runId`` returned by
    ``POST /backtest`` to retrieve completed metrics here.
    """
    run = ExecutionRepository(db).get_backtest_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run '{run_id}' not found",
        )
    return BacktestRunResponse.model_validate(run)


@router.get("/{job_id}", response_model=BacktestProgressResponse)
def get_backtest_status(job_id: str) -> BacktestProgressResponse:
    """Poll a backtest job for status and progress."""
    job = _job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest job '{job_id}' not found",
        )

    return BacktestProgressResponse(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fail_run(run_id: str, error: str) -> None:
    """Mark BacktestRun as failed; best-effort, logs on error."""
    try:
        with database_manager.get_session() as session:
            repo = ExecutionRepository(session)
            repo.update_backtest_status(
                uuid.UUID(run_id), "failed", {"error_message": error}
            )
            session.commit()
    except Exception:
        logger.warning("Could not update BacktestRun %s to failed", run_id)
