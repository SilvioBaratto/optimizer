"""FastAPI router for walk-forward validation endpoints.

Endpoints:
  POST /validate/walk-forward  — start background CV job, returns 202 + job_id
  GET  /validate/walk-forward/{job_id} — poll status and progress

The route is named after its primary CV strategy (walk-forward) per the spec,
but the underlying service supports multiple CV strategies via
``request.cv_type`` (walk_forward | cpcv | multiple_randomized).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.database import database_manager
from app.schemas.validation import (
    ValidateJobResponse,
    ValidateProgress,
    ValidateRequest,
)
from app.services._progress import make_progress
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services.validation_service import run_validation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validate", tags=["Validation"])

# Module-level job service — follows macro_regime.py / backtest.py pattern.
_job_service = BackgroundJobService(
    job_type="validate",
    session_factory=database_manager.get_session,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_walk_forward_bg(job_id: str, request: ValidateRequest) -> None:
    """Execute walk-forward validation in a daemon thread with its own DB session."""
    _job_service.update_job(job_id, status="running")
    try:
        with database_manager.get_session() as session:
            result = run_validation(
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                cv_type=request.cv_type.value,
                cv_config=request.cv_config,
                optimizer_type=request.optimizer_type,
                optimizer_config=request.optimizer_config,
                session=session,
                on_progress=make_progress(job_id, _job_service),
            )
        _job_service.update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result=result,
        )
        logger.info("Walk-forward validation job %s completed", job_id)
    except Exception as exc:
        logger.error("Walk-forward validation job %s failed: %s", job_id, exc)
        _job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/walk-forward",
    response_model=ValidateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_walk_forward(request: ValidateRequest) -> JSONResponse:
    """Start a walk-forward validation background job.

    Returns 202 + job_id immediately. Poll GET /validate/walk-forward/{job_id}.
    """
    try:
        job_id = _job_service.create_job(current_fold=0, total_folds=0)
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    _job_service.start_background(
        target=_run_walk_forward_bg,
        args=(job_id, request),
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "pending",
            "message": (
                f"Walk-forward validation started. "
                f"Poll GET /validate/walk-forward/{job_id} for progress."
            ),
        },
    )


@router.get(
    "/walk-forward/{job_id}",
    response_model=ValidateProgress,
)
def get_walk_forward_status(job_id: str) -> ValidateProgress:
    """Poll the status and progress of a walk-forward validation job."""
    job = _job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Walk-forward validation job '{job_id}' not found",
        )

    return ValidateProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        current_fold=job.get("current_fold", 0),
        total_folds=job.get("total_folds", 0),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )
