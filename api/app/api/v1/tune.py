"""FastAPI router for hyperparameter tuning endpoints.

POST /api/v1/tune  — launch a background grid or randomized search job.
GET  /api/v1/tune/{job_id} — poll job progress.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import CancelledError

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.database import database_manager
from app.metrics import time_endpoint
from app.schemas.base_job import AsyncJobCreateResponse, AsyncJobProgress
from app.schemas.tuning import TuneRequest, TuneResult
from app.services._progress import make_cancellable_progress
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services.tuning_service import run_tune

_SUPPORTED_OPTIMIZER_TYPES: frozenset[str] = frozenset({"mean_risk"})

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tune", tags=["Tuning"])

_tune_job_service = BackgroundJobService(
    job_type="tune",
    session_factory=database_manager.get_session,
    heartbeat_cadence_seconds=settings.scheduler_heartbeat_cadence_seconds,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_tune_bg(
    job_id: str,
    request: TuneRequest,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    """Thin wrapper managing job lifecycle around the tuning service."""
    if cancel_event is None:
        cancel_event = threading.Event()
    on_progress = make_cancellable_progress(job_id, _tune_job_service, cancel_event)
    _tune_job_service.update_job(job_id, status="running")
    try:
        with database_manager.get_session() as session:
            result: TuneResult = run_tune(session, request, on_progress=on_progress)
        _tune_job_service.update_job(
            job_id, status="completed", result=result.model_dump()
        )
    except CancelledError:
        return
    except Exception as exc:
        logger.error("Tune job %s failed: %s", job_id, exc)
        _tune_job_service.update_job(job_id, status="failed", error=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=202, response_model=AsyncJobCreateResponse)
def post_tune(request: TuneRequest) -> AsyncJobCreateResponse:
    """Launch a background hyperparameter search job."""
    with time_endpoint("tune"):
        _validate_optimizer_type(request.optimizer_type)
        try:
            job_id = _tune_job_service.create_job()
        except JobAlreadyRunningError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        _tune_job_service.start_background(_run_tune_bg, args=(job_id, request))
        return AsyncJobCreateResponse(job_id=job_id, status="pending", message="")


@router.get("/{job_id}", response_model=AsyncJobProgress)
def get_tune_job(job_id: str) -> AsyncJobProgress:
    """Poll progress for a background tune job."""
    job = _tune_job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return AsyncJobProgress(**job)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_optimizer_type(optimizer_type: str) -> None:
    """Raise 422 for unknown optimizer_type strings."""
    if optimizer_type not in _SUPPORTED_OPTIMIZER_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unknown optimizer_type: {optimizer_type!r}. "
                f"Valid types: {sorted(_SUPPORTED_OPTIMIZER_TYPES)}"
            ),
        )
