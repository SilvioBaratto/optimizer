"""FastAPI router for portfolio optimization endpoints.

Supports synchronous execution (small universes) and background jobs (large ones).
Threshold is configurable via the OPTIMIZE_SYNC_THRESHOLD environment variable.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import database_manager, get_db
from app.metrics import time_endpoint
from app.repositories.execution_repository import ExecutionRepository
from app.schemas.optimization import OptimizationRunResponse, OptimizeRequest
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services.optimization_service import (
    build_pipeline,
    extract_results,
    resolve_optimizer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimize", tags=["Optimization"])

# Module-level BackgroundJobService instance — follows macro_regime.py pattern.
_job_service = BackgroundJobService(
    job_type="optimize",
    session_factory=database_manager.get_session,
)

_DEFAULT_SYNC_THRESHOLD = 50


def _sync_threshold() -> int:
    """Return tickers-per-request threshold below which runs are synchronous."""
    return int(os.environ.get("OPTIMIZE_SYNC_THRESHOLD", _DEFAULT_SYNC_THRESHOLD))


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_optimize_bg(job_id: str, run_id: str, request: OptimizeRequest) -> None:
    """Background wrapper managing job + OptimizationRun lifecycle."""
    _job_service.update_job(job_id, status="running")
    try:
        with database_manager.get_session() as session:
            _execute_and_persist(request, session, run_id=uuid.UUID(run_id), job_id=job_id)
    except Exception as exc:
        logger.error("Optimize job %s failed: %s", job_id, exc)
        _job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        with database_manager.get_session() as session:
            _fail_run(session, uuid.UUID(run_id), str(exc))


def _execute_and_persist(
    request: OptimizeRequest,
    session: Session,
    run_id: uuid.UUID | None = None,
    job_id: str | None = None,
) -> OptimizationRun:  # type: ignore[name-defined]  # noqa: F821
    """Run optimization and persist results. Returns the completed OptimizationRun."""
    optimizer = resolve_optimizer(request.optimizer_type, request.config)
    pipeline, returns_df = build_pipeline(
        request.tickers, request.start_date, request.end_date, optimizer, session
    )
    t0 = time.monotonic()
    pipeline.fit(returns_df)
    duration = time.monotonic() - t0
    results = extract_results(pipeline, returns_df)

    repo = ExecutionRepository(session)
    data = _build_run_data(request, results, duration, job_id)
    if run_id is not None:
        return repo.update_optimization_status(run_id, "completed", data)
    run = repo.create_optimization_run({**data, "status": "completed"})
    session.commit()
    return run


def _build_run_data(
    request: OptimizeRequest,
    results: dict,
    duration: float,
    job_id: str | None,
) -> dict:
    """Assemble dict of OptimizationRun fields from request + results."""
    return {
        "optimizer_type": request.optimizer_type,
        "universe_tickers": request.tickers,
        "config": request.config,
        "weights": results["weights"],
        "metrics": results["metrics"],
        "risk_contributions": results["risk_contributions"],
        "efficient_frontier": results["efficient_frontier"],
        "duration_seconds": round(duration, 4),
        "job_id": uuid.UUID(job_id) if job_id else None,
    }


def _fail_run(session: Session, run_id: uuid.UUID, error: str) -> None:
    """Mark an OptimizationRun as failed with an error message."""
    repo = ExecutionRepository(session)
    try:
        repo.update_optimization_status(run_id, "failed", {"error_message": error})
        session.commit()
    except Exception:
        logger.warning("Could not update run %s to failed state", run_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=OptimizationRunResponse,
    response_model_by_alias=True,
)
def post_optimize(
    request: OptimizeRequest,
    db: Session = Depends(get_db),
):
    """Run portfolio optimization.

    - **≤ threshold tickers** (default 50): synchronous, returns 200 + result.
    - **> threshold tickers**: background job, returns 202 + job_id.
    """
    with time_endpoint("optimize"):
        _validate_optimizer_type(request.optimizer_type)
        if len(request.tickers) <= _sync_threshold():
            return _handle_sync(request, db)
        return _handle_async(request, db)


@router.get(
    "/{run_id}",
    response_model=OptimizationRunResponse,
    response_model_by_alias=True,
)
def get_optimize_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Retrieve a completed or in-progress optimization run by UUID."""
    repo = ExecutionRepository(db)
    run = repo.get_optimization_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OptimizationRun {run_id} not found",
        )
    return OptimizationRunResponse.model_validate(run)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_optimizer_type(optimizer_type: str) -> None:
    """Raise 422 for unknown optimizer_type strings."""
    from app.services.optimization_service import OPTIMIZER_REGISTRY

    if optimizer_type not in OPTIMIZER_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unknown optimizer_type: {optimizer_type!r}. "
                f"Valid types: {sorted(OPTIMIZER_REGISTRY)}"
            ),
        )


def _handle_sync(request: OptimizeRequest, db: Session) -> OptimizationRunResponse:
    """Run optimization synchronously and return the persisted result."""
    run = _execute_and_persist(request, db)
    return OptimizationRunResponse.model_validate(run)


def _handle_async(request: OptimizeRequest, db: Session):
    """Create a pending run + background job, return 202."""
    repo = ExecutionRepository(db)
    pending_run = repo.create_optimization_run({
        "status": "pending",
        "optimizer_type": request.optimizer_type,
        "universe_tickers": request.tickers,
        "config": request.config,
        "weights": {},
        "metrics": {},
        "risk_contributions": {},
    })
    db.commit()

    try:
        job_id = _job_service.create_job()
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    _job_service.start_background(
        target=_run_optimize_bg,
        args=(job_id, str(pending_run.id), request),
    )

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id, "run_id": str(pending_run.id)},
    )
