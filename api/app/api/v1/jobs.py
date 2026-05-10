"""Job history API — read-only access to background job records."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from app.database import database_manager, get_db
from app.models.background_job import BackgroundJob
from app.repositories.background_job_repository import BackgroundJobRepository
from app.schemas.jobs import JobListResponse, JobSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _get_repo(db: Session = Depends(get_db)) -> BackgroundJobRepository:
    return BackgroundJobRepository(db)


def _job_to_summary(row: BackgroundJob) -> JobSummary:
    duration: float | None = None
    if row.finished_at and row.started_at:
        duration = (row.finished_at - row.started_at).total_seconds()

    return JobSummary(
        id=str(row.id),
        domain=row.job_type,
        status=row.status,
        current=row.current,
        total=row.total,
        error=row.error,
        errors_count=len(row.errors or []),
        started_at=row.started_at,
        finished_at=row.finished_at,
        duration_seconds=duration,
    )


@router.get("", response_model=JobListResponse)
def list_jobs(
    domain: str | None = Query(None, description="Filter by job type/domain"),
    status_filter: str | None = Query(
        None, alias="status", description="Filter by status"
    ),
    limit: int = Query(20, ge=1, le=100, description="Max results (1–100)"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    repo: BackgroundJobRepository = Depends(_get_repo),
) -> JobListResponse:
    total = repo.count_jobs(domain=domain, status=status_filter)
    rows = repo.list_jobs(
        domain=domain, status=status_filter, limit=limit, offset=offset
    )
    return JobListResponse(
        jobs=[_job_to_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobSummary)
def get_job(
    job_id: str,
    repo: BackgroundJobRepository = Depends(_get_repo),
) -> JobSummary:
    try:
        uid = uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid job ID format",
        ) from exc

    row = repo.get(uid)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return _job_to_summary(row)


_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _sse_event(event_type: str, payload: JobSummary) -> str:
    """Format a single named SSE event block."""
    data = payload.model_dump_json()
    return f"event: {event_type}\ndata: {data}\n\n"


def _fetch_job(uid: uuid.UUID) -> BackgroundJob | None:
    """Blocking DB fetch; designed to run in a thread-pool executor."""
    with database_manager.get_session() as session:
        repo = BackgroundJobRepository(session)
        return repo.get(uid)


async def _poll_job(uid: uuid.UUID, interval: int) -> AsyncGenerator[str, None]:
    """Yield SSE events until terminal state; non-blocking async generator."""
    loop = asyncio.get_running_loop()
    while True:
        row = await loop.run_in_executor(None, _fetch_job, uid)

        if row is None:
            return

        summary = _job_to_summary(row)

        if summary.status in _TERMINAL_STATUSES:
            yield _sse_event("done", summary)
            return

        yield _sse_event("progress", summary)
        await asyncio.sleep(interval)


@router.get("/{job_id}/stream")
async def stream_job(
    job_id: str,
    interval: int = Query(1, ge=1, le=10, description="Poll interval in seconds (1–10)"),
) -> StreamingResponse:
    """Stream job progress as SSE: ``event: progress`` while running, ``event: done`` on terminal."""
    try:
        uid = uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid job ID format",
        ) from exc

    # Pre-flight check: fail fast with 404 if job doesn't exist
    loop = asyncio.get_running_loop()
    exists = await loop.run_in_executor(None, _fetch_job, uid) is not None

    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    return StreamingResponse(
        _poll_job(uid, interval),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _parse_uuid_or_422(job_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid job ID format",
        ) from exc


def _assert_cancellable(row: BackgroundJob) -> None:
    if row.status in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job already terminal (status={row.status})",
        )


@router.post("/{job_id}/cancel", response_model=JobSummary)
def cancel_job(
    job_id: str,
    repo: BackgroundJobRepository = Depends(_get_repo),
) -> JobSummary:
    uid = _parse_uuid_or_422(job_id)
    row = repo.get(uid)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found",
        )
    _assert_cancellable(row)
    repo.update(
        uid,
        status="failed",
        error="operator-cancelled via POST /cancel",
        finished_at=datetime.now(timezone.utc),
    )
    repo.session.commit()
    return _job_to_summary(row)
