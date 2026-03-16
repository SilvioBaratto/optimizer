"""Job history API — read-only access to background job records."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
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
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid job ID format",
        )

    row = repo.get(uid)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return _job_to_summary(row)
