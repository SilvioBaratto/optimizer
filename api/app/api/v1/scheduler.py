"""Read-only endpoint exposing APScheduler job statuses (issue #375).

Each APScheduler job_id is mapped to its primary ``background_jobs`` domain
so the last-run time and status can be retrieved from the database.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.background_job_repository import BackgroundJobRepository
from app.schemas.scheduler import SchedulerJobStatus, SchedulerStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])

# Maps each APScheduler job_id to the primary background_jobs.job_type that
# represents the first (or only) step of that job.  Used to retrieve the most
# recent run record from the database.
_JOB_ID_TO_DOMAIN: dict[str, str] = {
    "daily_pipeline": "yfinance_fetch",
    "midday_news": "macro_news_fetch",
    "weekly_refetch": "yfinance_fetch",
    "fred_monthly": "fred_fetch",
    "news_refresh": "news_summarize",
}


def _get_repo(db: Session = Depends(get_db)) -> BackgroundJobRepository:
    return BackgroundJobRepository(db)


def _build_job_status(
    job,
    repo: BackgroundJobRepository,
) -> SchedulerJobStatus:
    """Map a single APScheduler Job + DB row into a SchedulerJobStatus."""
    domain = _JOB_ID_TO_DOMAIN.get(job.id)
    latest = repo.get_latest_by_type(domain) if domain else None

    return SchedulerJobStatus(
        job_id=job.id,
        name=job.name,
        next_run_time=job.next_run_time,
        last_run_time=latest.finished_at if latest else None,
        last_status=latest.status if latest else None,
        trigger=str(job.trigger),
    )


@router.get("/status", response_model=SchedulerStatusResponse)
def get_scheduler_status(
    request: Request,
    repo: BackgroundJobRepository = Depends(_get_repo),
) -> SchedulerStatusResponse:
    """Return the running state and per-job statuses of the APScheduler instance.

    - ``scheduler_running`` is ``false`` when the scheduler is not initialised
      or has been shut down.
    - ``jobs`` is empty when the scheduler is not running.
    - ``last_run_time`` and ``last_status`` are ``null`` when no
      ``background_jobs`` row exists for a job's primary domain.
    """
    scheduler = getattr(request.app.state, "scheduler", None)

    if scheduler is None or not getattr(scheduler, "running", False):
        return SchedulerStatusResponse(scheduler_running=False, jobs=[])

    jobs = [_build_job_status(job, repo) for job in scheduler.get_jobs()]
    return SchedulerStatusResponse(scheduler_running=True, jobs=jobs)
