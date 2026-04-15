"""Response schemas for the scheduler status endpoint."""

from datetime import datetime

from pydantic import Field

from app.schemas.base import CamelCaseModel


class SchedulerJobStatus(CamelCaseModel):
    """Status of a single APScheduler job enriched with last-run data."""

    job_id: str = Field(..., description="APScheduler job identifier")
    name: str = Field(..., description="Human-readable job name")
    next_run_time: datetime | None = Field(
        None, description="Next scheduled execution (UTC)"
    )
    last_run_time: datetime | None = Field(
        None, description="finished_at of the most recent completed/failed run"
    )
    last_status: str | None = Field(
        None, description="Status of the most recent completed/failed run"
    )
    trigger: str = Field(..., description="APScheduler trigger description")


class SchedulerStatusResponse(CamelCaseModel):
    """Top-level response for GET /scheduler/status."""

    scheduler_running: bool = Field(
        ..., description="True when APScheduler is active"
    )
    jobs: list[SchedulerJobStatus] = Field(
        default_factory=list, description="Per-job status entries"
    )
