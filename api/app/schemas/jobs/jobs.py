"""Response schemas for the job history API."""

from datetime import datetime

from pydantic import BaseModel, Field


class JobSummary(BaseModel):
    """Single job record."""

    id: str = Field(..., description="Unique job ID")
    domain: str = Field(..., description="Job type / domain")
    status: str = Field(..., description="pending | running | completed | failed")
    current: int = Field(0, description="Items processed so far")
    total: int = Field(0, description="Total items to process")
    error: str | None = Field(None, description="Fatal error message")
    errors_count: int = Field(0, description="Number of non-fatal errors")
    started_at: datetime | None = Field(None, description="When the job started")
    finished_at: datetime | None = Field(None, description="When the job finished")
    duration_seconds: float | None = Field(
        None, description="Wall-clock duration in seconds"
    )


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: list[JobSummary]
    total: int = Field(..., description="Total matching jobs")
    limit: int = Field(..., description="Page size used")
    offset: int = Field(..., description="Page offset used")
