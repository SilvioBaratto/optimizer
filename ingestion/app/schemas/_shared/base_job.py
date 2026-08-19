"""Base schemas for async background job endpoints.

Domain-specific job schemas should inherit from these and add only
their own extra fields (e.g. ``current_ticker``, ``current_country``).
"""

from typing import Any

from pydantic import BaseModel, Field


class AsyncJobCreateResponse(BaseModel):
    """Returned when a background job is created."""

    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="pending | running | completed | failed")
    message: str = Field("", description="Human-readable status message")


class AsyncJobProgress(BaseModel):
    """Progress info for a background job."""

    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="pending | running | completed | failed")
    current: int = Field(0, description="Items processed so far")
    total: int = Field(0, description="Total items to process")
    errors: list[str] = Field(
        default_factory=list, description="Non-fatal error messages"
    )
    result: Any = Field(None, description="Final result (when completed)")
    error: str | None = Field(None, description="Fatal error message (when failed)")
