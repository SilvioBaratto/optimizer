"""Pydantic schemas for report generation endpoints."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas._shared import AsyncJobProgress

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

VALID_SECTION_IDS: frozenset[str] = frozenset(
    {
        "portfolio_summary",
        "weights",
        "performance_metrics",
        "risk_analytics",
    }
)


class ReportTemplate(str, Enum):
    standard = "standard"
    executive = "executive"
    detailed = "detailed"


class ReportFormat(str, Enum):
    pdf = "pdf"


class ReportOrientation(str, Enum):
    portrait = "portrait"
    landscape = "landscape"


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class ReportBranding(BaseModel):
    company_name: str = Field(default="", description="Company name shown in header")
    primary_color: str = Field(default="#003366", description="Hex color for headings")
    include_disclaimer: bool = Field(
        default=False, description="Append legal disclaimer to last page"
    )


class ReportGenerateRequest(BaseModel):
    """Request body for POST /reports/generate."""

    template: ReportTemplate = Field(
        default=ReportTemplate.standard,
        description="Report template variant",
    )
    sections: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of section IDs to include in the report",
    )
    branding: ReportBranding = Field(
        default_factory=ReportBranding,
        description="Branding overrides",
    )
    format: ReportFormat = Field(
        default=ReportFormat.pdf,
        description="Output format (only pdf for v1)",
    )
    orientation: ReportOrientation = Field(
        default=ReportOrientation.portrait,
        description="Page orientation",
    )
    portfolio_id: str | None = Field(
        default=None,
        description="Optional portfolio UUID to scope data aggregation",
    )

    @model_validator(mode="after")
    def validate_sections(self) -> ReportGenerateRequest:
        unknown = set(self.sections) - VALID_SECTION_IDS
        if unknown:
            raise ValueError(
                f"Unknown section IDs: {sorted(unknown)}. "
                f"Valid IDs: {sorted(VALID_SECTION_IDS)}"
            )
        return self


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ReportJobCreateResponse(BaseModel):
    """Returned when a report generation job is created."""

    job_id: str = Field(..., description="Background job ID")
    status: str = Field("pending", description="Initial job status")
    message: str = Field("", description="Human-readable status message")


class ReportJobProgress(AsyncJobProgress):
    """Progress polled via GET /reports/jobs/{job_id}."""
