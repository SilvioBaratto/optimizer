"""Pydantic schemas for factor research endpoints.

Covers factor_scores and factor_validation_reports ORM models.
"""

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.base import CamelCaseModel, StrFromUUID

# ---------------------------------------------------------------------------
# Factor Score
# ---------------------------------------------------------------------------


class FactorScoreResponse(CamelCaseModel):
    """Response schema for a single factor score row."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: StrFromUUID = Field(..., description="Row UUID")
    ticker: str = Field(..., description="Asset ticker symbol")
    score_date: datetime.date = Field(..., description="Date the score was computed")
    factor_type: str = Field(..., description="Factor type (e.g. momentum, quality)")
    factor_group: str = Field(..., description="Factor group the type belongs to")
    raw_score: float = Field(..., description="Raw un-standardized factor score")
    standardized_score: float | None = Field(
        default=None, description="Z-score normalized factor score"
    )
    composite_score: float | None = Field(
        default=None, description="Composite score across factors"
    )
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


class FactorScoreListResponse(CamelCaseModel):
    """Paginated list of factor score responses."""

    items: list[FactorScoreResponse] = Field(
        default_factory=list, description="Factor score rows"
    )
    total: int = Field(..., ge=0, description="Total number of rows")


# ---------------------------------------------------------------------------
# Factor Validation Report
# ---------------------------------------------------------------------------


class FactorValidationReportResponse(CamelCaseModel):
    """Response schema for a factor validation report row."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: StrFromUUID = Field(..., description="Row UUID")
    report_date: datetime.date = Field(..., description="Date the report was generated")
    factor_type: str | None = Field(
        default=None, description="Factor type; null for aggregate reports"
    )
    validation_type: str = Field(
        ..., description="Validation methodology (in_sample / out_of_sample)"
    )
    ic_mean: float | None = Field(
        default=None, description="Mean information coefficient"
    )
    ic_std: float | None = Field(default=None, description="Standard deviation of IC")
    icir: float | None = Field(default=None, description="IC information ratio")
    t_stat: float | None = Field(default=None, description="Newey-West t-statistic")
    p_value: float | None = Field(
        default=None, description="p-value for the t-statistic"
    )
    vif: float | None = Field(default=None, description="Variance inflation factor")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional validation details"
    )
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class FactorComputeRequest(BaseModel):
    """Request body for POST /factors/compute."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Non-empty list of asset tickers"
    )
    start_date: datetime.date = Field(..., description="Start of computation window")
    end_date: datetime.date = Field(..., description="End of computation window")
    factor_config: dict[str, Any] | None = Field(
        default=None, description="Optional factor computation configuration"
    )


class FactorValidateRequest(BaseModel):
    """Request body for POST /factors/validate."""

    factor_type: str = Field(..., description="Factor type to validate")
    validation_type: str | None = Field(
        default=None,
        description="Validation methodology (in_sample / out_of_sample)",
    )
