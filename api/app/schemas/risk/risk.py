"""Pydantic schemas for risk limit endpoints.

Covers risk_limits ORM model.
"""

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas._shared import CamelCaseModel, StrFromUUID


class RiskLimitCreate(BaseModel):
    """Request body for POST /risk/{portfolio_name}/limits."""

    metric: str = Field(
        ..., min_length=1, description="Risk metric identifier (e.g. max_drawdown)"
    )
    limit_type: Literal["upper", "lower"] = Field(
        ..., description="Constraint direction: 'upper' or 'lower'"
    )
    threshold: float = Field(
        ..., gt=0, description="Limit threshold value (must be > 0)"
    )


class RiskLimitUpdate(BaseModel):
    """Request body for PUT /risk/{portfolio_name}/limits/{limit_id}."""

    current_value: float = Field(..., description="Latest observed metric value")
    is_breached: bool = Field(
        ..., description="Whether the limit is currently breached"
    )


class RiskLimitResponse(CamelCaseModel):
    """Response schema for a single risk limit row."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: StrFromUUID = Field(..., description="Row UUID")
    portfolio_id: StrFromUUID = Field(..., description="Owning portfolio UUID")
    metric: str = Field(..., description="Risk metric identifier")
    limit_type: str = Field(..., description="Constraint direction (upper / lower)")
    threshold: float = Field(..., description="Limit threshold value")
    current_value: float | None = Field(
        default=None, description="Most recent observed metric value"
    )
    is_breached: bool = Field(
        ..., description="Whether the limit is currently breached"
    )
    last_checked_at: datetime.datetime | None = Field(
        default=None, description="Timestamp of the last breach check"
    )
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


class RiskLimitListResponse(CamelCaseModel):
    """List of risk limit responses with breach summary."""

    items: list[RiskLimitResponse] = Field(
        default_factory=list, description="Risk limit rows"
    )
    breach_count: int = Field(
        ..., ge=0, description="Number of currently breached limits"
    )
