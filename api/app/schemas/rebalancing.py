"""Pydantic schemas for rebalancing policy endpoints.

Covers rebalancing_policies ORM model.
"""

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.base import CamelCaseModel, StrFromUUID


class RebalancingPolicyCreate(BaseModel):
    """Request body for POST /rebalancing-policies."""

    name: str = Field(
        ..., min_length=1, max_length=100, description="Non-empty policy name"
    )
    policy_type: str = Field(
        ..., description="Policy type (calendar / threshold / hybrid)"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Policy-specific configuration parameters"
    )


class RebalancingPolicyResponse(CamelCaseModel):
    """Response schema for a single rebalancing policy row."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: StrFromUUID = Field(..., description="Row UUID")
    portfolio_id: StrFromUUID = Field(..., description="Owning portfolio UUID")
    name: str = Field(..., description="Policy name")
    policy_type: str = Field(
        ..., description="Policy type (calendar / threshold / hybrid)"
    )
    config: dict[str, Any] = Field(..., description="Effective policy configuration")
    is_active: bool = Field(..., description="Whether this is the active policy")
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


class RebalancingPolicyListResponse(CamelCaseModel):
    """Paginated list of rebalancing policy responses."""

    items: list[RebalancingPolicyResponse] = Field(
        default_factory=list, description="Rebalancing policy rows"
    )
    total: int = Field(..., ge=0, description="Total number of rows")
