"""Pydantic schemas for rebalancing policy and decide/preview endpoints.

Covers rebalancing_policies ORM model + decide/preview request/response.
"""

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.base import CamelCaseModel, StrFromUUID


PolicyType = Literal["calendar", "threshold", "hybrid"]


class RebalancingPolicyCreate(BaseModel):
    """Request body for POST /rebalancing-policies."""

    name: str = Field(
        ..., min_length=1, max_length=100, description="Non-empty policy name"
    )
    policy_type: PolicyType = Field(
        ..., description="Policy type: calendar, threshold, or hybrid"
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


# ---------------------------------------------------------------------------
# Decide endpoint schemas
# ---------------------------------------------------------------------------

_VALID_POLICY_TYPES = Literal["calendar", "threshold", "hybrid"]


class RebalanceDecideRequest(BaseModel):
    """Request body for POST /api/v1/rebalance/decide."""

    current_weights: dict[str, float] = Field(
        ..., description="Current portfolio weights (ticker → weight)"
    )
    target_weights: dict[str, float] = Field(
        ..., description="Target portfolio weights (ticker → weight)"
    )
    policy_type: _VALID_POLICY_TYPES = Field(
        ..., description="Rebalancing policy type: calendar, threshold, or hybrid"
    )
    policy_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Policy-specific config parameters",
    )
    current_date: datetime.date | None = Field(
        default=None,
        description="Evaluation date (defaults to today)",
    )
    last_review_date: datetime.date | None = Field(
        default=None,
        description="Date of last rebalance review (required for hybrid policy)",
    )
    transaction_costs: float = Field(
        default=0.001,
        ge=0.0,
        description="Per-unit transaction cost fraction (default 10 bps)",
    )

    @model_validator(mode="after")
    def _validate_hybrid_requires_last_review_date(self) -> "RebalanceDecideRequest":
        if self.policy_type == "hybrid" and self.last_review_date is None:
            raise ValueError(
                "last_review_date is required when policy_type is 'hybrid'"
            )
        return self


class RebalanceDecideResponse(CamelCaseModel):
    """Response body for POST /api/v1/rebalance/decide."""

    should_rebalance: bool = Field(
        ..., description="Whether the policy recommends rebalancing now"
    )
    turnover: float = Field(
        ..., description="One-way portfolio turnover (sum |Δw| / 2)"
    )
    estimated_cost: float = Field(
        ..., description="Total estimated transaction cost as portfolio fraction"
    )
    trade_weights: dict[str, float] = Field(
        ..., description="Per-asset weight delta: target - current (+ buy, − sell)"
    )


# ---------------------------------------------------------------------------
# Preview endpoint schemas
# ---------------------------------------------------------------------------


class TradeItem(CamelCaseModel):
    """A single trade recommendation in the preview response."""

    ticker: str = Field(..., description="Asset ticker")
    weight_delta: float = Field(
        ..., description="Target weight minus current weight"
    )
    side: Literal["buy", "sell"] = Field(
        ..., description="Trade direction"
    )
    shares: float | None = Field(
        default=None,
        description="Estimated share count to trade (requires portfolio_value)",
    )


class RebalancePreviewResponse(CamelCaseModel):
    """Response body for GET /api/v1/rebalance/preview/{portfolio_name}.

    Empty-state semantics (issue #425): when the portfolio exists but a
    policy or snapshot is missing, ``status`` carries a machine-readable
    reason and the other fields default to empty (portfolio is not 404 —
    the computation simply has no input yet). ``status`` is ``None`` on the
    happy path.
    """

    portfolio_name: str = Field(..., description="Portfolio name")
    policy_type: str | None = Field(
        default=None,
        description="Active rebalancing policy type, null when no active policy",
    )
    target_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Target weights from latest snapshot (empty without a snapshot)",
    )
    current_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Current broker weights (empty when no positions)",
    )
    trades: list[TradeItem] = Field(
        default_factory=list,
        description="Trade list (empty when no broker positions / policy / snapshot)",
    )
    portfolio_value: float | None = Field(
        default=None,
        description="Total portfolio value used for share-count conversion",
    )
    status: str | None = Field(
        default=None,
        description=(
            "Empty-state reason when the preview is not computable. "
            "One of 'no_active_policy', 'no_snapshots', or null on the happy path."
        ),
    )
