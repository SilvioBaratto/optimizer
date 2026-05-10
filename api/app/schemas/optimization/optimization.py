"""Pydantic schemas for optimization run endpoints.

Covers optimization_runs ORM model.
"""

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas._shared import CamelCaseModel, StrFromUUID


class OptimizeRequest(BaseModel):
    """Request body for POST /optimize."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Non-empty list of universe tickers"
    )
    start_date: datetime.date = Field(..., description="Start of price history window")
    end_date: datetime.date = Field(..., description="End of price history window")
    optimizer_type: Literal["mean_risk"] = Field(
        ..., description="Optimizer strategy: 'mean_risk'"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Optimizer-specific parameter overrides"
    )
    constraints: dict[str, Any] | None = Field(
        default=None, description="Optional weight / exposure constraints"
    )


class OptimizationRunResponse(CamelCaseModel):
    """Response schema for a single optimization run row."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: StrFromUUID = Field(..., description="Run UUID")
    portfolio_id: StrFromUUID | None = Field(
        default=None, description="Owning portfolio UUID, null for ad-hoc runs"
    )
    job_id: StrFromUUID | None = Field(
        default=None, description="Background job UUID, null for synchronous runs"
    )
    status: str = Field(
        ..., description="Run status (pending/running/completed/failed)"
    )
    optimizer_type: str = Field(
        ..., description="Optimizer identifier used for the run"
    )
    universe_tickers: list[str] = Field(
        ..., description="Universe tickers passed to the run"
    )
    config: dict[str, Any] = Field(..., description="Effective optimizer configuration")
    weights: dict[str, float] = Field(..., description="Optimal portfolio weights")
    metrics: dict[str, Any] = Field(
        ..., description="Portfolio metrics (Sharpe, CVaR, …)"
    )
    risk_contributions: dict[str, float] = Field(
        ..., description="Per-asset risk contribution"
    )
    efficient_frontier: list[Any] | None = Field(
        default=None, description="Efficient frontier points, null when not computed"
    )
    error_message: str | None = Field(
        default=None, description="Error details for failed synchronous runs"
    )
    solver_log: str | None = Field(
        default=None, description="Solver stdout/stderr captured during optimization"
    )
    duration_seconds: float | None = Field(
        default=None, description="Wall-clock optimization duration in seconds"
    )
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


class OptimizationRunListResponse(CamelCaseModel):
    """Paginated list of optimization run responses."""

    items: list[OptimizationRunResponse] = Field(
        default_factory=list, description="Optimization run rows"
    )
    total: int = Field(..., ge=0, description="Total number of rows")
