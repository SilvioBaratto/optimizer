"""Pydantic schemas for backtest run endpoints.

Covers backtest_runs ORM model and background job lifecycle.
"""

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas._shared import (
    AsyncJobCreateResponse,
    AsyncJobProgress,
    CamelCaseModel,
    StrFromUUID,
)


class BacktestJobResponse(AsyncJobCreateResponse):
    """Returned when a backtest background job is created (POST /backtest)."""

    run_id: str = Field(..., description="Pre-created BacktestRun UUID")


class BacktestProgressResponse(AsyncJobProgress):
    """Progress info polled via GET /backtest/{job_id}."""


class BacktestRequest(BaseModel):
    """Request body for POST /backtest."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Non-empty list of universe tickers"
    )
    start_date: datetime.date = Field(..., description="Start of backtest window")
    end_date: datetime.date = Field(..., description="End of backtest window")
    pipeline_config: dict[str, Any] = Field(
        default_factory=dict, description="Pipeline configuration overrides"
    )


class BacktestRunResponse(CamelCaseModel):
    """Response schema for a single backtest run row."""

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
    config: dict[str, Any] = Field(..., description="Effective pipeline configuration")
    equity_curve: dict[str, Any] = Field(
        ..., description="Date-indexed equity curve values"
    )
    drawdowns: dict[str, Any] = Field(
        ..., description="Drawdown series and summary stats"
    )
    monthly_returns: dict[str, Any] = Field(
        ..., description="Month-indexed simple returns"
    )
    yearly_returns: dict[str, Any] = Field(
        ..., description="Year-indexed compounded returns"
    )
    rolling_metrics: dict[str, Any] = Field(
        ..., description="Rolling risk/return metrics (Sharpe, vol, …)"
    )
    turnover_history: dict[str, Any] = Field(
        ..., description="Date-indexed portfolio turnover"
    )
    cv_fold_metrics: list[Any] | None = Field(
        default=None, description="Per-fold cross-validation metrics, null when no CV"
    )
    summary_stats: dict[str, Any] = Field(
        ..., description="Aggregate backtest summary statistics"
    )
    error_message: str | None = Field(
        default=None, description="Error details for failed synchronous runs"
    )
    duration_seconds: float | None = Field(
        default=None, description="Wall-clock backtest duration in seconds"
    )
    created_at: datetime.datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Row last-update timestamp")


class BacktestRunListResponse(CamelCaseModel):
    """Paginated list of backtest run responses."""

    items: list[BacktestRunResponse] = Field(
        default_factory=list, description="Backtest run rows"
    )
    total: int = Field(..., ge=0, description="Total number of rows")
