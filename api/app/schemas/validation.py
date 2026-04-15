"""Pydantic v2 schemas for cross-validation endpoints.

Covers request/response shapes for POST /validate/cross-validation
and GET /validate/cross-validation/{job_id}.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.base_job import AsyncJobCreateResponse, AsyncJobProgress


class CvType(str, Enum):
    """Supported cross-validation strategies."""

    walk_forward = "walk_forward"
    cpcv = "cpcv"
    multiple_randomized = "multiple_randomized"


class ValidateRequest(BaseModel):
    """Request body for POST /validate/cross-validation."""

    tickers: list[str] = Field(
        ..., min_length=1, description="Non-empty list of universe tickers"
    )
    start_date: datetime.date = Field(..., description="Start of price history window")
    end_date: datetime.date = Field(..., description="End of price history window")
    cv_type: CvType = Field(
        default=CvType.walk_forward,
        description="Cross-validation strategy: walk_forward | cpcv | multiple_randomized",
    )
    cv_config: dict[str, Any] = Field(
        default_factory=dict,
        description="CV config overrides (test_size, n_folds, …)",
    )
    optimizer_type: str = Field(
        default="hrp",
        description="Optimizer type key (e.g. hrp, mean_risk, …)",
    )
    optimizer_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Optimizer config kwargs forwarded to the factory",
    )


class FoldResult(BaseModel):
    """Per-fold result extracted from a skfolio Portfolio object."""

    weights: dict[str, float] = Field(
        default_factory=dict, description="Asset weights for this fold"
    )
    measures: dict[str, float] = Field(
        default_factory=dict, description="Portfolio measures for this fold"
    )


class ValidateJobResponse(AsyncJobCreateResponse):
    """Returned when a cross-validation background job is created."""


class ValidateProgress(AsyncJobProgress):
    """Progress polled via GET /validate/cross-validation/{job_id}."""

    current_fold: int = Field(0, description="Folds completed so far")
    total_folds: int = Field(0, description="Total folds to process")
