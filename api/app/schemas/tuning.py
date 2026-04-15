"""Pydantic schemas for hyperparameter tuning endpoints."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class TuneRequest(BaseModel):
    """Request body for POST /api/v1/tune."""

    tickers: list[str] = Field(..., min_length=1)
    start_date: date
    end_date: date
    optimizer_type: str
    optimizer_config: dict[str, Any] = Field(default_factory=dict)
    search_type: Literal["grid", "randomized"] = "grid"
    param_grid: dict[str, list] = Field(..., min_length=1)
    n_iter: int = Field(50, ge=1)
    cv_config: dict | None = None
    top_n: int = Field(5, ge=1)


class TuneResult(BaseModel):
    """Result returned when a tune job completes."""

    best_params: dict[str, Any]
    best_score: float
    top_n: list[dict[str, Any]]
    cv_results_summary: dict[str, Any]
