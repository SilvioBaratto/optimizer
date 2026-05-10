"""Pydantic schemas for attribution endpoints.

Covers:
  - BrinsonRequest / BrinsonResponse   → POST /attribution/brinson
  - FactorAttributionRequest / FactorAttributionResponse  → POST /attribution/factor
"""

from __future__ import annotations

import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas._shared import CamelCaseModel

_WEIGHT_SUM_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------


def _check_weights_sum_to_one(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"Weights must sum to 1.0 (±{_WEIGHT_SUM_TOLERANCE}); got {total:.4f}"
        )
    return weights


def _check_non_empty(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("Weights must not be empty")
    return weights


# ---------------------------------------------------------------------------
# Brinson
# ---------------------------------------------------------------------------


class BrinsonRequest(BaseModel):
    """Request body for POST /attribution/brinson."""

    portfolio_weights: Annotated[
        dict[str, float],
        Field(
            ..., description="Portfolio ticker → weight mapping (must sum to 1.0 ±0.01)"
        ),
    ]
    benchmark_weights: Annotated[
        dict[str, float],
        Field(
            ..., description="Benchmark ticker → weight mapping (must sum to 1.0 ±0.01)"
        ),
    ]
    start_date: datetime.date = Field(..., description="Period start date (inclusive)")
    end_date: datetime.date = Field(..., description="Period end date (inclusive)")

    @field_validator("portfolio_weights", mode="before")
    @classmethod
    def validate_portfolio_weights(cls, v: dict[str, float]) -> dict[str, float]:
        _check_non_empty(v)
        return _check_weights_sum_to_one(v)

    @field_validator("benchmark_weights", mode="before")
    @classmethod
    def validate_benchmark_weights(cls, v: dict[str, float]) -> dict[str, float]:
        _check_non_empty(v)
        return _check_weights_sum_to_one(v)

    @model_validator(mode="after")
    def validate_date_range(self) -> BrinsonRequest:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class BrinsonSectorRow(CamelCaseModel):
    """Attribution effects for a single sector."""

    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float


class BrinsonResponse(CamelCaseModel):
    """Response for POST /attribution/brinson."""

    sectors: list[BrinsonSectorRow]
    total_allocation: float
    total_selection: float
    total_interaction: float
    total_active_return: float
    portfolio_return: float
    benchmark_return: float


# ---------------------------------------------------------------------------
# Factor attribution
# ---------------------------------------------------------------------------


class FactorAttributionRequest(BaseModel):
    """Request body for POST /attribution/factor."""

    portfolio_weights: Annotated[
        dict[str, float],
        Field(
            ..., description="Portfolio ticker → weight mapping (must sum to 1.0 ±0.01)"
        ),
    ]
    start_date: datetime.date = Field(..., description="Period start date (inclusive)")
    end_date: datetime.date = Field(..., description="Period end date (inclusive)")

    @field_validator("portfolio_weights", mode="before")
    @classmethod
    def validate_portfolio_weights(cls, v: dict[str, float]) -> dict[str, float]:
        _check_non_empty(v)
        return _check_weights_sum_to_one(v)

    @model_validator(mode="after")
    def validate_date_range(self) -> FactorAttributionRequest:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class FactorRow(CamelCaseModel):
    """Contribution of a single factor to portfolio return."""

    factor_name: str
    exposure: float
    factor_return: float
    contribution: float


class FactorAttributionResponse(CamelCaseModel):
    """Response for POST /attribution/factor."""

    factors: list[FactorRow]
    portfolio_return: float
    explained_return: float
    residual: float
