"""Pydantic schemas for LLM-augmented moment estimation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.base import CamelCaseModel


class CalibrateDeltaRequest(BaseModel):
    macro_text: str = Field(
        ...,
        min_length=20,
        description="Free-form macro regime description (Fed statement, GDP release, etc.)",
    )


class CalibrateDeltaResponse(CamelCaseModel):
    delta: float = Field(
        ..., description="Calibrated risk aversion scalar in [1.0, 10.0]."
    )
    rationale: str = Field(..., description="One-sentence explanation.")


class AdaptFactorWeightsRequest(BaseModel):
    macro_indicators: str = Field(
        ...,
        min_length=20,
        description="Textual summary of macro indicators (PMI, yield curve, unemployment, etc.).",
    )
    factor_groups: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of factor group names to weight.",
    )

    @field_validator("factor_groups")
    @classmethod
    def non_empty_groups(cls, v: list[str]) -> list[str]:
        if not all(g.strip() for g in v):
            raise ValueError("factor_groups must not contain empty strings")
        return [g.strip() for g in v]


class AdaptFactorWeightsResponse(CamelCaseModel):
    phase: str = Field(..., description="Detected business cycle phase.")
    weights: dict[str, float] = Field(
        ..., description="Factor group → weight multiplier (sum = len(factor_groups))."
    )
    rationale: str = Field(
        ..., description="Explanation of phase and weight adjustments."
    )


class SelectCovRegimeRequest(BaseModel):
    news_headlines: list[str] = Field(
        ...,
        min_length=1,
        description="Recent market news headlines.",
    )
    avg_sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Average sentiment score in [-1.0, 1.0].",
    )
    realized_vol_30d: float = Field(
        ...,
        ge=0.0,
        description="Annualised 30-day realised volatility (e.g. 0.15 = 15%).",
    )

    @field_validator("news_headlines")
    @classmethod
    def non_empty_headlines(cls, v: list[str]) -> list[str]:
        if not all(h.strip() for h in v):
            raise ValueError("news_headlines must not contain empty strings")
        return [h.strip() for h in v]


class SelectCovRegimeResponse(CamelCaseModel):
    estimator: str = Field(
        ..., description="Recommended covariance estimator (CovEstimatorChoice value)."
    )
    estimator_type: str = Field(
        ..., description="Optimizer CovEstimatorType string (e.g. 'ledoit_wolf')."
    )
    confidence: float = Field(
        ..., description="Confidence in selection, in [0.0, 1.0]."
    )
    rationale: str = Field(
        ..., description="Explanation referencing sentiment and volatility."
    )
