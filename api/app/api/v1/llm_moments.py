"""FastAPI router for LLM-augmented moment estimation endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.services.llm_moments import (
    adapt_factor_weights,
    calibrate_delta,
    cov_estimator_type_str,
    select_cov_regime,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-moments", tags=["LLM Moments"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CalibrateDeltaRequest(BaseModel):
    macro_text: str = Field(
        ...,
        min_length=20,
        description="Free-form macro regime description (Fed statement, GDP release, etc.)",
    )


class CalibrateDeltaResponse(BaseModel):
    delta: float = Field(..., description="Calibrated risk aversion scalar in [1.0, 10.0].")
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


class AdaptFactorWeightsResponse(BaseModel):
    phase: str = Field(..., description="Detected business cycle phase.")
    weights: dict[str, float] = Field(
        ..., description="Factor group → weight multiplier (sum = len(factor_groups))."
    )
    rationale: str = Field(..., description="Explanation of phase and weight adjustments.")


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


class SelectCovRegimeResponse(BaseModel):
    estimator: str = Field(
        ..., description="Recommended covariance estimator (CovEstimatorChoice value)."
    )
    estimator_type: str = Field(
        ..., description="Optimizer CovEstimatorType string (e.g. 'ledoit_wolf')."
    )
    confidence: float = Field(..., description="Confidence in selection, in [0.0, 1.0].")
    rationale: str = Field(..., description="Explanation referencing sentiment and volatility.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/calibrate-delta",
    response_model=CalibrateDeltaResponse,
    summary="Calibrate Black-Litterman risk aversion (delta) from macro text",
)
def calibrate_delta_endpoint(request: CalibrateDeltaRequest) -> CalibrateDeltaResponse:
    """Use an LLM to infer a risk-aversion scalar from qualitative macro context.

    The returned ``delta`` is clamped to [1.0, 10.0] regardless of LLM output.
    """
    try:
        result = calibrate_delta(request.macro_text)
    except Exception as exc:
        logger.exception("CalibrateDelta LLM call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    return CalibrateDeltaResponse(delta=result.delta, rationale=result.rationale)


@router.post(
    "/adapt-factor-weights",
    response_model=AdaptFactorWeightsResponse,
    summary="Adapt factor group weights to the current business cycle phase",
)
def adapt_factor_weights_endpoint(
    request: AdaptFactorWeightsRequest,
) -> AdaptFactorWeightsResponse:
    """Use an LLM to classify the business cycle and return factor weight multipliers.

    Multipliers are post-processed to ensure they are positive and sum to
    ``len(factor_groups)`` (preserving overall scale).
    """
    try:
        result = adapt_factor_weights(
            macro_indicators=request.macro_indicators,
            factor_groups=request.factor_groups,
        )
    except Exception as exc:
        logger.exception("AdaptFactorWeights LLM call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    return AdaptFactorWeightsResponse(
        phase=result.phase.value,
        weights=result.weights,
        rationale=result.rationale,
    )


@router.post(
    "/select-cov-regime",
    response_model=SelectCovRegimeResponse,
    summary="Select covariance estimator from news sentiment and volatility regime",
)
def select_cov_regime_endpoint(request: SelectCovRegimeRequest) -> SelectCovRegimeResponse:
    """Use an LLM to recommend which covariance estimator suits current market conditions.

    The ``estimator_type`` field maps directly to :class:`optimizer.moments.CovEstimatorType`.
    """
    try:
        result = select_cov_regime(
            news_headlines=request.news_headlines,
            avg_sentiment_score=request.avg_sentiment_score,
            realized_vol_30d=request.realized_vol_30d,
        )
    except Exception as exc:
        logger.exception("SelectCovRegime LLM call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    return SelectCovRegimeResponse(
        estimator=result.estimator.value,
        estimator_type=cov_estimator_type_str(result),
        confidence=result.confidence,
        rationale=result.rationale,
    )
