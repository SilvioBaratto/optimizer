"""FastAPI router for LLM-augmented moment estimation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.llm_moments import (
    AdaptFactorWeightsRequest,
    AdaptFactorWeightsResponse,
    CalibrateDeltaRequest,
    CalibrateDeltaResponse,
    SelectCovRegimeRequest,
    SelectCovRegimeResponse,
)
from app.services.llm_moments import (
    adapt_factor_weights,
    calibrate_delta,
    cov_estimator_type_str,
    select_cov_regime,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-moments", tags=["LLM Moments"])


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
def select_cov_regime_endpoint(
    request: SelectCovRegimeRequest,
) -> SelectCovRegimeResponse:
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
