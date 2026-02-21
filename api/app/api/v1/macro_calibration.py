"""FastAPI router for LLM macro regime classification and BL parameter calibration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.macro_calibration import (
    DELTA_MAX,
    DELTA_MIN,
    TAU_MAX,
    TAU_MIN,
    CalibrationResult,
    build_bl_config_from_calibration,
    classify_macro_regime,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/views", tags=["Views"])


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class MacroCalibrationResponse(BaseModel):
    """Calibrated Black-Litterman parameters from LLM macro regime classification."""

    phase: str = Field(
        ...,
        description="Business cycle phase: EARLY_EXPANSION | MID_EXPANSION | LATE_EXPANSION | RECESSION.",
    )
    delta: float = Field(
        ...,
        description=f"Risk aversion scalar δ, clamped to [{DELTA_MIN}, {DELTA_MAX}].",
    )
    tau: float = Field(
        ..., description=f"Uncertainty scaling τ, clamped to [{TAU_MIN}, {TAU_MAX}]."
    )
    confidence: float = Field(
        ..., description="LLM classification confidence in [0.0, 1.0]."
    )
    rationale: str = Field(..., description="LLM explanation of phase classification.")
    macro_summary: str = Field(..., description="Macro indicator text fed to the LLM.")
    bl_config: dict = Field(
        ...,
        description=(
            "Ready-to-use kwargs for BlackLittermanConfig. "
            "Pass ``bl_config['tau']`` and ``bl_config['prior_config']['risk_aversion']`` "
            "directly to the optimizer config layer."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/macro-calibration",
    response_model=MacroCalibrationResponse,
    summary="Classify macro regime and calibrate Black-Litterman δ and τ",
)
def get_macro_calibration(
    country: str = Query(
        default="United States",
        description="Country/region to fetch macro indicators for.",
    ),
    macro_text: str | None = Query(
        default=None,
        description=(
            "Optional free-form macro text override. "
            "If provided, DB fetch is skipped and this text is passed directly to the LLM."
        ),
    ),
    db: Session = Depends(get_db),
) -> MacroCalibrationResponse:
    """Fetch recent macro indicators from the DB, classify the business cycle phase
    via an LLM, and return calibrated (δ, τ) ready for ``BlackLittermanConfig``.

    **Parameter ranges enforced:**
    - ``delta`` ∈ [1.0, 10.0]
    - ``tau`` ∈ [0.001, 0.1]

    **Phase → parameter mapping:**
    | Phase            | δ        | τ      |
    |------------------|----------|--------|
    | EARLY_EXPANSION  | 2.0–2.5  | 0.05   |
    | MID_EXPANSION    | 2.5–3.0  | 0.025  |
    | LATE_EXPANSION   | 3.0–4.0  | 0.01   |
    | RECESSION        | 4.0–6.0  | 0.05   |
    """
    try:
        result: CalibrationResult = classify_macro_regime(
            session=db,
            country=country,
            macro_summary_override=macro_text,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Macro calibration failed for country=%s", country)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    return MacroCalibrationResponse(
        phase=result.phase.value,
        delta=result.delta,
        tau=result.tau,
        confidence=result.confidence,
        rationale=result.rationale,
        macro_summary=result.macro_summary,
        bl_config=build_bl_config_from_calibration(result),
    )
