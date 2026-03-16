"""FastAPI router for LLM macro regime classification and BL parameter calibration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import database_manager, get_db  # noqa: F401 — get_db used as dep
from app.schemas.macro_regime import (
    MacroCalibrateBatchJobResponse,
    MacroCalibrateBatchProgress,
    MacroCalibrateBatchRequest,
)
from app.services._progress import make_progress
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services.macro_calibration import (
    DELTA_MAX,
    DELTA_MIN,
    TAU_MAX,
    TAU_MIN,
    CalibrationResult,
    build_bl_config_from_calibration,
    classify_macro_regime,
    run_bulk_calibrate,
)
from app.services.scrapers import PORTFOLIO_COUNTRIES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/views", tags=["Views"])

_calibrate_job_service = BackgroundJobService(
    job_type="macro_calibrate",
    session_factory=database_manager.get_session,
)


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
        default="USA",
        description="Country/region to fetch macro indicators for.",
    ),
    macro_text: str | None = Query(
        default=None,
        description=(
            "Optional free-form macro text override. "
            "If provided, DB fetch is skipped and this text is passed directly to the LLM."
        ),
    ),
    force_refresh: bool = Query(
        default=False,
        description=(
            "Bypass cached calibration and invoke the LLM. "
            "The fresh result is persisted to the ``macro_calibrations`` table."
        ),
    ),
    db: Session = Depends(get_db),
) -> MacroCalibrationResponse:
    """Fetch recent macro indicators from the DB, classify the business cycle phase
    via an LLM, and return calibrated (δ, τ) ready for ``BlackLittermanConfig``.

    By default returns the cached result from ``macro_calibrations``.
    Pass ``force_refresh=true`` to re-run the LLM (e.g. after a data refresh).

    **Parameter ranges enforced:**
    - ``delta`` ∈ [1.0, 10.0]
    - ``tau`` ∈ [0.001, 0.1]

    **Phase → parameter mapping:**
    | Phase            | δ        | τ      |
    |------------------|----------|--------|
    | EARLY_EXPANSION  | 2.0-2.5  | 0.05   |
    | MID_EXPANSION    | 2.5-3.0  | 0.025  |
    | LATE_EXPANSION   | 3.0-4.0  | 0.01   |
    | RECESSION        | 4.0-6.0  | 0.05   |
    """
    try:
        result: CalibrationResult = classify_macro_regime(
            session=db,
            country=country,
            macro_summary_override=macro_text,
            force_refresh=force_refresh,
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

    db.commit()

    return MacroCalibrationResponse(
        phase=result.phase.value,
        delta=result.delta,
        tau=result.tau,
        confidence=result.confidence,
        rationale=result.rationale,
        macro_summary=result.macro_summary,
        bl_config=build_bl_config_from_calibration(result),
    )


# ---------------------------------------------------------------------------
# Batch calibration (background job)
# ---------------------------------------------------------------------------


def _run_bulk_calibrate(job_id: str, countries: list[str], force_refresh: bool) -> None:
    """Thin wrapper managing job lifecycle around the service function."""
    _calibrate_job_service.update_job(job_id, status="running", total=len(countries))
    try:
        run_bulk_calibrate(
            countries,
            force_refresh,
            on_progress=make_progress(job_id, _calibrate_job_service),
        )
    except Exception as exc:
        logger.error("Bulk calibration %s failed: %s", job_id, exc)
        _calibrate_job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


@router.post(
    "/macro-calibration/batch",
    response_model=MacroCalibrateBatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch-calibrate macro regimes for multiple countries",
)
def start_batch_calibration(
    request: MacroCalibrateBatchRequest = MacroCalibrateBatchRequest(),
):
    countries = request.countries or list(PORTFOLIO_COUNTRIES)
    try:
        job_id = _calibrate_job_service.create_job(current_country="")
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    _calibrate_job_service.start_background(
        target=_run_bulk_calibrate,
        args=(job_id, countries, request.force_refresh),
    )

    return MacroCalibrateBatchJobResponse(
        job_id=job_id,
        status="pending",
        message="Batch calibration started. Poll GET /views/macro-calibration/batch/{job_id}.",
    )


@router.get(
    "/macro-calibration/batch/{job_id}",
    response_model=MacroCalibrateBatchProgress,
    summary="Poll batch calibration job status",
)
def get_batch_calibration_status(job_id: str):
    job = _calibrate_job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return MacroCalibrateBatchProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        current_country=job.get("current_country", ""),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )
