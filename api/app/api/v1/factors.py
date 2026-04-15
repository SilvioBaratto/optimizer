"""FastAPI router for factor computation endpoints.

Runs ``compute_all_factors()`` over a ticker universe and date range as a
background job, persisting results into the ``factor_scores`` table via
``FactorRepository.bulk_upsert_scores()``.

Pattern mirrors: backtest.py / macro_regime.py / reference_indices.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import database_manager, get_db
from app.schemas.base_job import AsyncJobCreateResponse, AsyncJobProgress
from app.schemas.factors import (
    FactorCompositeScoreResponse,
    FactorComputeRequest,
    FactorExposureConstraintsRequest,
    FactorExposureConstraintsResponse,
    FactorQuintileSpreadRequest,
    FactorQuintileSpreadResponse,
    FactorRegimeTiltRequest,
    FactorRegimeTiltResponse,
    FactorScoreRequest,
    FactorSelectRequest,
    FactorSelectResponse,
    FactorValidateRequest,
    FactorValidateResponse,
)
from app.services._progress import make_progress
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError
from app.services._factor_helpers import FactorDataError
from app.services.factor_analysis_service import (
    build_exposure_constraints_for_tickers,
    compute_quintile_spread_for_tickers,
    compute_regime_tilt,
    select_stocks_for_tickers,
)
from app.services.factor_compute_service import run_factor_compute
from app.services.factor_scoring_service import compute_factor_scores, validate_factors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/factors", tags=["Factors"])

# Module-level job service scoped to factor compute jobs.
_factor_job_service = BackgroundJobService(
    job_type="factors_compute",
    session_factory=database_manager.get_session,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_factor_compute_bg(
    job_id: str,
    request: FactorComputeRequest,
) -> None:
    """Execute factor computation in a daemon thread."""
    _factor_job_service.update_job(job_id, status="running")
    try:
        with database_manager.get_session() as session:
            rows_written = run_factor_compute(
                session=session,
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                factor_config=request.factor_config,
                on_progress=make_progress(job_id, _factor_job_service),
            )
        _factor_job_service.update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result={"rows_written": rows_written},
        )
        logger.info("Factor compute job %s completed: %d rows", job_id, rows_written)
    except Exception as exc:
        logger.error("Factor compute job %s failed: %s", job_id, exc)
        _factor_job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/compute",
    response_model=AsyncJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_factor_compute(request: FactorComputeRequest) -> AsyncJobCreateResponse:
    """Start a background job that computes factor scores for a ticker universe."""
    try:
        job_id = _factor_job_service.create_job()
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    _factor_job_service.start_background(
        target=_run_factor_compute_bg,
        args=(job_id, request),
    )

    return AsyncJobCreateResponse(
        job_id=job_id,
        status="pending",
        message=f"Factor compute started. Poll GET /factors/compute/{job_id} for progress.",
    )


@router.get("/compute/{job_id}", response_model=AsyncJobProgress)
def get_factor_compute_status(job_id: str) -> AsyncJobProgress:
    """Poll the status and progress of a factor compute background job."""
    job = _factor_job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Factor compute job '{job_id}' not found",
        )

    return AsyncJobProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# Validation endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=FactorValidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Run factor validation (in-sample or out-of-sample)",
)
def run_factor_validation_endpoint(
    request: FactorValidateRequest,
    db: Session = Depends(get_db),
) -> FactorValidateResponse:
    """Run IC-based factor validation and persist the report.

    Supports both in-sample (``run_factor_validation``) and out-of-sample
    (``run_factor_oos_validation``) via the ``validation_type`` field.
    """
    try:
        result = validate_factors(
            session=db,
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            factor_type=request.factor_type,
            validation_type=request.validation_type,
        )
    except FactorDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return FactorValidateResponse(**result)


# ---------------------------------------------------------------------------
# Scoring endpoint
# ---------------------------------------------------------------------------

_VALID_COMPOSITE_METHODS = frozenset(
    {"equal_weight", "ic_weighted", "icir_weighted", "ridge_weighted", "gbt_weighted"}
)


@router.post(
    "/score",
    response_model=FactorCompositeScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute composite factor scores for a ticker universe",
)
def run_factor_score_endpoint(
    request: FactorScoreRequest,
    db: Session = Depends(get_db),
) -> FactorCompositeScoreResponse:
    """Compute composite factor scores using the selected weighting method.

    Supports all 5 CompositeMethod values: equal_weight, ic_weighted,
    icir_weighted, ridge_weighted, gbt_weighted.
    """
    if request.composite_method not in _VALID_COMPOSITE_METHODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid composite_method {request.composite_method!r}. "
                f"Valid: {sorted(_VALID_COMPOSITE_METHODS)}"
            ),
        )

    try:
        result = compute_factor_scores(
            session=db,
            tickers=request.tickers,
            score_date=request.score_date,
            composite_method=request.composite_method,
            training_start_date=request.training_start_date,
            training_end_date=request.training_end_date,
            group_weights=request.group_weights,
        )
    except FactorDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return FactorCompositeScoreResponse(**result)


# ---------------------------------------------------------------------------
# Stock selection endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/select",
    response_model=FactorSelectResponse,
    status_code=status.HTTP_200_OK,
    summary="Select stocks from pre-computed factor scores",
)
def run_factor_select_endpoint(
    request: FactorSelectRequest,
    db: Session = Depends(get_db),
) -> FactorSelectResponse:
    """Select stocks using buffer-zone hysteresis and optional sector balancing.

    Fetches composite factor scores from the database and calls
    ``select_stocks()`` from the optimizer library.  When
    ``current_members`` is provided the selection applies hysteresis so
    that existing holdings inside the buffer zone are retained.
    """
    if request.method == "fixed_count" and request.target_count is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="target_count is required when method='fixed_count'",
        )
    if request.method == "quantile" and request.target_quantile is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="target_quantile is required when method='quantile'",
        )
    try:
        result = select_stocks_for_tickers(
            session=db,
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            current_members=request.current_members,
            method=request.method,
            target_count=request.target_count,
            target_quantile=request.target_quantile,
            buffer_fraction=request.buffer_fraction,
            sector_balance=request.sector_balance,
        )
    except FactorDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return FactorSelectResponse(**result)


# ---------------------------------------------------------------------------
# Exposure constraints endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/exposure-constraints",
    response_model=FactorExposureConstraintsResponse,
    status_code=status.HTTP_200_OK,
    summary="Build factor exposure constraint matrices",
)
def run_exposure_constraints_endpoint(
    request: FactorExposureConstraintsRequest,
    db: Session = Depends(get_db),
) -> FactorExposureConstraintsResponse:
    """Build linear factor exposure constraints compatible with the /optimize endpoint.

    Returns ``left_inequality`` and ``right_inequality`` as JSON-serializable
    nested lists ready to be forwarded to ``MeanRisk`` or any optimizer that
    accepts those arguments.
    """
    try:
        result = build_exposure_constraints_for_tickers(
            session=db,
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            bounds=request.bounds,
        )
    except FactorDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return FactorExposureConstraintsResponse(**result)


# ---------------------------------------------------------------------------
# Quintile spread endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/quintile-spread",
    response_model=FactorQuintileSpreadResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute quintile spread diagnostics for a factor",
)
def run_quintile_spread_endpoint(
    request: FactorQuintileSpreadRequest,
    db: Session = Depends(get_db),
) -> FactorQuintileSpreadResponse:
    """Compute quintile portfolio returns and long-short spread for a single factor.

    Assets are ranked by factor score at each date and split into
    ``n_quantiles`` equal buckets.  Returns cumulative return series per
    bucket and the annualised mean of the top-minus-bottom spread.
    """
    try:
        result = compute_quintile_spread_for_tickers(
            session=db,
            tickers=request.tickers,
            factor_name=request.factor_name,
            start_date=request.start_date,
            end_date=request.end_date,
            n_quantiles=request.n_quantiles,
        )
    except FactorDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return FactorQuintileSpreadResponse(**result)


# ---------------------------------------------------------------------------
# Regime tilt endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/regime-tilt",
    response_model=FactorRegimeTiltResponse,
    status_code=status.HTTP_200_OK,
    summary="Apply macro regime tilts to factor group weights",
)
def run_regime_tilt_endpoint(
    request: FactorRegimeTiltRequest,
    db: Session = Depends(get_db),
) -> FactorRegimeTiltResponse:
    """Classify the current macro regime and apply multiplicative tilts.

    Fetches macro indicator observations from the database, classifies the
    regime using ``classify_regime()``, then applies ``apply_regime_tilts()``
    to the provided group weights.  Returns the detected regime, tilted
    weights, and effective multiplier per group.
    """
    try:
        result = compute_regime_tilt(
            session=db,
            group_weights=request.group_weights,
            enable=request.enable,
            max_tilt_multiplier=request.max_tilt_multiplier,
            min_post_tilt_weight=request.min_post_tilt_weight,
        )
    except FactorDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return FactorRegimeTiltResponse(**result)
