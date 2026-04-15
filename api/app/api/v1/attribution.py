"""FastAPI router for attribution endpoints.

Synchronous compute-only endpoints — no background jobs.

  POST /attribution/brinson  — Brinson-Fachler sector decomposition
  POST /attribution/factor   — factor exposure attribution
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.attribution import (
    BrinsonRequest,
    BrinsonResponse,
    FactorAttributionRequest,
    FactorAttributionResponse,
)
from app.services.attribution_service import (
    InsufficientSectorCoverageError,
    MissingDataError,
    run_brinson_attribution,
    run_factor_attribution,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attribution", tags=["Attribution"])


@router.post(
    "/brinson",
    response_model=BrinsonResponse,
    status_code=status.HTTP_200_OK,
    summary="Brinson-Fachler attribution by sector",
)
def brinson_attribution(
    request: BrinsonRequest,
    db: Session = Depends(get_db),
) -> BrinsonResponse:
    """Decompose active return into allocation, selection, and interaction effects.

    Sector mapping sourced from ``TickerProfile.sector``.
    Benchmark and portfolio returns computed from ``price_history``.
    """
    try:
        result = run_brinson_attribution(
            session=db,
            portfolio_weights=request.portfolio_weights,
            benchmark_weights=request.benchmark_weights,
            start_date=request.start_date,
            end_date=request.end_date,
        )
    except MissingDataError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InsufficientSectorCoverageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return BrinsonResponse(**result)


@router.post(
    "/factor",
    response_model=FactorAttributionResponse,
    status_code=status.HTTP_200_OK,
    summary="Factor-exposure attribution using stored factor scores",
)
def factor_attribution(
    request: FactorAttributionRequest,
    db: Session = Depends(get_db),
) -> FactorAttributionResponse:
    """Decompose portfolio return into factor contributions and a residual term.

    Factor exposures read from ``factor_scores`` table.
    Asset returns computed from ``price_history``.
    """
    try:
        result = run_factor_attribution(
            session=db,
            portfolio_weights=request.portfolio_weights,
            start_date=request.start_date,
            end_date=request.end_date,
        )
    except MissingDataError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return FactorAttributionResponse(**result)
