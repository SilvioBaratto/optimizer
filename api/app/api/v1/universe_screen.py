"""FastAPI router for universe investability screening (issue #361).

Endpoint:
  POST /universe/screen — run investability screens, return passing tickers + diagnostics
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.universe_screen import UniverseScreenRequest, UniverseScreenResponse
from app.services.universe_screening_service import run_universe_screen

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/universe", tags=["Universe"])


@router.post(
    "/screen",
    response_model=UniverseScreenResponse,
    summary="Screen instrument universe for investability",
)
def screen_universe(
    request: UniverseScreenRequest,
    session: Session = Depends(get_db),
) -> UniverseScreenResponse:
    """Apply investability screens to all active instruments in the database.

    Returns passing tickers and per-ticker per-screen pass/fail diagnostics.
    Hysteresis is supported via optional ``current_members`` field.
    """
    try:
        result = run_universe_screen(request, session)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return UniverseScreenResponse(**result)
