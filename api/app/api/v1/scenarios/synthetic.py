"""FastAPI router for synthetic scenario generation (issue #366).

POST /synthetic/scenario
  - n_samples <= 10_000: inline JSON  (HTTP 200)
  - n_samples >  10_000: stored result (HTTP 201 + download_id)

GET  /synthetic/scenario/{download_id}
  - Returns stored result if found and unexpired (HTTP 200)
  - HTTP 404 otherwise
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.scenarios.synthetic import (
    ScenarioInlineResponse,
    ScenarioRequest,
    ScenarioStoredResponse,
)
from app.services.scenarios.synthetic_service import (
    LARGE_THRESHOLD,
    generate_scenarios,
    load_result,
    store_result,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/synthetic", tags=["Synthetic"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/scenario",
    summary="Generate synthetic return scenarios via vine copula",
    status_code=status.HTTP_200_OK,
)
def generate_scenario(
    request: ScenarioRequest,
    session: Session = Depends(get_db),
) -> JSONResponse:
    """Generate synthetic return scenarios from a fitted vine copula model.

    - **Small results** (``n_samples`` ≤ 10 000): returns inline JSON with
      ``scenarios`` and ``columns`` arrays.  HTTP 200.
    - **Large results** (``n_samples`` > 10 000): stores the result to a
      temporary gzip file and returns a ``download_id`` + ``expires_at``.
      HTTP 201.  Use the GET endpoint to retrieve the full data.
    """
    try:
        scenarios, columns = generate_scenarios(
            tickers=request.tickers,
            n_samples=request.n_samples,
            conditioning=request.conditioning,
            preset=request.preset,
            session=session,
        )
    except ValueError as exc:
        _msg = str(exc)
        # Conditioning-ticker mismatch → 400 Bad Request
        # Insufficient data → 422 Unprocessable Entity
        if "Conditioning tickers" in _msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_msg,
        ) from exc

    if request.n_samples > LARGE_THRESHOLD:
        payload = {
            "scenarios": scenarios.tolist(),
            "columns": columns,
        }
        download_id, expires_at = store_result(payload)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ScenarioStoredResponse(
                download_id=download_id,
                expires_at=expires_at.isoformat(),
            ).model_dump(),
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ScenarioInlineResponse(
            scenarios=scenarios.tolist(),
            columns=columns,
        ).model_dump(),
    )


@router.get(
    "/scenario/{download_id}",
    summary="Retrieve a previously stored large scenario result",
    response_model=ScenarioInlineResponse,
)
def get_scenario(download_id: str) -> ScenarioInlineResponse:
    """Retrieve a large scenario result by its ``download_id``.

    Returns HTTP 404 if the ID is unknown or the result has expired.
    """
    result = load_result(download_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario result {download_id!r} not found or has expired.",
        )
    return ScenarioInlineResponse(**result)
