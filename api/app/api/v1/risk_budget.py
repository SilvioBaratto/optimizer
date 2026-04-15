"""FastAPI router for LLM-driven risk budget calibration (issue #17)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.risk_budget import RiskBudgetRequest, RiskBudgetResponse
from app.services.risk_budget_service import calibrate_risk_budget

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/budget-calibration",
    response_model=RiskBudgetResponse,
    summary="Calibrate risk budget vector from qualitative sector outlook via LLM",
)
def calibrate_budget(
    request: RiskBudgetRequest,
) -> RiskBudgetResponse:
    """Translate a qualitative sector outlook into a numeric risk budget vector.

    The returned ``budget_vector`` can be passed directly as ``risk_budget``
    to ``build_risk_budgeting()`` — no reshaping needed.

    Budget allocation logic:
    - Overweight / bullish sectors → higher budget (allowed to take more risk)
    - Underweight / bearish sectors → lower budget
    - Neutral sectors → proportional to 1/N baseline
    - Budget is distributed equally among assets within each sector
    - Result is normalised to sum to 1.0
    """
    try:
        budget = calibrate_risk_budget(
            sector_outlook=request.sector_outlook,
            sector_universe=request.sector_universe,
            asset_sector_map=request.asset_sector_map,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    assets = list(request.asset_sector_map.keys())

    return RiskBudgetResponse(
        n_assets=len(assets),
        assets=assets,
        budget_vector=budget.tolist(),
        budget_sum=float(budget.sum()),
    )
