"""FastAPI router for LLM-driven risk budget calibration (issue #17)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.services.risk_budget_service import calibrate_risk_budget

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RiskBudgetRequest(BaseModel):
    sector_outlook: str = Field(
        ...,
        min_length=10,
        description=(
            "Qualitative sector outlook text (e.g. 'Overweight Technology "
            "and Healthcare; underweight Energy and Utilities.')."
        ),
    )
    sector_universe: list[str] = Field(
        ...,
        min_length=1,
        description="Exhaustive list of sector names present in the portfolio.",
    )
    asset_sector_map: dict[str, str] = Field(
        ...,
        min_length=1,
        description="Asset ticker → sector name mapping for the full portfolio universe.",
    )

    @field_validator("asset_sector_map")
    @classmethod
    def tickers_non_empty(cls, v: dict[str, str]) -> dict[str, str]:
        cleaned = {k.strip().upper(): s.strip() for k, s in v.items() if k.strip()}
        if not cleaned:
            raise ValueError("asset_sector_map must contain at least one asset")
        return cleaned

    @field_validator("sector_universe")
    @classmethod
    def sectors_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("sector_universe must contain at least one sector")
        return cleaned


class RiskBudgetResponse(BaseModel):
    n_assets: int = Field(..., description="Number of assets in the budget vector.")
    assets: list[str] = Field(..., description="Asset tickers in budget vector order.")
    budget_vector: list[float] = Field(
        ...,
        description=(
            "Risk budget weights, shape (n_assets,). Non-negative and sum to 1.0. "
            "Pass directly as risk_budget= to build_risk_budgeting()."
        ),
    )
    budget_sum: float = Field(
        ..., description="Sum of budget_vector (should be ≈ 1.0)."
    )


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
