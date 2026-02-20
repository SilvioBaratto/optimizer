"""FastAPI router for LLM-driven stress scenario design (issue #16)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.services.stress_scenarios import (
    generate_stress_scenarios,
    scenario_to_synthetic_data_args,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class StressScenarioRequest(BaseModel):
    current_portfolio: dict[str, float] = Field(
        ...,
        description="Asset ticker → portfolio weight. Weights need not sum to 1.",
        min_length=1,
    )
    macro_context: str = Field(
        ...,
        min_length=10,
        description=(
            "Free-text description of current macro/geopolitical conditions "
            "used as LLM context (e.g. inflation, rate outlook, sector risks)."
        ),
    )
    n_scenarios: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of distinct stress scenarios to generate (1–10).",
    )

    @field_validator("current_portfolio")
    @classmethod
    def tickers_non_empty(cls, v: dict[str, float]) -> dict[str, float]:
        cleaned = {k.strip().upper(): w for k, w in v.items() if k.strip()}
        if not cleaned:
            raise ValueError("current_portfolio must contain at least one ticker")
        return cleaned


class StressScenarioItem(BaseModel):
    name: str
    description: str
    shocks: dict[str, float] = Field(
        ..., description="Ticker → expected return shock in (-1, 1)."
    )
    probability: float = Field(..., description="Subjective probability in (0, 1).")
    horizon_days: int = Field(..., description="Shock horizon in trading days.")
    synthetic_data_args: dict[str, object] = Field(
        ...,
        description=(
            "Ready-to-use sample_args for build_synthetic_data(). "
            "Pass as sample_args= kwarg directly."
        ),
    )


class StressScenarioResponse(BaseModel):
    n_scenarios: int
    tickers: list[str]
    scenarios: list[StressScenarioItem]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/stress-scenarios",
    response_model=StressScenarioResponse,
    summary="Generate forward-looking stress scenarios via LLM",
)
def design_stress_scenarios(
    request: StressScenarioRequest,
) -> StressScenarioResponse:
    """Use an LLM to design plausible forward-looking tail risk scenarios.

    Each scenario contains:
    - A macro narrative and name
    - Per-ticker return shocks in (-1, 1)
    - A subjective probability and time horizon
    - ``synthetic_data_args`` ready for ``build_synthetic_data(sample_args=...)``

    At least one scenario always represents a broad market drawdown.
    """
    try:
        scenarios = generate_stress_scenarios(
            n_scenarios=request.n_scenarios,
            current_portfolio=request.current_portfolio,
            macro_context=request.macro_context,
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

    tickers = list(request.current_portfolio.keys())

    return StressScenarioResponse(
        n_scenarios=len(scenarios),
        tickers=tickers,
        scenarios=[
            StressScenarioItem(
                name=s.name,
                description=s.description,
                shocks=s.shocks,
                probability=s.probability,
                horizon_days=s.horizon_days,
                synthetic_data_args=scenario_to_synthetic_data_args(s),
            )
            for s in scenarios
        ],
    )
