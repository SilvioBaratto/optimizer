"""FastAPI router for LLM-driven stress scenario design (issue #16)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.stress_scenarios import (
    StressScenarioItem,
    StressScenarioRequest,
    StressScenarioResponse,
)
from app.services.stress_scenarios import (
    generate_stress_scenarios,
    scenario_to_synthetic_data_args,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


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
