"""FastAPI router for Multi-LLM Opinion Pooling view generation."""

from __future__ import annotations

import logging

from baml_client.types import ExpertPersona
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.view_generation_repository import ViewGenerationRepository
from app.schemas.views import (
    ExpertViewSummary,
    ICHistory,
    OpinionPoolRequest,
    OpinionPoolResponse,
)
from app.services.opinion_pooling import (
    ALL_PERSONAS,
    OpinionPoolResult,
    build_llm_opinion_pool,
)
from app.services.view_generation import fetch_factor_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/views", tags=["Views"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_personas(persona_strs: list[str] | None) -> list[ExpertPersona]:
    if persona_strs is None:
        return ALL_PERSONAS
    return [ExpertPersona(p) for p in persona_strs]


def _resolve_ic_histories(
    ic_histories: list[ICHistory] | None,
    personas: list[ExpertPersona],
) -> list | None:
    import pandas as pd

    if ic_histories is None:
        return None

    # Build a lookup from persona string → IC series
    ic_map: dict[str, list[float]] = {h.persona: h.ic_values for h in ic_histories}
    result: list[pd.Series] = []
    for persona in personas:
        vals = ic_map.get(persona.value, [])
        result.append(pd.Series(vals, dtype=float))
    return result


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _get_view_repo(db: Session = Depends(get_db)) -> ViewGenerationRepository:
    return ViewGenerationRepository(db)


@router.post(
    "/opinion-pool",
    response_model=OpinionPoolResponse,
    summary="Run N LLM expert personas and combine views with IC-calibrated Opinion Pooling",
)
def generate_opinion_pool(
    request: OpinionPoolRequest,
    repo: ViewGenerationRepository = Depends(_get_view_repo),
) -> OpinionPoolResponse:
    """Fetch per-asset factor data from the DB, run each LLM expert persona,
    compute IC-calibrated credibility weights, and combine via Opinion Pooling.

    **IC weight rules:**
    - Weight ∝ max(ICIR, 0) + ε (near-zero, not hard-zero, for ICIR ≤ 0)
    - Weights normalised to sum to 1.0
    - If no ``ic_histories`` provided, equal weights are used

    **At least 2 expert personas** are supported (value, momentum, macro).
    """
    tickers = request.tickers
    personas = _parse_personas(request.personas)
    ic_series = _resolve_ic_histories(request.ic_histories, personas)

    # 1. Fetch factor data
    try:
        factor_data = fetch_factor_data(repo, tickers)
    except Exception as exc:
        logger.exception("DB fetch failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        ) from exc

    tickers_with_data = [fd.ticker for fd in factor_data]
    tickers_missing = [t for t in tickers if t not in set(tickers_with_data)]

    if not factor_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No factor data found in DB for any of the requested tickers.",
        )

    # 2. Run experts + pool
    try:
        result: OpinionPoolResult = build_llm_opinion_pool(
            assets=factor_data,
            tickers=tickers,
            ic_histories=ic_series,
            personas=personas,
            tau=request.tau,
            is_linear_pooling=request.is_linear_pooling,
            divergence_penalty=request.divergence_penalty,
        )
    except Exception as exc:
        logger.exception("Opinion pooling failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    # 3. Build response
    expert_summaries = [
        ExpertViewSummary(
            persona=er.persona.value,
            name=er.name,
            n_views=len(er.view_strings),
            view_strings=er.view_strings,
            idzorek_alphas=er.idzorek_alphas,
            ic_weight=float(result.ic_weights[i]),
        )
        for i, er in enumerate(result.expert_results)
    ]

    total_views = sum(len(er.view_strings) for er in result.expert_results)

    return OpinionPoolResponse(
        n_experts=len(result.expert_results),
        tickers=tickers,
        tickers_with_data=tickers_with_data,
        tickers_missing_data=tickers_missing,
        experts=expert_summaries,
        ic_weights=result.ic_weights.tolist(),
        pooling_type="linear" if request.is_linear_pooling else "geometric",
        total_views=total_views,
    )
