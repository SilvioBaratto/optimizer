"""FastAPI router for LLM-driven Black-Litterman view generation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.view_generation import (
    GeneratedViews,
    fetch_factor_data,
    generate_views,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/views", tags=["Views"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class GenerateViewsRequest(BaseModel):
    tickers: list[str] = Field(
        ...,
        min_length=2,
        description="Ordered list of asset tickers in the portfolio universe.",
    )

    @field_validator("tickers")
    @classmethod
    def non_empty_tickers(cls, v: list[str]) -> list[str]:
        stripped = [t.strip().upper() for t in v]
        if not all(stripped):
            raise ValueError("tickers must not contain empty strings")
        return stripped


class AssetViewResponse(BaseModel):
    asset: str
    direction: int
    magnitude_bps: float
    confidence: float
    reasoning: str


class GenerateViewsResponse(BaseModel):
    """BL-ready view components.

    All arrays are serialised as nested lists for JSON transport.
    """

    n_views: int = Field(..., description="Number of views generated.")
    n_assets: int = Field(..., description="Number of assets in the universe.")
    view_strings: list[str] = Field(
        ..., description="skfolio-compatible view expressions (e.g. 'AAPL == 0.02')."
    )
    P: list[list[float]] = Field(
        ...,
        description="Pick matrix, shape (n_views, n_assets). Row i is the i-th view.",
    )
    Q: list[float] = Field(
        ...,
        description="Expected excess returns vector, shape (n_views,). Decimal units.",
    )
    view_confidences: list[float] = Field(
        ...,
        description="Idzorek alpha_k per view, in (0, 1). Same order as view_strings.",
    )
    idzorek_alphas: dict[str, float] = Field(
        ..., description="Asset ticker → Idzorek alpha_k. All values in (0, 1)."
    )
    views: list[AssetViewResponse] = Field(
        ..., description="Structured per-asset views."
    )
    rationale: str = Field(..., description="LLM narrative explaining view generation.")
    tickers_with_data: list[str] = Field(
        ..., description="Tickers for which factor data was found in the DB."
    )
    tickers_missing_data: list[str] = Field(
        ..., description="Tickers not found in the DB (excluded from view generation)."
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=GenerateViewsResponse,
    summary="Generate Black-Litterman views from multi-factor asset data via LLM",
)
def generate_bl_views(
    request: GenerateViewsRequest,
    db: Session = Depends(get_db),
) -> GenerateViewsResponse:
    """Fetch per-asset factor signals from the DB and use an LLM to generate
    structured Black-Litterman views.

    The returned ``P``, ``Q``, and ``view_confidences`` pass directly into
    ``build_black_litterman()`` without reshaping.

    **Shape guarantees:**
    - ``P``: *(n_views, n_assets)*
    - ``Q``: *(n_views,)*
    - ``view_confidences``: *(n_views,)* — all values in (0, 1)
    - ``idzorek_alphas``: one entry per view asset — all values in (0, 1)
    """
    tickers = request.tickers

    # 1. Fetch factor data from DB
    try:
        factor_data = fetch_factor_data(db, tickers)
    except Exception as exc:
        logger.exception("DB fetch failed for tickers %s", tickers)
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

    # 2. Call BAML and convert to BL arrays
    try:
        result: GeneratedViews = generate_views(tickers, factor_data)
    except Exception as exc:
        logger.exception("LLM view generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        ) from exc

    # 3. Validate shape guarantees
    n_views = len(result.view_strings)
    n_assets = len(tickers)

    if n_views > 0:
        assert result.P.shape == (n_views, n_assets), (
            f"P shape mismatch: expected ({n_views}, {n_assets}), got {result.P.shape}"
        )
        assert result.Q.shape == (n_views,), (
            f"Q shape mismatch: expected ({n_views},), got {result.Q.shape}"
        )

    return GenerateViewsResponse(
        n_views=n_views,
        n_assets=n_assets,
        view_strings=result.view_strings,
        P=result.P.tolist(),
        Q=result.Q.tolist(),
        view_confidences=result.view_confidences,
        idzorek_alphas=result.idzorek_alphas,
        views=[
            AssetViewResponse(
                asset=v.asset,
                direction=v.direction,
                magnitude_bps=v.magnitude_bps,
                confidence=v.confidence,
                reasoning=v.reasoning,
            )
            for v in result.asset_views
        ],
        rationale=result.rationale,
        tickers_with_data=tickers_with_data,
        tickers_missing_data=tickers_missing,
    )
