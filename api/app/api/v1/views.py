"""FastAPI router for LLM-driven Black-Litterman view generation and entropy pooling."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.view_generation_repository import ViewGenerationRepository
from app.schemas.views import (
    AssetViewResponse,
    EntropyPoolingRequest,
    EntropyPoolingResponse,
    GenerateViewsRequest,
    GenerateViewsResponse,
)
from app.services.entropy_pooling_service import fetch_prices_df, run_entropy_pooling
from app.services.view_generation import (
    GeneratedViews,
    fetch_factor_data,
    generate_views,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/views", tags=["Views"])


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _get_view_repo(db: Session = Depends(get_db)) -> ViewGenerationRepository:
    return ViewGenerationRepository(db)


@router.post(
    "/generate",
    response_model=GenerateViewsResponse,
    summary="Generate Black-Litterman views from multi-factor asset data via LLM",
)
def generate_bl_views(
    request: GenerateViewsRequest,
    repo: ViewGenerationRepository = Depends(_get_view_repo),
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
        factor_data = fetch_factor_data(repo, tickers)
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
        if result.P.shape != (n_views, n_assets):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"P shape mismatch: expected ({n_views}, {n_assets}), got {result.P.shape}",
            )
        if result.Q.shape != (n_views,):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Q shape mismatch: expected ({n_views},), got {result.Q.shape}",
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


# ---------------------------------------------------------------------------
# Entropy Pooling endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/entropy-pooling",
    response_model=EntropyPoolingResponse,
    summary="Compute posterior moments via Entropy Pooling",
)
def entropy_pooling(
    request: EntropyPoolingRequest,
    db: Session = Depends(get_db),
) -> EntropyPoolingResponse:
    """Fetch historical prices, apply entropy pooling views, and return posterior moments.

    The endpoint:
    1. Fetches close prices from DB via the yfinance repository.
    2. Converts prices to returns via ``prices_to_returns()``.
    3. Builds an ``EntropyPoolingConfig`` from the request views.
    4. Fits the estimator and extracts ``return_distribution_`` (mu, covariance).

    **Returns 422** when tickers are missing or no price data is found.
    **Returns 500** when the entropy pooling solver fails to converge.
    """
    prices = fetch_prices_df(
        request.tickers,
        request.start_date,
        request.end_date,
        db,
    )

    if prices.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No price data found for tickers: {request.tickers}. "
                "Ensure the tickers exist in the database."
            ),
        )

    import pandas as pd
    from skfolio.preprocessing import prices_to_returns

    returns = pd.DataFrame(prices_to_returns(prices))

    try:
        result = run_entropy_pooling(
            returns=returns,
            mean_views=tuple(request.mean_views) if request.mean_views else None,
            variance_views=tuple(request.variance_views) if request.variance_views else None,
            correlation_views=tuple(request.correlation_views) if request.correlation_views else None,
            skew_views=tuple(request.skew_views) if request.skew_views else None,
            kurtosis_views=tuple(request.kurtosis_views) if request.kurtosis_views else None,
            cvar_views=tuple(request.cvar_views) if request.cvar_views else None,
        )
    except Exception as exc:
        logger.exception("Entropy pooling solver failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entropy pooling solver failed to converge: {exc}",
        ) from exc

    return EntropyPoolingResponse(
        mu=result["mu"],
        covariance=result["covariance"],
        tickers=result["tickers"],
    )
