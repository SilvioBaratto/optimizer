"""FastAPI router for risk analytics endpoints (issues #368, #369).

Routes:
  GET /portfolio/{name}/risk/var             — VaR / CVaR at 90/95/99%
  GET /portfolio/{name}/risk/correlation     — clustered correlation matrix
  GET /portfolio/{name}/risk/factor-exposure — portfolio-weighted factor exposures
  GET /portfolio/{name}/risk/concentration   — HHI, effective-N, per-asset weights
  GET /portfolio/{name}/risk/liquidity       — ADDV-based days-to-liquidate per asset

All endpoints are synchronous (SQLAlchemy Session, not AsyncSession).
Pattern: Routes → Services → Repositories → Models.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.portfolio.portfolio_repository import PortfolioRepository
from app.schemas.risk.risk_analytics import (
    ConcentrationResponse,
    CorrelationResponse,
    FactorExposureResponse,
    LiquidityResponse,
    VarResponse,
)
from app.services.risk.risk_analytics_service import RiskAnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Risk Analytics"])


# ---------------------------------------------------------------------------
# Query param types
# ---------------------------------------------------------------------------


class VarMethod(str, Enum):
    historical = "historical"
    parametric = "parametric"


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_portfolio_repository(db: Session = Depends(get_db)) -> PortfolioRepository:
    return PortfolioRepository(db)


def get_risk_analytics_service(db: Session = Depends(get_db)) -> RiskAnalyticsService:
    return RiskAnalyticsService(db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_portfolio_or_404(name: str, repo: PortfolioRepository):  # type: ignore[return]
    portfolio = repo.get_by_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )
    return portfolio


def _get_snapshot_weights_or_404(
    name: str, portfolio, repo: PortfolioRepository
) -> dict:
    snap = repo.get_latest_snapshot(portfolio.id)
    if snap is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No snapshots found for portfolio '{name}'. Create a snapshot first.",
        )
    return snap.weights


# ---------------------------------------------------------------------------
# VaR / CVaR endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{name}/risk/var",
    response_model=VarResponse,
    summary="Compute VaR and CVaR at 90/95/99% confidence levels",
)
def get_var(
    name: str,
    lookback: Annotated[
        int, Query(ge=1, description="Lookback window in trading days")
    ] = 252,
    method: VarMethod = VarMethod.historical,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
    service: RiskAnalyticsService = Depends(get_risk_analytics_service),
) -> VarResponse:
    """Compute historical or parametric VaR and CVaR for the latest portfolio snapshot."""
    portfolio = _get_portfolio_or_404(name, repo)
    weights = _get_snapshot_weights_or_404(name, portfolio, repo)

    try:
        result = service.compute_var(
            portfolio_id=str(portfolio.id),
            weights=weights,
            lookback=lookback,
            method=method.value,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return VarResponse(**result)


# ---------------------------------------------------------------------------
# Correlation endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{name}/risk/correlation",
    response_model=CorrelationResponse,
    summary="Compute block-clustered correlation matrix for portfolio assets",
)
def get_correlation(
    name: str,
    lookback: Annotated[
        int, Query(ge=1, description="Lookback window in trading days")
    ] = 252,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
    service: RiskAnalyticsService = Depends(get_risk_analytics_service),
) -> CorrelationResponse:
    """Return a hierarchically clustered correlation matrix for the latest portfolio."""
    portfolio = _get_portfolio_or_404(name, repo)
    weights = _get_snapshot_weights_or_404(name, portfolio, repo)

    try:
        result = service.compute_correlation(
            portfolio_id=str(portfolio.id),
            weights=weights,
            lookback=lookback,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return CorrelationResponse(**result)


# ---------------------------------------------------------------------------
# Factor exposure endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{name}/risk/factor-exposure",
    response_model=FactorExposureResponse,
    summary="Compute portfolio-weighted factor exposures from stored factor scores",
)
def get_factor_exposure(
    name: str,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
    service: RiskAnalyticsService = Depends(get_risk_analytics_service),
) -> FactorExposureResponse:
    """Return portfolio-weighted factor exposures based on stored factor scores.

    Empty-state semantics (issue #424): when no factor scores exist for the
    portfolio's tickers, the service returns empty ``exposures`` / ``asset_exposures``
    mappings and this route responds 200. The 404 path is reserved for the
    portfolio / snapshot not existing (the real not-found resources).
    """
    portfolio = _get_portfolio_or_404(name, repo)
    weights = _get_snapshot_weights_or_404(name, portfolio, repo)
    result = service.compute_factor_exposure(
        portfolio_id=str(portfolio.id),
        weights=weights,
    )
    return FactorExposureResponse(**result)


# ---------------------------------------------------------------------------
# Concentration endpoint (issue #369)
# ---------------------------------------------------------------------------


@router.get(
    "/{name}/risk/concentration",
    response_model=ConcentrationResponse,
    summary="Compute per-asset concentration breakdown and HHI/effective-N summary",
)
def get_concentration(
    name: str,
    n: Annotated[int, Query(ge=1, description="Top-N threshold for top_n_ratio")] = 5,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
    service: RiskAnalyticsService = Depends(get_risk_analytics_service),
) -> ConcentrationResponse:
    """Return per-asset weights and portfolio-level HHI, effective-N, and top-N ratio."""
    portfolio = _get_portfolio_or_404(name, repo)
    weights = _get_snapshot_weights_or_404(name, portfolio, repo)
    result = service.compute_concentration(weights=weights, n=n)
    return ConcentrationResponse(**result)


# ---------------------------------------------------------------------------
# Liquidity endpoint (issue #369)
# ---------------------------------------------------------------------------


@router.get(
    "/{name}/risk/liquidity",
    response_model=LiquidityResponse,
    summary="Compute ADDV-based days-to-liquidate and liquidity cost per asset",
)
def get_liquidity(
    name: str,
    lookback_days: Annotated[
        int, Query(ge=1, description="Lookback window in trading days for ADDV")
    ] = 20,
    participation_rate: Annotated[
        float,
        Query(gt=0, le=1, description="Daily participation rate (0, 1]"),
    ] = 0.1,
    repo: PortfolioRepository = Depends(get_portfolio_repository),
    service: RiskAnalyticsService = Depends(get_risk_analytics_service),
) -> LiquidityResponse:
    """Return per-asset ADDV, days-to-liquidate, liquidity cost, and portfolio weighted average."""
    portfolio = _get_portfolio_or_404(name, repo)
    weights = _get_snapshot_weights_or_404(name, portfolio, repo)
    result = service.compute_liquidity(
        weights=weights,
        lookback_days=lookback_days,
        participation_rate=participation_rate,
    )
    return LiquidityResponse(**result)
