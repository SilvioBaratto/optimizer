"""FastAPI router for risk limits CRUD endpoints (issue #370).

Routes are resource-centric under /risk/{portfolio_name}/limits.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.risk import RiskLimit
from app.repositories.portfolio_repository import PortfolioRepository
from app.repositories.risk_repository import RiskRepository
from app.schemas.risk import (
    RiskLimitCreate,
    RiskLimitListResponse,
    RiskLimitResponse,
    RiskLimitUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{portfolio_name}/limits",
    response_model=RiskLimitListResponse,
    summary="List risk limits for a portfolio",
)
def list_limits(
    portfolio_name: str,
    refresh: bool = Query(default=False, description="Re-evaluate breach flags from stored values"),
    db: Session = Depends(get_db),
) -> RiskLimitListResponse:
    """Return all risk limits for a portfolio.

    When ``refresh=true``, re-evaluates ``is_breached`` from ``current_value``
    and ``threshold`` without fetching new external data.
    """
    portfolio = _resolve_portfolio_or_404(portfolio_name, db)
    risk_repo = RiskRepository(db)
    limits = risk_repo.get_limits_by_portfolio(portfolio.id)

    if refresh:
        limits = _refresh_breach_flags(limits, risk_repo)
        db.commit()

    items = [RiskLimitResponse.model_validate(lim) for lim in limits]
    breach_count = sum(1 for item in items if item.is_breached)
    return RiskLimitListResponse(items=items, breach_count=breach_count)


@router.post(
    "/{portfolio_name}/limits",
    response_model=RiskLimitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a risk limit for a portfolio",
)
def create_limit(
    portfolio_name: str,
    body: RiskLimitCreate,
    db: Session = Depends(get_db),
) -> RiskLimitResponse:
    """Create a new risk limit.

    Returns 409 Conflict when a limit with the same portfolio + metric + limit_type
    already exists (unique constraint violation).
    """
    portfolio = _resolve_portfolio_or_404(portfolio_name, db)
    risk_repo = RiskRepository(db)
    try:
        limit = risk_repo.create_limit(
            {
                "portfolio_id": portfolio.id,
                "metric": body.metric,
                "limit_type": body.limit_type,
                "threshold": body.threshold,
                "is_breached": False,
            }
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Risk limit for metric '{body.metric}' with type "
                f"'{body.limit_type}' already exists for portfolio '{portfolio_name}'"
            ),
        ) from exc
    return RiskLimitResponse.model_validate(limit)


@router.put(
    "/{portfolio_name}/limits/{limit_id}",
    response_model=RiskLimitResponse,
    summary="Update current value and breach flag of a risk limit",
)
def update_limit(
    portfolio_name: str,
    limit_id: uuid.UUID,
    body: RiskLimitUpdate,
    db: Session = Depends(get_db),
) -> RiskLimitResponse:
    """Update ``current_value`` and ``is_breached`` for an existing risk limit."""
    _resolve_portfolio_or_404(portfolio_name, db)
    risk_repo = RiskRepository(db)
    try:
        limit = risk_repo.update_limit_value(
            limit_id=limit_id,
            current_value=body.current_value,
            is_breached=body.is_breached,
        )
        db.commit()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Risk limit '{limit_id}' not found",
        ) from exc
    return RiskLimitResponse.model_validate(limit)


@router.delete(
    "/{portfolio_name}/limits/{limit_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a risk limit",
)
def delete_limit(
    portfolio_name: str,
    limit_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """Delete a risk limit by ID. Returns 404 if the limit does not exist."""
    _resolve_portfolio_or_404(portfolio_name, db)
    risk_repo = RiskRepository(db)
    deleted = risk_repo.delete_limit(limit_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Risk limit '{limit_id}' not found",
        )
    db.commit()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_portfolio_or_404(portfolio_name: str, db: Session):
    """Lookup portfolio by name; raise 404 if not found."""
    repo = PortfolioRepository(db)
    portfolio = repo.get_by_name(portfolio_name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_name}' not found",
        )
    return portfolio


def _refresh_breach_flags(
    limits: list[RiskLimit], risk_repo: RiskRepository
) -> list[RiskLimit]:
    """Re-evaluate is_breached from stored current_value for each limit.

    upper → breached when current_value > threshold
    lower → breached when current_value < threshold
    Limits without a current_value are left unchanged.
    """
    refreshed: list[RiskLimit] = []
    for lim in limits:
        if lim.current_value is None:
            refreshed.append(lim)
            continue
        if lim.limit_type == "upper":
            is_breached = lim.current_value > lim.threshold
        else:
            is_breached = lim.current_value < lim.threshold
        if is_breached != lim.is_breached:
            lim = risk_repo.update_limit_value(lim.id, lim.current_value, is_breached)
        refreshed.append(lim)
    return refreshed
