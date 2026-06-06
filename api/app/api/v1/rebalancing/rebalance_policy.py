"""FastAPI router for rebalancing policy CRUD and activate endpoints.

Routes:
  GET  /api/v1/portfolio/{name}/rebalance-policy            — list policies
  POST /api/v1/portfolio/{name}/rebalance-policy            — create policy
  POST /api/v1/portfolio/{name}/activate-policy/{policy_id} — activate policy
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.portfolio.portfolio_repository import PortfolioRepository
from app.repositories.rebalancing.rebalancing_repository import RebalancingRepository
from app.schemas.rebalancing.rebalancing import (
    RebalancingPolicyCreate,
    RebalancingPolicyListResponse,
    RebalancingPolicyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Rebalancing Policy"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_portfolio_or_404(name: str, db: Session):
    portfolio = PortfolioRepository(db).get_by_id_or_name(name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{name}' not found",
        )
    return portfolio


# ---------------------------------------------------------------------------
# GET /{name}/rebalance-policy
# ---------------------------------------------------------------------------


@router.get(
    "/{name}/rebalance-policy",
    response_model=RebalancingPolicyListResponse,
    response_model_by_alias=True,
    summary="List rebalancing policies for a portfolio",
)
def list_rebalance_policies(
    name: str,
    db: Session = Depends(get_db),
) -> RebalancingPolicyListResponse:
    """Return all rebalancing policies for the named portfolio.

    Returns 404 when the portfolio does not exist.
    """
    portfolio = _resolve_portfolio_or_404(name, db)
    repo = RebalancingRepository(db)
    policies = repo.get_policies_by_portfolio(portfolio.id)
    items = [RebalancingPolicyResponse.model_validate(p) for p in policies]
    return RebalancingPolicyListResponse(items=items, total=len(items))


# ---------------------------------------------------------------------------
# POST /{name}/rebalance-policy
# ---------------------------------------------------------------------------


@router.post(
    "/{name}/rebalance-policy",
    response_model=RebalancingPolicyResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
    summary="Create a rebalancing policy for a portfolio",
)
def create_rebalance_policy(
    name: str,
    body: RebalancingPolicyCreate,
    db: Session = Depends(get_db),
) -> RebalancingPolicyResponse:
    """Create a new rebalancing policy.

    Returns 404 when the portfolio does not exist.
    Returns 409 when a policy with the same name already exists for this portfolio.
    Returns 422 when policy_type is not calendar, threshold, or hybrid.
    """
    portfolio = _resolve_portfolio_or_404(name, db)
    repo = RebalancingRepository(db)
    try:
        policy = repo.create_policy(
            {
                "portfolio_id": portfolio.id,
                "name": body.name,
                "policy_type": body.policy_type,
                "config": body.config,
                "is_active": False,
            }
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"A policy named '{body.name}' already exists for portfolio '{name}'"
            ),
        ) from exc
    return RebalancingPolicyResponse.model_validate(policy)


# ---------------------------------------------------------------------------
# POST /{name}/activate-policy/{policy_id}
# ---------------------------------------------------------------------------


@router.post(
    "/{name}/activate-policy/{policy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Atomically activate a rebalancing policy for a portfolio",
)
def activate_rebalance_policy(
    name: str,
    policy_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """Activate a rebalancing policy, deactivating all others for the portfolio.

    Returns 404 when the portfolio does not exist.
    Returns 404 when policy_id does not exist or belongs to a different portfolio.
    """
    portfolio = _resolve_portfolio_or_404(name, db)
    repo = RebalancingRepository(db)

    policies = repo.get_policies_by_portfolio(portfolio.id)
    if not any(p.id == policy_id for p in policies):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy_id}' not found for portfolio '{name}'",
        )

    repo.activate_policy(policy_id, portfolio.id)
    db.commit()
