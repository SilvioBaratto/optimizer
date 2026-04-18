"""FastAPI router for rebalancing decide and preview endpoints.

Routes:
  POST /api/v1/rebalance/decide          — stateless rebalancing decision
  GET  /api/v1/rebalance/preview/{name}  — trade preview using active DB policy
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.portfolio_repository import PortfolioRepository
from app.repositories.rebalancing_repository import RebalancingRepository
from app.schemas.rebalancing import (
    RebalanceDecideRequest,
    RebalanceDecideResponse,
    RebalancePreviewResponse,
    TradeItem,
)
from app.services.rebalancing_service import (
    build_trade_list,
    compute_broker_weights,
    decide,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rebalance", tags=["Rebalancing"])


# ---------------------------------------------------------------------------
# POST /rebalance/decide
# ---------------------------------------------------------------------------


@router.post(
    "/decide",
    response_model=RebalanceDecideResponse,
    response_model_by_alias=True,
)
def post_rebalance_decide(
    request: RebalanceDecideRequest,
) -> RebalanceDecideResponse:
    """Compute whether to rebalance given weights and a policy.

    Fully stateless — all inputs come from the request body.
    Returns should_rebalance flag, turnover, estimated cost, and trade weights.
    """
    result = decide(
        current_weights=request.current_weights,
        target_weights=request.target_weights,
        policy_type=request.policy_type,
        policy_config=request.policy_config,
        current_date=request.current_date,
        last_review_date=request.last_review_date,
        transaction_costs=request.transaction_costs,
    )
    return RebalanceDecideResponse(**result)


# ---------------------------------------------------------------------------
# GET /rebalance/preview/{portfolio_name}
# ---------------------------------------------------------------------------


@router.get(
    "/preview/{portfolio_name}",
    response_model=RebalancePreviewResponse,
    response_model_by_alias=True,
)
def get_rebalance_preview(
    portfolio_name: str,
    db: Session = Depends(get_db),
) -> RebalancePreviewResponse:
    """Produce a trade list from the latest snapshot and active policy.

    Returns 404 **only** when the portfolio itself does not exist. Empty-state
    semantics (issue #425): when the portfolio exists but an active policy or
    a snapshot is missing, the response is 200 with empty fields and a
    ``status`` reason (``'no_active_policy'`` / ``'no_snapshots'``). This lets
    the UI render a contextual empty state instead of an HTTP error toast.
    """
    portfolio_repo = PortfolioRepository(db)
    rebalancing_repo = RebalancingRepository(db)

    portfolio = portfolio_repo.get_by_name(portfolio_name)
    if portfolio is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_name}' not found",
        )

    policy = rebalancing_repo.get_active_policy(portfolio.id)
    if policy is None:
        return RebalancePreviewResponse(
            portfolio_name=portfolio_name,
            status="no_active_policy",
        )

    snapshot = portfolio_repo.get_latest_snapshot(portfolio.id)
    if snapshot is None:
        return RebalancePreviewResponse(
            portfolio_name=portfolio_name,
            policy_type=policy.policy_type,
            status="no_snapshots",
        )

    target_weights = snapshot.weights
    positions = list(portfolio_repo.get_positions(portfolio.id))

    account = portfolio_repo.get_latest_account_snapshot(portfolio.id)
    portfolio_value = account.total if account is not None else None

    broker_weights = compute_broker_weights(positions, portfolio_value)
    raw_trades = build_trade_list(target_weights, broker_weights) if positions else []
    trade_items = [TradeItem(**t) for t in raw_trades]

    return RebalancePreviewResponse(
        portfolio_name=portfolio_name,
        policy_type=policy.policy_type,
        target_weights=target_weights,
        current_weights=broker_weights,
        trades=trade_items,
        portfolio_value=portfolio_value,
    )
