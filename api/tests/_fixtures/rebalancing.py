"""DB seed builder for the rebalancing domain (FK-dependent on ``Portfolio``).

Guards parent existence so an unflushed / ghost portfolio fails fast.
"""

from __future__ import annotations

from typing import NamedTuple

from sqlalchemy.orm import Session

from app.models.portfolio.portfolio import Portfolio
from app.models.rebalancing.rebalancing import RebalancingPolicy
from tests._fixtures._helpers import add_and_flush


class RebalancingSeed(NamedTuple):
    policy: RebalancingPolicy


def seed_rebalancing(
    session: Session,
    portfolio: Portfolio,
    *,
    name: str = "default-policy",
    policy_type: str = "calendar",
    config: dict | None = None,
) -> RebalancingSeed:
    assert session.get(Portfolio, portfolio.id) is not None, (
        "seed_rebalancing requires a flushed parent Portfolio"
    )
    policy = RebalancingPolicy(
        portfolio_id=portfolio.id,
        name=name,
        policy_type=policy_type,
        config=dict(config) if config is not None else {"interval_days": 21},
        is_active=False,
    )
    return RebalancingSeed(policy=add_and_flush(session, policy))
