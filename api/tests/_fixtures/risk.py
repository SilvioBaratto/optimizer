"""DB seed builder for the risk domain (FK-dependent on ``Portfolio``).

Guards parent existence so an unflushed / ghost portfolio fails fast
instead of producing an orphan FK at flush time.
"""

from __future__ import annotations

from typing import NamedTuple

from sqlalchemy.orm import Session

from app.models.portfolio.portfolio import Portfolio
from app.models.risk.risk import RiskLimit
from tests._fixtures._helpers import add_and_flush


class RiskSeed(NamedTuple):
    limit: RiskLimit


def seed_risk(
    session: Session,
    portfolio: Portfolio,
    *,
    metric: str = "volatility",
    limit_type: str = "upper",
    threshold: float = 0.20,
) -> RiskSeed:
    assert session.get(Portfolio, portfolio.id) is not None, (
        "seed_risk requires a flushed parent Portfolio"
    )
    limit = RiskLimit(
        portfolio_id=portfolio.id,
        metric=metric,
        limit_type=limit_type,
        threshold=threshold,
        is_breached=False,
    )
    return RiskSeed(limit=add_and_flush(session, limit))
