"""SQLAlchemy ORM model for rebalancing policy persistence."""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, Boolean, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models._shared import BaseModel

# JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class RebalancingPolicy(BaseModel):
    """Rebalancing scheduling configuration for a portfolio.

    Unique constraint on (portfolio_id, name) prevents duplicate policy names
    within the same portfolio. Active-policy enforcement is handled at the
    service layer to allow flexible transitions.
    """

    __tablename__ = "rebalancing_policies"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id",
            "name",
            name="uq_rebalancing_policies_portfolio_name",
        ),
        Index("ix_rebalancing_policies_portfolio_id", "portfolio_id"),
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    policy_type: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[dict] = mapped_column(_JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="0",
    )

    # Forward-only relationship — no back_populates to avoid modifying portfolio.py
    portfolio: Mapped["Portfolio"] = relationship(  # type: ignore[name-defined]  # noqa: F821, UP037
        "Portfolio",
        foreign_keys=[portfolio_id],
    )
