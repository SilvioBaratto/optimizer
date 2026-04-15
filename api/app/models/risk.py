"""SQLAlchemy ORM model for risk limit persistence."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class RiskLimit(BaseModel):
    """Portfolio risk constraint with breach detection.

    Unique constraint on (portfolio_id, metric, limit_type) prevents
    duplicate limit definitions for the same risk measure per portfolio.
    """

    __tablename__ = "risk_limits"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id",
            "metric",
            "limit_type",
            name="uq_risk_limits_portfolio_metric_limit_type",
        ),
        Index("ix_risk_limits_portfolio_id", "portfolio_id"),
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric: Mapped[str] = mapped_column(String(100), nullable=False)
    limit_type: Mapped[str] = mapped_column(String(20), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    current_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_breached: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="0",
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Forward-only relationship — no back_populates to avoid modifying portfolio.py
    portfolio: Mapped["Portfolio"] = relationship(  # type: ignore[name-defined]  # noqa: F821, UP037
        "Portfolio",
        foreign_keys=[portfolio_id],
    )
