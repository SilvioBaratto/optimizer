"""SQLAlchemy models for portfolio state persistence."""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class Portfolio(BaseModel):
    """Top-level portfolio entity."""

    __tablename__ = "portfolios"
    __table_args__ = (
        UniqueConstraint("name", name="uq_portfolio_name"),
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    currency: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default="EUR",
    )
    benchmark_ticker: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="SPY",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true",
    )

    # Relationships
    snapshots: Mapped[list["PortfolioSnapshot"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan",
    )
    positions: Mapped[list["BrokerPosition"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan",
    )
    account_snapshots: Mapped[list["BrokerAccountSnapshot"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan",
    )
    events: Mapped[list["ActivityEvent"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan",
    )


class PortfolioSnapshot(BaseModel):
    """Point-in-time snapshot of portfolio weights from optimizer or rebalance."""

    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "snapshot_date", "snapshot_type",
            name="uq_snapshot_portfolio_date_type",
        ),
        Index("ix_snapshots_portfolio_id", "portfolio_id"),
        Index("ix_snapshots_date", "snapshot_date"),
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    snapshot_type: Mapped[str] = mapped_column(
        String(30), nullable=False,
    )  # "optimization" | "rebalance" | "manual"
    weights: Mapped[dict] = mapped_column(_JSON, nullable=False)
    sector_mapping: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    summary: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    optimizer_config: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationship
    portfolio: Mapped["Portfolio"] = relationship(back_populates="snapshots")


class BrokerPosition(BaseModel):
    """Real brokerage position synced from Trading 212."""

    __tablename__ = "broker_positions"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "ticker",
            name="uq_broker_position_portfolio_ticker",
        ),
        Index("ix_broker_positions_portfolio_id", "portfolio_id"),
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )
    ticker: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )  # T212 ticker (e.g. "ORAp_EQ")
    yfinance_ticker: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
    )
    name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    average_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ppl: Mapped[float | None] = mapped_column(Float, nullable=True)
    fx_ppl: Mapped[float | None] = mapped_column(Float, nullable=True)
    initial_fill_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    # Relationship
    portfolio: Mapped["Portfolio"] = relationship(back_populates="positions")


class BrokerAccountSnapshot(BaseModel):
    """Snapshot of brokerage account cash/value state."""

    __tablename__ = "broker_account_snapshots"
    __table_args__ = (
        Index("ix_broker_account_portfolio_id", "portfolio_id"),
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )
    total: Mapped[float] = mapped_column(Float, nullable=False)
    free: Mapped[float] = mapped_column(Float, nullable=False)
    invested: Mapped[float] = mapped_column(Float, nullable=False)
    blocked: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    # Relationship
    portfolio: Mapped["Portfolio"] = relationship(back_populates="account_snapshots")


class ActivityEvent(BaseModel):
    """Cross-cutting activity log for portfolio events."""

    __tablename__ = "activity_events"
    __table_args__ = (
        Index("ix_activity_events_portfolio_id", "portfolio_id"),
        Index("ix_activity_events_event_type", "event_type"),
        Index("ix_activity_events_created_at", "created_at"),
    )

    portfolio_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(30), nullable=False,
    )  # matches frontend ActivityType
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", _JSON, nullable=True,
    )

    # Relationship
    portfolio: Mapped["Portfolio | None"] = relationship(back_populates="events")


class RegimeState(BaseModel):
    """Cached HMM regime state for the market regime endpoint."""

    __tablename__ = "regime_states"
    __table_args__ = (
        UniqueConstraint(
            "state_date", "model_type",
            name="uq_regime_state_date_model",
        ),
    )

    state_date: Mapped[date] = mapped_column(Date, nullable=False)
    regime: Mapped[str] = mapped_column(
        String(20), nullable=False,
    )  # "bull" | "bear" | "sideways" | "volatile"
    probabilities: Mapped[dict] = mapped_column(_JSON, nullable=False)
    model_type: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default="hmm",
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", _JSON, nullable=True,
    )
