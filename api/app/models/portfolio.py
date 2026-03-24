"""SQLAlchemy models for portfolio state persistence."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


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
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    portfolio: Mapped[Portfolio] = relationship(back_populates="snapshots")
    weight_entries: Mapped[list[SnapshotWeight]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", lazy="selectin",
    )
    sector_mapping_entries: Mapped[list[SnapshotSectorMapping]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", lazy="selectin",
    )
    summary_entries: Mapped[list[SnapshotSummaryEntry]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", lazy="selectin",
    )
    optimizer_config_entries: Mapped[list[SnapshotOptimizerParam]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", lazy="selectin",
    )

    @property
    def weights(self) -> dict[str, float]:
        return {e.ticker: e.weight for e in self.weight_entries}

    @property
    def sector_mapping(self) -> dict[str, str] | None:
        if not self.sector_mapping_entries:
            return None
        return {e.ticker: e.sector for e in self.sector_mapping_entries}

    @property
    def summary(self) -> dict[str, Any] | None:
        if not self.summary_entries:
            return None
        return {e.key: json.loads(e.value_text) for e in self.summary_entries}

    @property
    def optimizer_config(self) -> dict[str, Any] | None:
        if not self.optimizer_config_entries:
            return None
        return {e.key: json.loads(e.value_text) for e in self.optimizer_config_entries}


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
    """Snapshot of brokerage account cash/value state.

    Deduplicated per (portfolio_id, snapshot_date): at most one row per
    portfolio per calendar day.  Repeated syncs on the same day update the
    existing row rather than inserting a new one, preventing unbounded growth.
    """

    __tablename__ = "broker_account_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "snapshot_date",
            name="uq_broker_account_snapshot_portfolio_date",
        ),
        Index("ix_broker_account_portfolio_id", "portfolio_id"),
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
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
        UniqueConstraint("external_id", name="uq_activity_event_external_id"),
        Index("ix_activity_events_portfolio_id", "portfolio_id"),
        Index("ix_activity_events_event_type", "event_type"),
        Index("ix_activity_events_created_at", "created_at"),
        Index("ix_activity_events_external_id", "external_id"),
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
    external_id: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
    )  # Broker order_id for idempotent sync deduplication

    # Relationships
    portfolio: Mapped[Portfolio | None] = relationship(back_populates="events")
    metadata_entries: Mapped[list[ActivityEventDetail]] = relationship(
        back_populates="event", cascade="all, delete-orphan", lazy="selectin",
    )

    @property
    def metadata_(self) -> dict | None:
        if not self.metadata_entries:
            return None
        return {e.key: json.loads(e.value_text) for e in self.metadata_entries}


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
    model_type: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default="hmm",
    )
    # Scalar columns extracted from the old metadata JSONB
    since: Mapped[date | None] = mapped_column(Date, nullable=True)
    n_states: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_fitted: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    probability_entries: Mapped[list[RegimeStateProbability]] = relationship(
        back_populates="regime_state", cascade="all, delete-orphan", lazy="selectin",
    )

    @property
    def probabilities(self) -> list[dict]:
        return [
            {"regime": e.regime, "probability": e.probability}
            for e in self.probability_entries
        ]

    @property
    def metadata_(self) -> dict | None:
        if self.since is None and self.n_states is None and self.last_fitted is None:
            return None
        d: dict[str, Any] = {}
        if self.since is not None:
            d["since"] = self.since.isoformat()
        if self.n_states is not None:
            d["n_states"] = self.n_states
        if self.last_fitted is not None:
            d["last_fitted"] = self.last_fitted.isoformat()
        return d


# ------------------------------------------------------------------
# Child tables (normalized from JSONB columns)
# ------------------------------------------------------------------


class SnapshotWeight(BaseModel):
    """Individual weight entry for a portfolio snapshot."""

    __tablename__ = "snapshot_weights"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "ticker", name="uq_snapshot_weight_ticker"),
        Index("ix_snapshot_weights_snapshot_id", "snapshot_id"),
    )

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    ticker: Mapped[str] = mapped_column(String(100), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)

    snapshot: Mapped[PortfolioSnapshot] = relationship(back_populates="weight_entries")


class SnapshotSectorMapping(BaseModel):
    """Sector mapping entry for a portfolio snapshot."""

    __tablename__ = "snapshot_sector_mappings"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id", "ticker", name="uq_snapshot_sector_mapping_ticker",
        ),
        Index("ix_snapshot_sector_mappings_snapshot_id", "snapshot_id"),
    )

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    ticker: Mapped[str] = mapped_column(String(100), nullable=False)
    sector: Mapped[str] = mapped_column(String(200), nullable=False)

    snapshot: Mapped[PortfolioSnapshot] = relationship(
        back_populates="sector_mapping_entries",
    )


class SnapshotSummaryEntry(BaseModel):
    """EAV entry for snapshot summary data."""

    __tablename__ = "snapshot_summary_entries"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "key", name="uq_snapshot_summary_key"),
        Index("ix_snapshot_summary_entries_snapshot_id", "snapshot_id"),
    )

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    value_text: Mapped[str] = mapped_column(Text, nullable=False)

    snapshot: Mapped[PortfolioSnapshot] = relationship(
        back_populates="summary_entries",
    )


class SnapshotOptimizerParam(BaseModel):
    """EAV entry for snapshot optimizer configuration."""

    __tablename__ = "snapshot_optimizer_params"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id", "key", name="uq_snapshot_optimizer_param_key",
        ),
        Index("ix_snapshot_optimizer_params_snapshot_id", "snapshot_id"),
    )

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    value_text: Mapped[str] = mapped_column(Text, nullable=False)

    snapshot: Mapped[PortfolioSnapshot] = relationship(
        back_populates="optimizer_config_entries",
    )


class ActivityEventDetail(BaseModel):
    """EAV entry for activity event metadata."""

    __tablename__ = "activity_event_metadata"
    __table_args__ = (
        UniqueConstraint("event_id", "key", name="uq_activity_event_detail_key"),
        Index("ix_activity_event_metadata_event_id", "event_id"),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("activity_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    value_text: Mapped[str] = mapped_column(Text, nullable=False)

    event: Mapped[ActivityEvent] = relationship(back_populates="metadata_entries")


class RegimeStateProbability(BaseModel):
    """Individual regime probability for a regime state."""

    __tablename__ = "regime_state_probabilities"
    __table_args__ = (
        UniqueConstraint(
            "regime_state_id", "regime", name="uq_regime_state_prob_regime",
        ),
        Index("ix_regime_state_probs_state_id", "regime_state_id"),
    )

    regime_state_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("regime_states.id", ondelete="CASCADE"),
        nullable=False,
    )
    regime: Mapped[str] = mapped_column(String(20), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    regime_state: Mapped[RegimeState] = relationship(
        back_populates="probability_entries",
    )
