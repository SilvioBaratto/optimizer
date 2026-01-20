import uuid
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String,
    Integer,
    Float,
    DateTime,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import Enum as SQLEnum

from optimizer.database.models.base import BaseModel

# ============================================================================
# TRADING ECONOMICS DATA MODELS
# ============================================================================


class TradingEconomicsSnapshot(BaseModel):
    """
    Daily snapshot of Trading Economics data for a country.

    Stores metadata and single-value fields (industrial production,
    capacity utilization). Related indicators and bond yields are stored
    in separate tables for flexibility.
    """

    __tablename__ = "trading_economics_snapshots"

    country: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Country code (USA, Germany, France, UK, Japan)",
    )

    fetch_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When data was fetched from Trading Economics",
    )

    source_url: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Source URL for indicators page"
    )

    # Single-value indicators
    industrial_production_value: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Industrial production current value"
    )

    industrial_production_previous: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Industrial production previous value"
    )

    industrial_production_reference: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Industrial production reference date"
    )

    industrial_production_unit: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Industrial production unit"
    )

    capacity_utilization_value: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Capacity utilization current value"
    )

    capacity_utilization_previous: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Capacity utilization previous value"
    )

    capacity_utilization_reference: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Capacity utilization reference date"
    )

    capacity_utilization_unit: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Capacity utilization unit"
    )

    num_indicators: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of indicators fetched"
    )

    num_bond_yields: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of bond yields fetched"
    )

    # Relationships
    indicators: Mapped[List["TradingEconomicsIndicator"]] = relationship(
        "TradingEconomicsIndicator",
        back_populates="snapshot",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    bond_yields: Mapped[List["TradingEconomicsBondYield"]] = relationship(
        "TradingEconomicsBondYield",
        back_populates="snapshot",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    __table_args__ = (
        Index("idx_te_snapshot_country", "country"),
        Index("idx_te_snapshot_timestamp", "fetch_timestamp"),
        Index("idx_te_snapshot_country_timestamp", "country", "fetch_timestamp"),
        UniqueConstraint("country", "fetch_timestamp", name="uq_te_country_timestamp"),
        CheckConstraint("num_indicators >= 0", name="ck_num_indicators_positive"),
        CheckConstraint("num_bond_yields >= 0", name="ck_num_bond_yields_positive"),
        {
            "comment": "Daily Trading Economics data snapshot per country",
        },
    )

    def __repr__(self) -> str:
        return f"<TradingEconomicsSnapshot(country='{self.country}', timestamp='{self.fetch_timestamp}')>"


class TradingEconomicsIndicator(BaseModel):
    """
    Individual economic indicator from Trading Economics.

    Stores key indicators like GDP growth, unemployment, inflation, PMI, etc.
    with current and previous values.
    """

    __tablename__ = "trading_economics_indicators"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trading_economics_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to snapshot",
    )

    indicator_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Standardized indicator name (e.g., 'gdp_growth_rate')",
    )

    raw_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Original indicator name from Trading Economics",
    )

    value: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Current indicator value"
    )

    previous: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Previous indicator value"
    )

    unit: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Unit of measurement (%, points, etc.)"
    )

    reference: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Reference period (e.g., 'Aug/25')"
    )

    # Relationships
    snapshot: Mapped["TradingEconomicsSnapshot"] = relationship(
        "TradingEconomicsSnapshot", back_populates="indicators"
    )

    __table_args__ = (
        Index("idx_te_indicator_snapshot", "snapshot_id"),
        Index("idx_te_indicator_name", "indicator_name"),
        Index("idx_te_indicator_snapshot_name", "snapshot_id", "indicator_name"),
        UniqueConstraint(
            "snapshot_id", "indicator_name", name="uq_te_snapshot_indicator"
        ),
        {
            "comment": "Individual economic indicators from Trading Economics",
        },
    )

    def __repr__(self) -> str:
        return f"<TradingEconomicsIndicator(name='{self.indicator_name}', value={self.value})>"


class TradingEconomicsBondYield(BaseModel):
    """
    Government bond yields from Trading Economics.

    Stores bond yields for different maturities (2Y, 5Y, 10Y, 30Y)
    with daily/monthly/yearly changes.
    """

    __tablename__ = "trading_economics_bond_yields"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trading_economics_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to snapshot",
    )

    maturity: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
        comment="Bond maturity (2Y, 5Y, 10Y, 30Y)",
    )

    raw_name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Original bond name from Trading Economics"
    )

    yield_value: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bond yield (%)"
    )

    day_change: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Day change (%)"
    )

    month_change: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Month change (%)"
    )

    year_change: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Year change (%)"
    )

    date: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Reference date"
    )

    # Relationships
    snapshot: Mapped["TradingEconomicsSnapshot"] = relationship(
        "TradingEconomicsSnapshot", back_populates="bond_yields"
    )

    __table_args__ = (
        Index("idx_te_bond_snapshot", "snapshot_id"),
        Index("idx_te_bond_maturity", "maturity"),
        Index("idx_te_bond_snapshot_maturity", "snapshot_id", "maturity"),
        UniqueConstraint("snapshot_id", "maturity", name="uq_te_snapshot_maturity"),
        CheckConstraint("yield_value >= 0", name="ck_yield_positive"),
        {
            "comment": "Government bond yields from Trading Economics",
        },
    )

    def __repr__(self) -> str:
        return f"<TradingEconomicsBondYield(maturity='{self.maturity}', yield={self.yield_value}%)>"
