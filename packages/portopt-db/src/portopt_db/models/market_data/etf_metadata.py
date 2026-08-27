"""SQLAlchemy models for ETF fund metadata (asset-class split, holdings,
sector weights, AUM/NAV).

T212 gives ETFs the same broker metadata as stocks and nothing fund-specific;
this data comes from yfinance ``funds_data`` + ``info``. One row-per-instrument
for the headline metadata, and point-in-time (``as_of``) rows for the
composition tables so history is retained. Every write is an idempotent upsert
on the natural keys declared below.
"""

from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Date,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from portopt_db.base import BaseModel

if TYPE_CHECKING:
    from portopt_db.models.universe.universe import Instrument


class ETFMetadata(BaseModel):
    """Headline fund metadata — one row per instrument."""

    __tablename__ = "etf_metadata"
    __table_args__ = (
        UniqueConstraint("instrument_id", name="uq_etf_metadata_instrument"),
        Index("ix_etf_metadata_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="etf_metadata")

    aum: Mapped[float | None] = mapped_column(Numeric(24, 2), nullable=True)
    nav: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    fund_family: Mapped[str | None] = mapped_column(String(255), nullable=True)
    legal_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    expense_ratio: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    base_currency: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # Fund overview (SPEC A8): Morningstar-style category + prospectus summary.
    category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    as_of: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)


class ETFAssetClass(BaseModel):
    """Asset-class allocation (stock/bond/cash/other %) at a point in time."""

    __tablename__ = "etf_asset_classes"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_asset_classes_instrument_asof"
        ),
        Index("ix_etf_asset_classes_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="etf_asset_classes")

    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    stock_pct: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    bond_pct: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    cash_pct: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    other_pct: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)


class ETFHolding(BaseModel):
    """Top-N constituent holding + weight at a point in time."""

    __tablename__ = "etf_holdings"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "as_of",
            "holding_symbol",
            name="uq_etf_holdings_instrument_asof_symbol",
        ),
        Index("ix_etf_holdings_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="etf_holdings")

    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    holding_symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    holding_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    weight: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)


class ETFSectorWeight(BaseModel):
    """Sector allocation weight at a point in time."""

    __tablename__ = "etf_sector_weights"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "as_of",
            "sector",
            name="uq_etf_sector_weights_instrument_asof_sector",
        ),
        Index("ix_etf_sector_weights_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="etf_sector_weights")

    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    sector: Mapped[str] = mapped_column(String(100), nullable=False)
    weight: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)


class ETFEquityHoldings(BaseModel):
    """Equity-holdings valuation metrics from funds_data.equity_holdings."""

    __tablename__ = "etf_equity_holdings"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_equity_holdings_instrument_asof"
        ),
        Index("ix_etf_equity_holdings_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    price_to_earnings: Mapped[float | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )
    price_to_book: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    price_to_sales: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    price_to_cashflow: Mapped[float | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )
    median_market_cap: Mapped[float | None] = mapped_column(
        Numeric(24, 2), nullable=True
    )
    three_year_earnings_growth: Mapped[float | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )


class ETFBondHoldings(BaseModel):
    """Bond-holdings metrics from funds_data.bond_holdings."""

    __tablename__ = "etf_bond_holdings"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_bond_holdings_instrument_asof"
        ),
        Index("ix_etf_bond_holdings_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    duration: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    maturity: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    credit_quality: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)


class ETFBondRating(BaseModel):
    """Credit-rating breakdown weight from funds_data.bond_ratings."""

    __tablename__ = "etf_bond_ratings"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "as_of",
            "rating",
            name="uq_etf_bond_ratings_instrument_asof_rating",
        ),
        Index("ix_etf_bond_ratings_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    rating: Mapped[str] = mapped_column(String(50), nullable=False)
    weight: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)


class ETFFundOperations(BaseModel):
    """Fund-operations metrics from funds_data.fund_operations."""

    __tablename__ = "etf_fund_operations"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_fund_operations_instrument_asof"
        ),
        Index("ix_etf_fund_operations_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    annual_report_expense_ratio: Mapped[float | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    annual_holdings_turnover: Mapped[float | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    total_net_assets: Mapped[float | None] = mapped_column(
        Numeric(24, 2), nullable=True
    )
