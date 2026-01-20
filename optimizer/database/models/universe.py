"""
Universe Models - Trading212 Universe Database Schema
=====================================================
SQLAlchemy models for storing universe data from build_universe.py

Tables:
- Exchange: Stock exchanges with metadata
- Instrument: Individual stocks/securities with yfinance mappings

Features:
- UUID primary keys for all tables
- Foreign key relationships
- Timestamps for data freshness tracking
- Filter status tracking for data quality
- Indexed columns for query performance
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from optimizer.database.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from optimizer.database.models.stock_signals import StockSignal


class Exchange(Base, TimestampMixin):
    """
    Stock exchange entity from Trading212 API.

    Represents a trading venue (e.g., NYSE, NASDAQ, Deutsche Börse Xetra)
    with its metadata and associated instruments.
    """

    __tablename__ = "exchanges"

    # Primary Key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="UUID primary key",
    )

    # Exchange Identifiers
    exchange_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
        index=True,
        comment="Trading212 exchange ID",
    )

    exchange_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Exchange name (e.g., 'NYSE', 'NASDAQ')",
    )

    # Metadata
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether exchange is currently active",
    )

    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time exchange data was refreshed",
    )

    # Relationships
    instruments: Mapped[list["Instrument"]] = relationship(
        "Instrument",
        back_populates="exchange",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Indexes
    __table_args__ = (
        Index("idx_exchange_name", "exchange_name"),
        Index("idx_exchange_id", "exchange_id"),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return f"<Exchange(id={self.id}, name='{self.exchange_name}', t212_id={self.exchange_id})>"


class Instrument(Base, TimestampMixin):
    """
    Individual stock/security from Trading212 with yfinance mapping.

    Represents a tradeable instrument with:
    - Trading212 metadata (ticker, ISIN, currency, etc.)
    - yfinance ticker mapping for data fetching
    - Filter status from build_universe.py basic filters
    - Historical data availability flags
    """

    __tablename__ = "instruments"

    # Primary Key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="UUID primary key",
    )

    # Foreign Key to Exchange
    exchange_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exchanges.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to parent exchange",
    )

    # Trading212 Identifiers
    ticker: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Trading212 ticker symbol (e.g., 'AAPL_US_EQ')",
    )

    short_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Short ticker name (e.g., 'AAPL')",
    )

    name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, comment="Full company/instrument name"
    )

    isin: Mapped[Optional[str]] = mapped_column(
        String(12),
        nullable=True,
        index=True,
        comment="ISIN (International Securities Identification Number)",
    )

    # Instrument Type
    instrument_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="STOCK",
        comment="Instrument type (STOCK, ETF, etc.)",
    )

    # Currency and Trading Info
    currency_code: Mapped[Optional[str]] = mapped_column(
        String(3), nullable=True, comment="Trading currency (e.g., 'USD', 'EUR', 'GBP')"
    )

    max_open_quantity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Maximum open quantity allowed by Trading212"
    )

    added_on: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Date instrument was added to Trading212",
    )

    # yfinance Mapping
    yfinance_ticker: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        comment="Yahoo Finance ticker (e.g., 'AAPL', 'AAPL.L')",
    )

    # Metadata
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether instrument is currently active",
    )

    last_validated: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time instrument data was validated",
    )

    # Relationships
    exchange: Mapped["Exchange"] = relationship(
        "Exchange", back_populates="instruments"
    )

    signals: Mapped[list["StockSignal"]] = relationship(
        "StockSignal",
        back_populates="instrument",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Indexes for common queries
    __table_args__ = (
        # Unique constraint on T212 ticker
        UniqueConstraint("ticker", name="uq_instruments_ticker"),
        # Indexes for filtering and lookups
        Index("idx_instruments_ticker", "ticker"),
        Index("idx_instruments_short_name", "short_name"),
        Index("idx_instruments_yfinance_ticker", "yfinance_ticker"),
        Index("idx_instruments_isin", "isin"),
        Index("idx_instruments_exchange_id", "exchange_id"),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return f"<Instrument(id={self.id}, ticker='{self.ticker}', yf='{self.yfinance_ticker}')>"

    def to_dict_summary(self) -> dict:
        """Return a summary dictionary of the instrument (for API responses)."""
        return {
            "id": str(self.id),
            "ticker": self.ticker,
            "short_name": self.short_name,
            "name": self.name,
            "yfinance_ticker": self.yfinance_ticker,
            "exchange_name": self.exchange.exchange_name if self.exchange else None,
        }
