"""SQLAlchemy models for universe building (exchanges and instruments)."""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from portopt_db.base import BaseModel

if TYPE_CHECKING:
    from portopt_db.models.market_data.etf_metadata import (
        ETFAssetClass,
        ETFHolding,
        ETFMetadata,
        ETFSectorWeight,
    )
    from portopt_db.models.market_data.yfinance_data import (
        AnalystPriceTarget,
        AnalystRecommendation,
        Dividend,
        FinancialStatement,
        InsiderTransaction,
        InstitutionalHolder,
        MutualFundHolder,
        PriceHistory,
        StockSplit,
        TickerNews,
        TickerProfile,
    )


class Exchange(BaseModel):
    __tablename__ = "exchanges"

    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    t212_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    instruments: Mapped[list[Instrument]] = relationship(
        back_populates="exchange", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Exchange(name={self.name!r})>"


class Instrument(BaseModel):
    __tablename__ = "instruments"
    __table_args__ = (
        UniqueConstraint("ticker", "exchange_id", name="uq_instrument_ticker_exchange"),
        Index("ix_instrument_exchange_id", "exchange_id"),
        Index("ix_instrument_isin", "isin"),
        Index("ix_instrument_yfinance_ticker", "yfinance_ticker"),
        # Pinned to match migration d4e5f6a7b8c9 (singular "ix_instrument_*"
        # family). A bare index=True on the column makes SQLAlchemy expect
        # "ix_instruments_delisted_at", causing autogenerate to emit a
        # needless drop+recreate.
        Index("ix_instrument_delisted_at", "delisted_at"),
    )

    ticker: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    short_name: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    isin: Mapped[str | None] = mapped_column(String(20), nullable=True)
    instrument_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    currency_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    yfinance_ticker: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Optional Trading 212 ticker, populated by the T212 annotation step (D14).
    # yfinance is the source of truth; this is a convenience cross-reference.
    t212_ticker: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Asset-class taxonomy (see services/universe/trading212/enums.py). Stored as
    # strings like instrument_type. asset_class is non-null (STOCK ⇒ "equity");
    # fi_subclass/duration_bucket are populated only for fixed_income.
    asset_class: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="equity"
    )
    fi_subclass: Mapped[str | None] = mapped_column(String(20), nullable=True)
    duration_bucket: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Survivorship-bias correction: populated when an instrument drops out of
    # the active Trading 212 universe.
    delisted_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    delisting_return: Mapped[float | None] = mapped_column(Float, nullable=True)

    exchange_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exchanges.id", ondelete="CASCADE"), nullable=False
    )
    exchange: Mapped[Exchange] = relationship(back_populates="instruments")

    # Yfinance data relationships
    profiles: Mapped[list[TickerProfile]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    price_history: Mapped[list[PriceHistory]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    financial_statements: Mapped[list[FinancialStatement]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    dividends: Mapped[list[Dividend]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    stock_splits: Mapped[list[StockSplit]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    analyst_recommendations: Mapped[list[AnalystRecommendation]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    analyst_price_targets: Mapped[list[AnalystPriceTarget]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    institutional_holders: Mapped[list[InstitutionalHolder]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    mutual_fund_holders: Mapped[list[MutualFundHolder]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    insider_transactions: Mapped[list[InsiderTransaction]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    news: Mapped[list[TickerNews]] = relationship(
        back_populates="instrument", passive_deletes=True
    )

    # ETF fund metadata (populated only for ETF instruments)
    etf_metadata: Mapped[list[ETFMetadata]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    etf_asset_classes: Mapped[list[ETFAssetClass]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    etf_holdings: Mapped[list[ETFHolding]] = relationship(
        back_populates="instrument", passive_deletes=True
    )
    etf_sector_weights: Mapped[list[ETFSectorWeight]] = relationship(
        back_populates="instrument", passive_deletes=True
    )

    @property
    def exchange_name(self) -> str | None:
        return self.exchange.name if self.exchange else None

    def __repr__(self) -> str:
        return f"<Instrument(ticker={self.ticker!r}, short_name={self.short_name!r})>"
