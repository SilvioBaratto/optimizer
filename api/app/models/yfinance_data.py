"""SQLAlchemy models for yfinance data storage."""

from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.universe import Instrument


class TickerProfile(BaseModel):
    """Company profile/info from yf.Ticker.info."""

    __tablename__ = "ticker_profiles"
    __table_args__ = (
        UniqueConstraint("instrument_id", name="uq_ticker_profile_instrument"),
        Index("ix_ticker_profiles_sector", "sector"),
        Index("ix_ticker_profiles_industry", "industry"),
        Index("ix_ticker_profiles_country", "country"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="profiles")

    # Identifiers
    symbol: Mapped[str | None] = mapped_column(String(50), nullable=True)
    short_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    long_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    isin: Mapped[str | None] = mapped_column(String(20), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(50), nullable=True)
    quote_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Classification
    sector: Mapped[str | None] = mapped_column(String(200), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(200), nullable=True)
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    website: Mapped[str | None] = mapped_column(String(500), nullable=True)
    long_business_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Market data
    market_cap: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    enterprise_value: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    shares_outstanding: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    float_shares: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    implied_shares_outstanding: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )

    # Price & volume
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    day_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    day_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    fifty_two_week_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    fifty_two_week_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    fifty_day_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    two_hundred_day_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    average_volume_10days: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    regular_market_volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    bid_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ask_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    beta: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Valuation ratios
    trailing_pe: Mapped[float | None] = mapped_column(Float, nullable=True)
    forward_pe: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    forward_eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_to_sales_trailing_12months: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    price_to_book: Mapped[float | None] = mapped_column(Float, nullable=True)
    enterprise_to_revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    enterprise_to_ebitda: Mapped[float | None] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    book_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Profitability
    profit_margins: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_margins: Mapped[float | None] = mapped_column(Float, nullable=True)
    gross_margins: Mapped[float | None] = mapped_column(Float, nullable=True)
    ebitda_margins: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_on_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_on_equity: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Income & revenue
    total_revenue: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    revenue_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_quarterly_growth: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    ebitda: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    gross_profits: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    free_cashflow: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    operating_cashflow: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_cash: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_cash_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_debt: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    debt_to_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    quick_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Dividends
    dividend_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[float | None] = mapped_column(Float, nullable=True)
    ex_dividend_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    payout_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    five_year_avg_dividend_yield: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    trailing_annual_dividend_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    trailing_annual_dividend_yield: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    last_dividend_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Analyst & target
    target_high_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_low_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_mean_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_median_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    number_of_analyst_opinions: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    recommendation_key: Mapped[str | None] = mapped_column(String(50), nullable=True)
    recommendation_mean: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Employees
    full_time_employees: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PriceHistory(BaseModel):
    """Daily OHLCV data from yf.Ticker.history()."""

    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "date", name="uq_price_history_instrument_date"
        ),
        Index("ix_price_history_instrument_id", "instrument_id"),
        Index("ix_price_history_date", "date"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="price_history")
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    open: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    high: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    low: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    close: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    dividends: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    stock_splits: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)


class FinancialStatement(BaseModel):
    """Normalized EAV table for income statement, balance sheet, cashflow, earnings."""

    __tablename__ = "financial_statements"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "statement_type",
            "period_type",
            "period_date",
            "line_item",
            name="uq_financial_statement_row",
        ),
        Index("ix_financial_statements_instrument_id", "instrument_id"),
        Index("ix_financial_statements_type_period", "statement_type", "period_type"),
        Index("ix_financial_statements_period_date", "period_date"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="financial_statements")
    statement_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # income_statement | balance_sheet | cashflow | earnings
    period_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )  # annual | quarterly
    period_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    line_item: Mapped[str] = mapped_column(String(200), nullable=False)
    value: Mapped[float | None] = mapped_column(Numeric(38, 6), nullable=True)


class Dividend(BaseModel):
    """Dividend payments from yf.Ticker.dividends."""

    __tablename__ = "dividends"
    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_dividend_instrument_date"),
        Index("ix_dividends_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="dividends")
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)


class StockSplit(BaseModel):
    """Stock split events from yf.Ticker.splits."""

    __tablename__ = "stock_splits"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "date", name="uq_stock_split_instrument_date"
        ),
        Index("ix_stock_splits_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="stock_splits")
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    ratio: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)


class AnalystRecommendation(BaseModel):
    """Analyst recommendation summary from yf.Ticker.recommendations_summary."""

    __tablename__ = "analyst_recommendations"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "period", name="uq_analyst_rec_instrument_period"
        ),
        Index("ix_analyst_recommendations_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(
        back_populates="analyst_recommendations"
    )
    period: Mapped[str] = mapped_column(String(50), nullable=False)
    strong_buy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    buy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hold: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sell: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strong_sell: Mapped[int | None] = mapped_column(Integer, nullable=True)


class AnalystPriceTarget(BaseModel):
    """Analyst price targets from yf.Ticker.analyst_price_targets."""

    __tablename__ = "analyst_price_targets"
    __table_args__ = (
        UniqueConstraint("instrument_id", name="uq_analyst_pt_instrument"),
        Index("ix_analyst_price_targets_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(
        back_populates="analyst_price_targets"
    )
    current: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    low: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    high: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    mean: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    median: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)


class InstitutionalHolder(BaseModel):
    """Institutional holders from yf.Ticker.institutional_holders."""

    __tablename__ = "institutional_holders"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "holder_name", name="uq_inst_holder_instrument_name"
        ),
        Index("ix_institutional_holders_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(
        back_populates="institutional_holders"
    )
    holder_name: Mapped[str] = mapped_column(String(500), nullable=False)
    date_reported: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    shares: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    value: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    pct_held: Mapped[float | None] = mapped_column(Float, nullable=True)


class MutualFundHolder(BaseModel):
    """Mutual fund holders from yf.Ticker.mutualfund_holders."""

    __tablename__ = "mutual_fund_holders"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "holder_name", name="uq_mutual_fund_holder_instrument_name"
        ),
        Index("ix_mutual_fund_holders_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="mutual_fund_holders")
    holder_name: Mapped[str] = mapped_column(String(500), nullable=False)
    date_reported: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    shares: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    value: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    pct_held: Mapped[float | None] = mapped_column(Float, nullable=True)


class InsiderTransaction(BaseModel):
    """Insider transactions from yf.Ticker.insider_transactions."""

    __tablename__ = "insider_transactions"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "insider_name",
            "start_date",
            "transaction_type",
            name="uq_insider_tx_row",
        ),
        Index("ix_insider_transactions_instrument_id", "instrument_id"),
        Index("ix_insider_transactions_start_date", "start_date"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="insider_transactions")
    insider_name: Mapped[str] = mapped_column(String(500), nullable=False)
    position: Mapped[str | None] = mapped_column(String(500), nullable=True)
    transaction_type: Mapped[str] = mapped_column(String(200), nullable=False)
    shares: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    value: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    ownership: Mapped[str | None] = mapped_column(String(50), nullable=True)


class TickerNews(BaseModel):
    """News articles from yf.Ticker.news."""

    __tablename__ = "ticker_news"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "news_uuid", name="uq_ticker_news_instrument_uuid"
        ),
        Index("ix_ticker_news_instrument_id", "instrument_id"),
        Index("ix_ticker_news_publish_time", "publish_time"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped[Instrument] = relationship(back_populates="news")
    news_uuid: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    publisher: Mapped[str | None] = mapped_column(String(500), nullable=True)
    link: Mapped[str | None] = mapped_column(Text, nullable=True)
    publish_time: Mapped[datetime.datetime | None] = mapped_column(nullable=True)
    news_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    related_tickers: Mapped[str | None] = mapped_column(Text, nullable=True)
