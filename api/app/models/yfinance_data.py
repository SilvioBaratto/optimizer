"""SQLAlchemy models for yfinance data storage."""

from __future__ import annotations

import uuid
import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    BigInteger,
    Boolean,
    Index,
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
    instrument: Mapped["Instrument"] = relationship(back_populates="profiles")

    # Identifiers
    symbol: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    short_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    long_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    isin: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    exchange: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    quote_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # Classification
    sector: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    long_business_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Market data
    market_cap: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    enterprise_value: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    shares_outstanding: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    float_shares: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    implied_shares_outstanding: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Price & volume
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    previous_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    open_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    day_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    day_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fifty_two_week_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fifty_two_week_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fifty_day_average: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    two_hundred_day_average: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    average_volume_10days: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    regular_market_volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    bid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bid_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ask_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    beta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Valuation ratios
    trailing_pe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    forward_pe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trailing_eps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    forward_eps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_to_sales_trailing_12months: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_to_book: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    enterprise_to_revenue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    enterprise_to_ebitda: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    book_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Profitability
    profit_margins: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_margins: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gross_margins: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ebitda_margins: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    return_on_assets: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    return_on_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Income & revenue
    total_revenue: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    revenue_per_share: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revenue_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    earnings_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    earnings_quarterly_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ebitda: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    gross_profits: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    free_cashflow: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    operating_cashflow: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    total_cash: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    total_cash_per_share: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_debt: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    debt_to_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quick_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Dividends
    dividend_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ex_dividend_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    payout_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    five_year_avg_dividend_yield: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trailing_annual_dividend_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trailing_annual_dividend_yield: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_dividend_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Analyst & target
    target_high_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_low_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_mean_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_median_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    number_of_analyst_opinions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    recommendation_key: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    recommendation_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Employees
    full_time_employees: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)



class PriceHistory(BaseModel):
    """Daily OHLCV data from yf.Ticker.history()."""

    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_price_history_instrument_date"),
        Index("ix_price_history_instrument_id", "instrument_id"),
        Index("ix_price_history_date", "date"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped["Instrument"] = relationship(back_populates="price_history")
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    open: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    close: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    dividends: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    stock_splits: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)


class FinancialStatement(BaseModel):
    """Normalized EAV table for income statement, balance sheet, cashflow, earnings."""

    __tablename__ = "financial_statements"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "statement_type", "period_type", "period_date", "line_item",
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
    instrument: Mapped["Instrument"] = relationship(back_populates="financial_statements")
    statement_type: Mapped[str] = mapped_column(
        String(50), nullable=False,
    )  # income_statement | balance_sheet | cashflow | earnings
    period_type: Mapped[str] = mapped_column(
        String(20), nullable=False,
    )  # annual | quarterly
    period_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    line_item: Mapped[str] = mapped_column(String(200), nullable=False)
    value: Mapped[Optional[float]] = mapped_column(Numeric(38, 6), nullable=True)


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
    instrument: Mapped["Instrument"] = relationship(back_populates="dividends")
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)


class StockSplit(BaseModel):
    """Stock split events from yf.Ticker.splits."""

    __tablename__ = "stock_splits"
    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_stock_split_instrument_date"),
        Index("ix_stock_splits_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped["Instrument"] = relationship(back_populates="stock_splits")
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    ratio: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)


class AnalystRecommendation(BaseModel):
    """Analyst recommendation summary from yf.Ticker.recommendations_summary."""

    __tablename__ = "analyst_recommendations"
    __table_args__ = (
        UniqueConstraint("instrument_id", "period", name="uq_analyst_rec_instrument_period"),
        Index("ix_analyst_recommendations_instrument_id", "instrument_id"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped["Instrument"] = relationship(back_populates="analyst_recommendations")
    period: Mapped[str] = mapped_column(String(50), nullable=False)
    strong_buy: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    buy: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hold: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sell: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    strong_sell: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


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
    instrument: Mapped["Instrument"] = relationship(back_populates="analyst_price_targets")
    current: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    mean: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)
    median: Mapped[Optional[float]] = mapped_column(Numeric(20, 6), nullable=True)


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
    instrument: Mapped["Instrument"] = relationship(back_populates="institutional_holders")
    holder_name: Mapped[str] = mapped_column(String(500), nullable=False)
    date_reported: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    shares: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    value: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    pct_held: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


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
    instrument: Mapped["Instrument"] = relationship(back_populates="mutual_fund_holders")
    holder_name: Mapped[str] = mapped_column(String(500), nullable=False)
    date_reported: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    shares: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    value: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    pct_held: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class InsiderTransaction(BaseModel):
    """Insider transactions from yf.Ticker.insider_transactions."""

    __tablename__ = "insider_transactions"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "insider_name", "start_date", "transaction_type",
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
    instrument: Mapped["Instrument"] = relationship(back_populates="insider_transactions")
    insider_name: Mapped[str] = mapped_column(String(500), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    transaction_type: Mapped[str] = mapped_column(String(200), nullable=False)
    shares: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    value: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    ownership: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class TickerNews(BaseModel):
    """News articles from yf.Ticker.news."""

    __tablename__ = "ticker_news"
    __table_args__ = (
        UniqueConstraint("instrument_id", "news_uuid", name="uq_ticker_news_instrument_uuid"),
        Index("ix_ticker_news_instrument_id", "instrument_id"),
        Index("ix_ticker_news_publish_time", "publish_time"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument: Mapped["Instrument"] = relationship(back_populates="news")
    news_uuid: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    publisher: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    publish_time: Mapped[Optional[datetime.datetime]] = mapped_column(nullable=True)
    news_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    related_tickers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
