"""SQLAlchemy models for macroeconomic regime data storage."""

import datetime

from sqlalchemy import (
    Date,
    Float,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class EconomicIndicator(BaseModel):
    """
    IlSole24Ore real indicators and forecasts per country snapshot.

    Each row represents either a real-indicator or forecast scrape
    for a given country, identified by the (country, source) pair.
    """

    __tablename__ = "economic_indicators"
    __table_args__ = (
        UniqueConstraint(
            "country", "source", name="uq_economic_indicator_country_source"
        ),
        Index("ix_economic_indicators_country", "country"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    source: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "ilsole_real" | "ilsole_forecast"

    # Real indicator columns (from get_real_indicators)
    gdp_growth_qq: Mapped[float | None] = mapped_column(Float, nullable=True)
    industrial_production: Mapped[float | None] = mapped_column(Float, nullable=True)
    unemployment: Mapped[float | None] = mapped_column(Float, nullable=True)
    consumer_prices: Mapped[float | None] = mapped_column(Float, nullable=True)
    deficit: Mapped[float | None] = mapped_column(Float, nullable=True)
    debt: Mapped[float | None] = mapped_column(Float, nullable=True)
    st_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    lt_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Forecast columns (from get_forecasts)
    last_inflation: Mapped[float | None] = mapped_column(Float, nullable=True)
    inflation_6m: Mapped[float | None] = mapped_column(Float, nullable=True)
    inflation_10y_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    gdp_growth_6m: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_12m: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps_expected_12m: Mapped[float | None] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    st_rate_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    lt_rate_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Reference date (e.g. first day of the reference month)
    reference_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)


class TradingEconomicsIndicator(BaseModel):
    """
    Trading Economics indicator row: one row per (country, indicator_key).

    Stores the latest value, previous value, unit, and reference date
    for each macro indicator scraped from Trading Economics.
    """

    __tablename__ = "trading_economics_indicators"
    __table_args__ = (
        UniqueConstraint(
            "country", "indicator_key", name="uq_te_indicator_country_key"
        ),
        Index("ix_trading_economics_indicators_country", "country"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    indicator_key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    reference: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_name: Mapped[str | None] = mapped_column(String(200), nullable=True)


class BondYield(BaseModel):
    """
    Government bond yield row: one row per (country, maturity).

    Stores yield value and period changes for key maturities (2Y, 5Y, 10Y, 30Y).
    """

    __tablename__ = "bond_yields"
    __table_args__ = (
        UniqueConstraint("country", "maturity", name="uq_bond_yield_country_maturity"),
        Index("ix_bond_yields_country", "country"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    maturity: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # "2Y", "5Y", "10Y", "30Y"
    yield_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    day_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    month_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    year_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
