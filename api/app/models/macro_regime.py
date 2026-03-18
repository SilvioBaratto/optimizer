"""SQLAlchemy models for macroeconomic regime data storage."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
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


class EconomicIndicator(BaseModel):
    """
    IlSole24Ore consensus forecasts per country snapshot.

    One row per country, storing forecast data from IlSole24Ore.
    Real macroeconomic indicators are sourced from TradingEconomics instead.
    """

    __tablename__ = "economic_indicators"
    __table_args__ = (
        UniqueConstraint("country", name="uq_economic_indicator_country"),
        Index("ix_economic_indicators_country", "country"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)

    # Forecast columns (from get_forecasts)
    last_inflation: Mapped[float | None] = mapped_column(Float, nullable=True)
    inflation_6m: Mapped[float | None] = mapped_column(Float, nullable=True)
    inflation_10y_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    gdp_growth_6m: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_12m: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps_expected_12m: Mapped[float | None] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    lt_rate_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Reference date (e.g. first day of the reference month)
    reference_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)


class EconomicIndicatorObservation(BaseModel):
    """IlSole24Ore forecast time-series: one row per (country, date).

    Wide-format snapshot preserving all 8 forecast columns per country
    per day, so that daily fetches accumulate history instead of
    overwriting the single ``economic_indicators`` row.
    """

    __tablename__ = "economic_indicator_observations"
    __table_args__ = (
        UniqueConstraint(
            "country", "date",
            name="uq_econ_obs_country_date",
        ),
        Index("ix_econ_observations_country", "country"),
        Index("ix_econ_observations_date", "date"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    last_inflation: Mapped[float | None] = mapped_column(Float, nullable=True)
    inflation_6m: Mapped[float | None] = mapped_column(Float, nullable=True)
    inflation_10y_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    gdp_growth_6m: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_12m: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps_expected_12m: Mapped[float | None] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    lt_rate_forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
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


class TradingEconomicsObservation(BaseModel):
    """Trading Economics time-series observation: one row per (country, indicator_key, date).

    Accumulates daily snapshots so that regime classifiers can access
    historical indicator values instead of only the latest overwritten row.
    """

    __tablename__ = "trading_economics_observations"
    __table_args__ = (
        UniqueConstraint(
            "country", "indicator_key", "date",
            name="uq_te_obs_country_key_date",
        ),
        Index("ix_te_observations_country", "country"),
        Index("ix_te_observations_date", "date"),
        Index("ix_te_obs_country_key_date", "country", "indicator_key", "date"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    indicator_key: Mapped[str] = mapped_column(String(100), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)


class BondYieldObservation(BaseModel):
    """Bond yield time-series observation: one row per (country, maturity, date).

    Accumulates daily yield snapshots so the yield curve history is
    preserved across daily fetches.
    """

    __tablename__ = "bond_yield_observations"
    __table_args__ = (
        UniqueConstraint(
            "country", "maturity", "date",
            name="uq_bond_obs_country_mat_date",
        ),
        Index("ix_bond_observations_country", "country"),
        Index("ix_bond_observations_date", "date"),
        Index("ix_bond_obs_country_maturity_date", "country", "maturity", "date"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    maturity: Mapped[str] = mapped_column(String(10), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    yield_value: Mapped[float | None] = mapped_column(Float, nullable=True)


class FredObservation(BaseModel):
    """FRED time-series observation: one row per (series_id, date).

    Stores daily spread/yield observations from the St. Louis FRED
    REST API.  Unique on ``(series_id, date)`` so repeated fetches
    safely upsert without duplication.
    """

    __tablename__ = "fred_observations"
    __table_args__ = (
        UniqueConstraint(
            "series_id", "date", name="uq_fred_observation_series_date"
        ),
        Index("ix_fred_observations_series_id", "series_id"),
        Index("ix_fred_observations_date", "date"),
    )

    series_id: Mapped[str] = mapped_column(String(50), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)


class MacroNews(BaseModel):
    """Macro-themed news articles from yfinance ticker feeds and search queries.

    One row per unique article (deduplicated by ``news_id``).
    """

    __tablename__ = "macro_news"
    __table_args__ = (
        UniqueConstraint("news_id", name="uq_macro_news_id"),
        Index("ix_macro_news_publish_time", "publish_time"),
    )

    news_id: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    publisher: Mapped[str | None] = mapped_column(String(500), nullable=True)
    link: Mapped[str | None] = mapped_column(Text, nullable=True)
    publish_time: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    source_ticker: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_query: Mapped[str | None] = mapped_column(String(200), nullable=True)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_content: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    theme_entries: Mapped[list[MacroNewsTheme]] = relationship(
        back_populates="news", cascade="all, delete-orphan", lazy="selectin",
    )

    @property
    def themes(self) -> str | None:
        if not self.theme_entries:
            return None
        return ",".join(sorted(e.theme for e in self.theme_entries))


class MacroCalibration(BaseModel):
    """Cached LLM macro regime calibration per country.

    One row per country, storing the most recent BAML ``ClassifyMacroRegime``
    result so the LLM is not invoked on every page load.  Re-generated when
    underlying macro data changes (via Refresh Data).
    """

    __tablename__ = "macro_calibrations"
    __table_args__ = (
        UniqueConstraint("country", name="uq_macro_calibration_country"),
        Index("ix_macro_calibrations_country", "country"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    phase: Mapped[str] = mapped_column(String(50), nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)
    tau: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    macro_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class MacroNewsSummary(BaseModel):
    """Daily country-level news summary aggregated from MacroNews articles.

    One row per (country, summary_date), storing AI-generated summaries,
    sentiment analysis, and key themes extracted from the day's news.
    """

    __tablename__ = "macro_news_summaries"
    __table_args__ = (
        UniqueConstraint(
            "country", "summary_date",
            name="uq_macro_news_summary_country_date",
        ),
        Index("ix_macro_news_summaries_country", "country"),
        Index("ix_macro_news_summaries_summary_date", "summary_date"),
    )

    country: Mapped[str] = mapped_column(String(100), nullable=False)
    summary_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(50), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    article_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    news_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class MacroNewsTheme(BaseModel):
    """Theme tag for a macro news article (junction table)."""

    __tablename__ = "macro_news_themes"
    __table_args__ = (
        UniqueConstraint("news_id", "theme", name="uq_macro_news_theme"),
        Index("ix_macro_news_themes_news_id", "news_id"),
        Index("ix_macro_news_themes_theme", "theme"),
    )

    news_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("macro_news.id", ondelete="CASCADE"),
        nullable=False,
    )
    theme: Mapped[str] = mapped_column(String(50), nullable=False)

    news: Mapped[MacroNews] = relationship(back_populates="theme_entries")
