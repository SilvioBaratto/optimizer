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
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from portopt_db.base import BaseModel


class FredObservation(BaseModel):
    """FRED time-series observation: one row per (series_id, date).

    Stores daily spread/yield observations from the St. Louis FRED
    REST API.  Unique on ``(series_id, date)`` so repeated fetches
    safely upsert without duplication.
    """

    __tablename__ = "fred_observations"
    __table_args__ = (
        UniqueConstraint("series_id", "date", name="uq_fred_observation_series_date"),
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
        DateTime(timezone=True),
        nullable=True,
    )
    source_ticker: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_query: Mapped[str | None] = mapped_column(String(200), nullable=True)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_content: Mapped[str | None] = mapped_column(Text, nullable=True)

    theme_entries: Mapped[list[MacroNewsTheme]] = relationship(
        back_populates="news",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @property
    def themes(self) -> str | None:
        """Return comma-separated sorted theme names, or None when no themes are linked."""
        if not self.theme_entries:
            return None
        return ",".join(sorted(e.theme for e in self.theme_entries))


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
