"""SQLAlchemy models for market-wide calendars (SPEC B2).

Earnings / IPO / splits / economic-event calendars from ``yf.Calendars``.
Market-wide (not per-ticker) — no Instrument FK; each row keyed on its natural
identity so an at-least-once refetch converges to one row.
"""

from __future__ import annotations

import datetime

from sqlalchemy import BigInteger, Date, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel


class EarningsCalendar(BaseModel):
    """Upcoming/most-recent earnings events (market-wide)."""

    __tablename__ = "earnings_calendar"
    __table_args__ = (
        UniqueConstraint("ticker", "event_date", name="uq_earnings_calendar"),
        Index("ix_earnings_calendar_event_date", "event_date"),
    )

    ticker: Mapped[str] = mapped_column(String(30), nullable=False)
    event_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    eps_estimate: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    eps_actual: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    eps_surprise_pct: Mapped[float | None] = mapped_column(
        Numeric(20, 6), nullable=True
    )


class IpoCalendar(BaseModel):
    """Upcoming/priced IPO events (market-wide)."""

    __tablename__ = "ipo_calendar"
    __table_args__ = (
        UniqueConstraint("ticker", "ipo_date", name="uq_ipo_calendar"),
        Index("ix_ipo_calendar_ipo_date", "ipo_date"),
    )

    ticker: Mapped[str] = mapped_column(String(30), nullable=False)
    ipo_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(50), nullable=True)
    price_range: Mapped[str | None] = mapped_column(String(100), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(10), nullable=True)
    shares: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class SplitCalendar(BaseModel):
    """Upcoming stock-split events (market-wide)."""

    __tablename__ = "split_calendar"
    __table_args__ = (
        UniqueConstraint("ticker", "split_date", name="uq_split_calendar"),
        Index("ix_split_calendar_split_date", "split_date"),
    )

    ticker: Mapped[str] = mapped_column(String(30), nullable=False)
    split_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ratio: Mapped[str | None] = mapped_column(String(50), nullable=True)


class EconomicEventCalendar(BaseModel):
    """Upcoming macro-economic events (market-wide)."""

    __tablename__ = "economic_event_calendar"
    __table_args__ = (
        UniqueConstraint(
            "event", "country", "event_date", name="uq_economic_event_calendar"
        ),
        Index("ix_economic_event_calendar_event_date", "event_date"),
    )

    event: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str] = mapped_column(String(50), nullable=False)
    event_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    actual: Mapped[str | None] = mapped_column(String(50), nullable=True)
    forecast: Mapped[str | None] = mapped_column(String(50), nullable=True)
    prior: Mapped[str | None] = mapped_column(String(50), nullable=True)


__all__ = [
    "EarningsCalendar",
    "EconomicEventCalendar",
    "IpoCalendar",
    "SplitCalendar",
]
