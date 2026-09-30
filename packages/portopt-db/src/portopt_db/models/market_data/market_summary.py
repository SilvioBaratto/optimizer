"""SQLAlchemy model for market summaries (SPEC B3).

Regional index/quote summaries from ``yf.Market(id).summary``. Market-wide (no
Instrument FK); one row per (market identifier, symbol, snapshot date).
"""

from __future__ import annotations

import datetime

from sqlalchemy import Date, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel


class MarketSummary(BaseModel):
    """Row in ``market_summaries`` for one index/quote in a regional market.

    Attributes:
        market: yfinance market identifier (e.g. ``"us_market"``, ``"gb_market"``).
        symbol: ticker within that market (e.g. ``"^GSPC"``).
        as_of: date the snapshot was captured.
        market_state: exchange session state as reported by yfinance
            (e.g. ``OPEN``, ``CLOSED``, ``PRE``, ``POST``); ``None`` when absent.
        change: absolute price change from the previous close; ``None`` when unavailable.
        change_percent: percentage change from the previous close; ``None`` when unavailable.
    """

    __tablename__ = "market_summaries"
    __table_args__ = (
        UniqueConstraint("market", "symbol", "as_of", name="uq_market_summary"),
        Index("ix_market_summaries_market", "market"),
    )

    market: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(40), nullable=False)
    as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    price: Mapped[float | None] = mapped_column(Numeric(24, 6), nullable=True)
    change: Mapped[float | None] = mapped_column(Numeric(24, 6), nullable=True)
    change_percent: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    previous_close: Mapped[float | None] = mapped_column(Numeric(24, 6), nullable=True)
    market_state: Mapped[str | None] = mapped_column(String(20), nullable=True)


__all__ = ["MarketSummary"]
