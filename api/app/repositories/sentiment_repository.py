"""Repository for sentiment-related database queries."""

import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.macro_regime import MacroNews
from app.models.universe import Instrument
from app.models.yfinance_data import TickerNews, TickerProfile
from app.repositories.base import RepositoryBase

logger = logging.getLogger(__name__)

# Sector ETF → GICS sector name (as stored in ticker_profiles.sector by yfinance)
_ETF_TO_SECTOR: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financial Services",
    "XLE": "Energy",
    "XLP": "Consumer Defensive",
    "XLU": "Utilities",
    "XLB": "Basic Materials",
    "XLI": "Industrials",
    "XLV": "Healthcare",
    "XLY": "Consumer Cyclical",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Reverse: sector name → set of ETF source_tickers
_SECTOR_TO_ETFS: dict[str, set[str]] = {}
for _etf, _sector in _ETF_TO_SECTOR.items():
    _SECTOR_TO_ETFS.setdefault(_sector, set()).add(_etf)


class SentimentRepository(RepositoryBase):
    """Sync repository for instrument lookups and news retrieval."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def get_instrument_id_by_ticker(self, ticker: str) -> UUID | None:
        """Return the instrument UUID for *ticker*, or ``None``."""
        return self.session.execute(
            select(Instrument.id).where(Instrument.ticker == ticker)
        ).scalar_one_or_none()

    def get_recent_news(
        self,
        instrument_id: UUID,
        cutoff: datetime,
    ) -> Sequence[TickerNews]:
        """Return news rows published after *cutoff* in ascending order."""
        return (
            self.session.execute(
                select(TickerNews)
                .where(
                    TickerNews.instrument_id == instrument_id,
                    TickerNews.title.isnot(None),
                    TickerNews.publish_time >= cutoff,
                )
                .order_by(TickerNews.publish_time.asc())
            )
            .scalars()
            .all()
        )

    def get_macro_news_fallback(
        self,
        ticker: str,
        cutoff: datetime,
        limit: int = 30,
    ) -> Sequence[MacroNews]:
        """Search ``macro_news`` for articles relevant to *ticker*.

        Uses a cascading strategy:
          A) Direct ``source_ticker`` match
          B) Sector ETF match (look up ticker's sector → matching ETF feeds)
          C) Title/content text search for the ticker symbol

        Returns ``MacroNews`` rows which have the same ``.title`` and
        ``.publish_time`` attributes that ``fetch_news_sentiment`` reads.
        """
        base_filters = [
            MacroNews.title.isnot(None),
            MacroNews.publish_time >= cutoff,
        ]

        # Strategy A: direct source_ticker match
        rows: Sequence[MacroNews] = (
            self.session.execute(
                select(MacroNews)
                .where(MacroNews.source_ticker == ticker, *base_filters)
                .order_by(MacroNews.publish_time.asc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        if rows:
            return rows

        # Strategy B: sector ETF match
        sector = self.session.execute(
            select(TickerProfile.sector)
            .join(Instrument, TickerProfile.instrument_id == Instrument.id)
            .where(Instrument.ticker == ticker)
        ).scalar_one_or_none()

        if sector:
            etf_tickers = _SECTOR_TO_ETFS.get(sector, set())
            if etf_tickers:
                rows = (
                    self.session.execute(
                        select(MacroNews)
                        .where(
                            MacroNews.source_ticker.in_(etf_tickers),
                            *base_filters,
                        )
                        .order_by(MacroNews.publish_time.asc())
                        .limit(limit)
                    )
                    .scalars()
                    .all()
                )
                if rows:
                    return rows

        # Strategy C: text search in title or full_content
        pattern = f"%{ticker}%"
        rows = (
            self.session.execute(
                select(MacroNews)
                .where(
                    or_(
                        MacroNews.title.ilike(pattern),
                        MacroNews.full_content.ilike(pattern),
                    ),
                    *base_filters,
                )
                .order_by(MacroNews.publish_time.asc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return rows
