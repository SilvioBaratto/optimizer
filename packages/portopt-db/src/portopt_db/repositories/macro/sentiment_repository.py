"""Repository for sentiment-related database queries."""

import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import MacroNews
from portopt_db.models.market_data.yfinance_data import TickerNews, TickerProfile
from portopt_db.models.universe.universe import Instrument
from portopt_db.repository import RepositoryBase

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
        """Return news rows for an instrument published at or after a cutoff.

        Args:
            instrument_id: Primary key of the instrument to query.
            cutoff: Lower bound (inclusive) on ``publish_time``.

        Returns:
            Rows ordered by ``publish_time`` ascending; empty if none found.
        """
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

        Uses a cascading strategy, returning as soon as one yields results:
        direct ``source_ticker`` match, then sector-ETF feed match (resolves
        the ticker's GICS sector to corresponding ETF tickers), then a
        case-insensitive text search on title and full content.

        Args:
            ticker: Equity ticker symbol to find relevant macro news for.
            cutoff: Only rows with ``publish_time`` at or after this value.
            limit: Maximum rows fetched per strategy attempt, not a total cap.

        Returns:
            MacroNews rows ordered by ``publish_time`` ascending; empty if no
            strategy yields results.
        """
        base_filters = [
            MacroNews.title.isnot(None),
            MacroNews.publish_time >= cutoff,
        ]

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
