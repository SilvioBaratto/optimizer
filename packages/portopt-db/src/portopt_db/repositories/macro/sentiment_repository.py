"""Repository for sentiment-related database queries."""

import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import ColumnElement, or_, select
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


# A symbol that moved venues has one instruments row per venue, the old one
# delisted. Ordering by this puts the active listing first and, among delisted
# rows only, the most recently delisted.
_ACTIVE_LISTING_FIRST = (
    Instrument.delisted_at.is_(None).desc(),
    Instrument.delisted_at.desc(),
    Instrument.id,
)


class SentimentRepository(RepositoryBase):
    """Sync repository for instrument lookups and news retrieval."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def get_instrument_id_by_ticker(self, ticker: str) -> UUID | None:
        """Return the instrument UUID for *ticker*, or ``None``.

        A symbol that moved venues has one row per venue, the old one delisted.
        The active listing wins; among delisted rows only, the most recently
        delisted does.
        """
        return (
            self.session.execute(
                select(Instrument.id)
                .where(Instrument.ticker == ticker)
                .order_by(*_ACTIVE_LISTING_FIRST)
                .limit(1)
            )
            .scalars()
            .first()
        )

    def get_recent_news(
        self,
        instrument_id: UUID,
        cutoff: datetime,
        as_of: datetime | None = None,
    ) -> Sequence[TickerNews]:
        """Return news rows for an instrument published inside a time window.

        Args:
            instrument_id: Primary key of the instrument to query.
            cutoff: Lower bound (inclusive) on ``publish_time``.
            as_of: Upper bound (inclusive) on ``publish_time``. ``None`` leaves
                the window open; point-in-time callers pass the decision date so
                later articles cannot leak in.

        Returns:
            Rows ordered by ``publish_time`` ascending; empty if none found.
        """
        conditions = [
            TickerNews.instrument_id == instrument_id,
            TickerNews.title.isnot(None),
            TickerNews.publish_time >= cutoff,
        ]
        if as_of is not None:
            conditions.append(TickerNews.publish_time <= as_of)
        return (
            self.session.execute(
                select(TickerNews)
                .where(*conditions)
                .order_by(TickerNews.publish_time.asc())
            )
            .scalars()
            .all()
        )

    def _newest_macro_news(
        self, conditions: Sequence[ColumnElement[bool]], limit: int
    ) -> Sequence[MacroNews]:
        """Return the *limit* newest matching rows, oldest first.

        Ordering descending before the LIMIT keeps the most recent articles of
        the window; the final reversal restores the ascending contract.
        """
        rows = (
            self.session.execute(
                select(MacroNews)
                .where(*conditions)
                .order_by(MacroNews.publish_time.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return list(reversed(rows))

    def get_macro_news_fallback(
        self,
        ticker: str,
        cutoff: datetime,
        limit: int = 30,
        as_of: datetime | None = None,
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
                When more rows match, the newest *limit* are kept.
            as_of: Upper bound (inclusive) on ``publish_time``. ``None`` leaves
                the window open; point-in-time callers pass the decision date so
                later articles cannot leak in.

        Returns:
            MacroNews rows ordered by ``publish_time`` ascending; empty if no
            strategy yields results.
        """
        base_filters = [
            MacroNews.title.isnot(None),
            MacroNews.publish_time >= cutoff,
        ]
        if as_of is not None:
            base_filters.append(MacroNews.publish_time <= as_of)

        rows = self._newest_macro_news(
            [MacroNews.source_ticker == ticker, *base_filters], limit
        )
        if rows:
            return rows

        sector = (
            self.session.execute(
                select(TickerProfile.sector)
                .join(Instrument, TickerProfile.instrument_id == Instrument.id)
                .where(Instrument.ticker == ticker)
                .order_by(*_ACTIVE_LISTING_FIRST)
                .limit(1)
            )
            .scalars()
            .first()
        )

        if sector:
            etf_tickers = _SECTOR_TO_ETFS.get(sector, set())
            if etf_tickers:
                rows = self._newest_macro_news(
                    [MacroNews.source_ticker.in_(etf_tickers), *base_filters], limit
                )
                if rows:
                    return rows

        pattern = f"%{ticker}%"
        return self._newest_macro_news(
            [
                or_(
                    MacroNews.title.ilike(pattern),
                    MacroNews.full_content.ilike(pattern),
                ),
                *base_filters,
            ],
            limit,
        )
