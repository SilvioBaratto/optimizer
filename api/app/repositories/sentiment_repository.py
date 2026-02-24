"""Repository for sentiment-related database queries."""

import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.universe import Instrument
from app.models.yfinance_data import TickerNews

logger = logging.getLogger(__name__)


class SentimentRepository:
    """Sync repository for instrument lookups and news retrieval."""

    def __init__(self, session: Session) -> None:
        self.session = session

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
