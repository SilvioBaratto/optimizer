"""Repository for view generation database queries."""

import logging
from collections.abc import Sequence
from datetime import date, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.universe import Instrument
from app.models.yfinance_data import PriceHistory, TickerProfile

logger = logging.getLogger(__name__)


class ViewGenerationRepository:
    """Sync repository for instrument, price, and profile lookups."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_instrument_by_ticker(self, ticker: str) -> Instrument | None:
        """Return the :class:`Instrument` for *ticker*, or ``None``."""
        return self.session.execute(
            select(Instrument).where(Instrument.ticker == ticker)
        ).scalar_one_or_none()

    def get_close_prices(
        self,
        instrument_id: UUID,
        lookback_days: int = 260,
    ) -> list[float]:
        """Return close prices in ascending date order for the last *lookback_days*."""
        cutoff = date.today() - timedelta(days=lookback_days)
        rows: Sequence[float | None] = (
            self.session.execute(
                select(PriceHistory.close)
                .where(
                    PriceHistory.instrument_id == instrument_id,
                    PriceHistory.date >= cutoff,
                    PriceHistory.close.isnot(None),
                )
                .order_by(PriceHistory.date.asc())
            )
            .scalars()
            .all()
        )
        return [float(c) for c in rows if c is not None]

    def get_ticker_profile(self, instrument_id: UUID) -> TickerProfile | None:
        """Return the :class:`TickerProfile` for *instrument_id*, or ``None``."""
        return self.session.execute(
            select(TickerProfile).where(TickerProfile.instrument_id == instrument_id)
        ).scalar_one_or_none()
