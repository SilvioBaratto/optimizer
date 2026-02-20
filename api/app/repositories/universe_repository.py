"""Repository for universe data access (exchanges and instruments)."""

import logging
import uuid as uuid_mod
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, joinedload

from app.models.universe import Exchange, Instrument

logger = logging.getLogger(__name__)


class UniverseRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_exchange(self, exchange_data: Dict[str, Any]) -> Exchange:
        name = exchange_data.get("name", "")
        t212_id = exchange_data.get("id")

        stmt = select(Exchange).where(Exchange.name == name)
        existing = self.session.execute(stmt).scalar_one_or_none()

        if existing:
            existing.t212_id = t212_id
            self.session.flush()
            return existing

        exchange = Exchange(name=name, t212_id=t212_id)
        self.session.add(exchange)
        self.session.flush()
        return exchange

    def save_instruments_batch(
        self, instruments_data: List[Dict[str, Any]], exchange_id: Any
    ) -> int:
        if not instruments_data:
            return 0

        rows = []
        for data in instruments_data:
            rows.append(
                {
                    "id": uuid_mod.uuid4(),
                    "ticker": data.get("ticker", ""),
                    "short_name": data.get("shortName", ""),
                    "name": data.get("name"),
                    "isin": data.get("isin"),
                    "instrument_type": data.get("type"),
                    "currency_code": data.get("currencyCode"),
                    "yfinance_ticker": data.get("yfinanceTicker"),
                    "exchange_id": exchange_id,
                }
            )

        stmt = pg_insert(Instrument).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_instrument_ticker_exchange",
            set_={
                "short_name": stmt.excluded.short_name,
                "name": stmt.excluded.name,
                "isin": stmt.excluded.isin,
                "instrument_type": stmt.excluded.instrument_type,
                "currency_code": stmt.excluded.currency_code,
                "yfinance_ticker": stmt.excluded.yfinance_ticker,
                # Re-activating an instrument clears its delisting status.
                "delisted_at": None,
                "delisting_return": None,
            },
        )
        self.session.execute(stmt)
        self.session.flush()
        return len(rows)

    def mark_delisted(
        self,
        ticker: str,
        exchange_id: Any,
        delisted_at: date,
        delisting_return: float = -0.30,
    ) -> bool:
        """Mark an instrument as delisted.

        Parameters
        ----------
        ticker : str
            Trading 212 ticker of the instrument.
        exchange_id : UUID
            Exchange the instrument belongs to.
        delisted_at : date
            The date the instrument was last seen in the T212 universe.
        delisting_return : float, default=-0.30
            CRSP-style default delisting return.  Use the actual value
            when known (e.g. acquisition premium or -1.0 for bankruptcy).

        Returns
        -------
        bool
            ``True`` if the record was updated, ``False`` if not found.
        """
        result = self.session.execute(
            update(Instrument)
            .where(Instrument.ticker == ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.delisted_at.is_(None))  # only if not already marked
            .values(delisted_at=delisted_at, delisting_return=delisting_return)
        )
        self.session.flush()
        return bool(result.rowcount)

    def get_active_tickers(self, exchange_id: Any) -> Set[str]:
        """Return the set of non-delisted tickers for an exchange."""
        rows = self.session.execute(
            select(Instrument.ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.delisted_at.is_(None))
        ).all()
        return {r[0] for r in rows}

    def clear_all(self) -> Tuple[int, int]:
        inst_count = self.session.execute(
            select(func.count()).select_from(Instrument)
        ).scalar_one()
        ex_count = self.session.execute(
            select(func.count()).select_from(Exchange)
        ).scalar_one()

        self.session.execute(delete(Instrument))
        self.session.execute(delete(Exchange))
        self.session.flush()

        return ex_count, inst_count

    def get_instrument_count(self) -> int:
        return self.session.execute(
            select(func.count()).select_from(Instrument)
        ).scalar_one()

    def get_exchange_count(self) -> int:
        return self.session.execute(
            select(func.count()).select_from(Exchange)
        ).scalar_one()

    def get_instruments(
        self,
        exchange_name: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[Instrument]:
        stmt = select(Instrument).options(joinedload(Instrument.exchange))
        if exchange_name:
            stmt = stmt.join(Exchange).where(Exchange.name == exchange_name)
        stmt = stmt.offset(skip).limit(limit)
        return self.session.execute(stmt).scalars().unique().all()

    def get_exchanges(self) -> Sequence[Exchange]:
        return self.session.execute(
            select(Exchange).order_by(Exchange.name)
        ).scalars().all()
