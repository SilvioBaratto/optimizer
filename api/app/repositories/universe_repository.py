"""Repository for universe data access (exchanges and instruments)."""

import logging
import uuid as uuid_mod
from typing import Dict, Any, List, Tuple, Optional, Sequence

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert

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
            },
        )
        self.session.execute(stmt)
        self.session.flush()
        return len(rows)

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
