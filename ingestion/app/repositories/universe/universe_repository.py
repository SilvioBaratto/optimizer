"""Repository for universe data access (exchanges and instruments)."""

import logging
import uuid as uuid_mod
from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, joinedload

from app.models.universe.universe import Exchange, Instrument
from app.repositories._shared import RepositoryBase

logger = logging.getLogger(__name__)


class UniverseRepository(RepositoryBase):
    def __init__(self, session: Session):
        super().__init__(session)

    def save_exchange(self, exchange_data: dict[str, Any]) -> Exchange:
        """Insert or update an exchange by name, returning the persisted row.

        T1.2 / ARCHITECTURE.md §5.4: written as an idempotent
        ``INSERT ... ON CONFLICT DO UPDATE`` on the unique ``exchanges.name``
        column (``index_elements=["name"]``, since the column carries a
        column-level ``unique=True`` with no named constraint) rather than a
        SELECT-then-INSERT, so an at-least-once re-run converges to one row
        without racing the unique index. Only ``t212_id`` (and ``updated_at``)
        are in the conflict update set, preserving the row's id.
        """
        name = exchange_data.get("name", "")
        t212_id = exchange_data.get("id")

        self._upsert(
            Exchange,
            [{"id": uuid_mod.uuid4(), "name": name, "t212_id": t212_id}],
            index_elements=["name"],
            update_columns=["t212_id", "updated_at"],
        )
        self.session.flush()

        # populate_existing refreshes an already-identity-mapped row so the
        # returned object reflects the just-upserted t212_id, not a stale value.
        return self.session.execute(
            select(Exchange)
            .where(Exchange.name == name)
            .execution_options(populate_existing=True)
        ).scalar_one()

    def save_instruments_batch(
        self, instruments_data: list[dict[str, Any]], exchange_id: Any
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
        result: CursorResult[Any] = self.session.execute(  # type: ignore[assignment]
            update(Instrument)
            .where(Instrument.ticker == ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.delisted_at.is_(None))  # only if not already marked
            .values(delisted_at=delisted_at, delisting_return=delisting_return)
        )
        self.session.flush()
        return bool(result.rowcount)

    def get_active_tickers(self, exchange_id: Any) -> set[str]:
        """Return the set of non-delisted tickers for an exchange."""
        rows = self.session.execute(
            select(Instrument.ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.delisted_at.is_(None))
        ).all()
        return {r[0] for r in rows}

    def clear_all(self) -> tuple[int, int]:
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
        exchange_name: str | None = None,
        search: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[Instrument]:
        stmt = select(Instrument).options(joinedload(Instrument.exchange))
        if exchange_name:
            stmt = stmt.join(Exchange).where(Exchange.name == exchange_name)
        if search:
            term = f"%{search.strip()}%"
            stmt = stmt.where(
                (Instrument.ticker.ilike(term)) | (Instrument.name.ilike(term))
            )
        stmt = stmt.offset(skip).limit(limit)
        return self.session.execute(stmt).scalars().unique().all()

    def get_exchanges(self) -> Sequence[Exchange]:
        return (
            self.session.execute(select(Exchange).order_by(Exchange.name))
            .scalars()
            .all()
        )
