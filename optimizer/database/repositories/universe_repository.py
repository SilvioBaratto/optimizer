import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from optimizer.database.database import DatabaseManager
from optimizer.database.models.universe import Exchange, Instrument
from optimizer.database.repositories.base import BaseRepository
from optimizer.domain.models.instrument import ExchangeDTO, InstrumentDTO

logger = logging.getLogger(__name__)


class UniverseRepositoryImpl(BaseRepository[Instrument]):
    """
    Repository implementation for universe data access.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize repository.
        """
        if db_manager is None:
            from optimizer.database.database import database_manager

            db_manager = database_manager

        super().__init__(db_manager, Instrument)

    def save_exchange(self, exchange_data: Dict[str, Any]) -> ExchangeDTO:
        """
        Save or update an exchange.
        """
        exchange_id = exchange_data["id"]
        exchange_name = exchange_data["name"]

        with self._get_session() as session:
            # Check if exchange already exists
            existing = session.execute(
                select(Exchange).where(Exchange.exchange_id == exchange_id)
            ).scalar_one_or_none()

            if existing:
                # Update existing exchange
                existing.exchange_name = exchange_name
                existing.is_active = True
                existing.last_updated = datetime.now()
                session.commit()
                return self._exchange_to_dto(existing)
            else:
                # Create new exchange
                new_exchange = Exchange(
                    exchange_id=exchange_id,
                    exchange_name=exchange_name,
                    is_active=True,
                    last_updated=datetime.now(),
                )
                session.add(new_exchange)
                session.commit()
                session.refresh(new_exchange)
                return self._exchange_to_dto(new_exchange)

    def get_exchange_by_t212_id(self, t212_exchange_id: int) -> Optional[ExchangeDTO]:
        """
        Get exchange by Trading212 ID.
        """
        with self._get_session() as session:
            exchange = session.execute(
                select(Exchange).where(Exchange.exchange_id == t212_exchange_id)
            ).scalar_one_or_none()

            if exchange:
                return self._exchange_to_dto(exchange)
            return None

    def save_instrument(
        self, instrument_data: Dict[str, Any], exchange_id: UUID
    ) -> Optional[InstrumentDTO]:
        """
        Save or update an instrument.
        """
        ticker = instrument_data.get("ticker")
        if not ticker:
            return None

        with self._get_session() as session:
            # Check if instrument exists
            existing = session.execute(
                select(Instrument).where(Instrument.ticker == ticker)
            ).scalar_one_or_none()

            # Parse addedOn date
            added_on = self._parse_added_on(instrument_data.get("addedOn"))

            if existing:
                # Update existing instrument
                existing.exchange_id = exchange_id
                existing.short_name = instrument_data.get("shortName") or ticker
                existing.name = instrument_data.get("name")
                existing.isin = instrument_data.get("isin")
                existing.instrument_type = instrument_data.get("type", "STOCK")
                existing.currency_code = instrument_data.get("currencyCode")
                existing.max_open_quantity = instrument_data.get("maxOpenQuantity")
                existing.added_on = added_on
                existing.yfinance_ticker = instrument_data.get("yfinanceTicker")
                existing.is_active = True
                existing.last_validated = datetime.now()
                session.commit()
                return self._instrument_to_dto(existing)
            else:
                # Create new instrument
                new_inst = Instrument(
                    exchange_id=exchange_id,
                    ticker=ticker,
                    short_name=instrument_data.get("shortName") or ticker,
                    name=instrument_data.get("name"),
                    isin=instrument_data.get("isin"),
                    instrument_type=instrument_data.get("type", "STOCK"),
                    currency_code=instrument_data.get("currencyCode"),
                    max_open_quantity=instrument_data.get("maxOpenQuantity"),
                    added_on=added_on,
                    yfinance_ticker=instrument_data.get("yfinanceTicker"),
                    is_active=True,
                    last_validated=datetime.now(),
                )
                session.add(new_inst)
                session.commit()
                session.refresh(new_inst)
                return self._instrument_to_dto(new_inst)

    def save_instruments_batch(
        self, instruments_data: List[Dict[str, Any]], exchange_id: UUID
    ) -> int:
        """
        Save multiple instruments in a batch.
        """
        if not instruments_data:
            return 0

        saved_count = 0

        with self._get_session() as session:
            # Get existing tickers for update detection
            tickers = [d.get("ticker") for d in instruments_data if d.get("ticker")]
            existing_query = select(Instrument).where(Instrument.ticker.in_(tickers))
            existing = {i.ticker: i for i in session.execute(existing_query).scalars()}

            new_instruments = []

            for inst_data in instruments_data:
                ticker = inst_data.get("ticker")
                if not ticker:
                    continue

                added_on = self._parse_added_on(inst_data.get("addedOn"))

                if ticker in existing:
                    # Update existing
                    inst = existing[ticker]
                    inst.exchange_id = exchange_id
                    inst.short_name = inst_data.get("shortName") or ticker
                    inst.name = inst_data.get("name")
                    inst.isin = inst_data.get("isin")
                    inst.instrument_type = inst_data.get("type", "STOCK")
                    inst.currency_code = inst_data.get("currencyCode")
                    inst.max_open_quantity = inst_data.get("maxOpenQuantity")
                    inst.added_on = added_on
                    inst.yfinance_ticker = inst_data.get("yfinanceTicker")
                    inst.is_active = True
                    inst.last_validated = datetime.now()
                    saved_count += 1
                else:
                    # Create new
                    new_inst = Instrument(
                        exchange_id=exchange_id,
                        ticker=ticker,
                        short_name=inst_data.get("shortName") or ticker,
                        name=inst_data.get("name"),
                        isin=inst_data.get("isin"),
                        instrument_type=inst_data.get("type", "STOCK"),
                        currency_code=inst_data.get("currencyCode"),
                        max_open_quantity=inst_data.get("maxOpenQuantity"),
                        added_on=added_on,
                        yfinance_ticker=inst_data.get("yfinanceTicker"),
                        is_active=True,
                        last_validated=datetime.now(),
                    )
                    new_instruments.append(new_inst)

            if new_instruments:
                session.bulk_save_objects(new_instruments)
                saved_count += len(new_instruments)

            session.commit()

        return saved_count

    def clear_all(self) -> Tuple[int, int]:
        """
        Clear all universe data (exchanges and instruments).
        """
        with self._get_session() as session:
            deleted_instruments = session.execute(delete(Instrument)).rowcount
            deleted_exchanges = session.execute(delete(Exchange)).rowcount
            session.commit()
            logger.info(
                f"Cleared {deleted_exchanges} exchanges and {deleted_instruments} instruments"
            )
            return deleted_exchanges, deleted_instruments

    def get_instrument_count(self) -> int:
        """Get total number of instruments."""
        return self._count()

    def get_exchange_count(self) -> int:
        """Get total number of exchanges."""
        with self._get_session() as session:
            return session.query(Exchange).count()

    def _exchange_to_dto(self, exchange: Exchange) -> ExchangeDTO:
        """Convert Exchange ORM model to ExchangeDTO."""
        return ExchangeDTO(
            id=exchange.id,
            exchange_id=exchange.exchange_id,
            exchange_name=exchange.exchange_name,
            is_active=exchange.is_active,
            last_updated=exchange.last_updated,
        )

    def _instrument_to_dto(self, instrument: Instrument) -> InstrumentDTO:
        """Convert Instrument ORM model to InstrumentDTO."""
        return InstrumentDTO(
            id=instrument.id,
            exchange_id=instrument.exchange_id,
            ticker=instrument.ticker,
            short_name=instrument.short_name,
            name=instrument.name,
            isin=instrument.isin,
            instrument_type=instrument.instrument_type,
            currency_code=instrument.currency_code,
            yfinance_ticker=instrument.yfinance_ticker,
            is_active=instrument.is_active,
            max_open_quantity=instrument.max_open_quantity,
            added_on=instrument.added_on,
            last_validated=instrument.last_validated,
            exchange_name=instrument.exchange.exchange_name if instrument.exchange else None,
        )

    def _parse_added_on(self, added_on_str: Optional[str]) -> Optional[datetime]:
        """Parse addedOn date string from T212 API."""
        if not added_on_str:
            return None
        try:
            return datetime.fromisoformat(added_on_str.replace("+", " +"))
        except (ValueError, AttributeError):
            return None
