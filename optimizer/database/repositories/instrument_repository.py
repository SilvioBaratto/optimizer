import logging
from typing import List, Dict, Optional

from sqlalchemy import select

from optimizer.database.database import DatabaseManager
from optimizer.database.models.universe import Instrument, Exchange
from optimizer.database.repositories.base import BaseRepository
from optimizer.domain.models.instrument import InstrumentDTO, ExchangeDTO

logger = logging.getLogger(__name__)


class InstrumentRepositoryImpl(BaseRepository[Instrument]):
    """
    SQLAlchemy implementation of the InstrumentRepository protocol.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize instrument repository.
        """
        super().__init__(db_manager, Instrument)

    def _to_dto(self, instrument: Instrument) -> InstrumentDTO:
        """
        Convert SQLAlchemy model to domain DTO.
        """
        exchange_name = None
        if instrument.exchange:
            exchange_name = instrument.exchange.exchange_name

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
            exchange_name=exchange_name,
        )

    def get_by_ticker(self, ticker: str) -> Optional[InstrumentDTO]:
        """
        Fetch a single instrument by Trading212 ticker.
        """
        with self._get_session() as session:
            query = select(Instrument).where(Instrument.ticker == ticker)
            result = session.execute(query).scalar_one_or_none()

            if result:
                return self._to_dto(result)
            return None

    def get_by_tickers(self, tickers: List[str]) -> Dict[str, InstrumentDTO]:
        """
        Fetch multiple instruments by tickers.
        """
        with self._get_session() as session:
            query = select(Instrument).where(Instrument.ticker.in_(tickers))
            results = session.execute(query).scalars().all()

            return {inst.ticker: self._to_dto(inst) for inst in results}

    def get_yfinance_mapping(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """
        Get mapping from Trading212 tickers to Yahoo Finance tickers.
        """
        with self._get_session() as session:
            query = select(Instrument.ticker, Instrument.yfinance_ticker).where(
                Instrument.ticker.in_(tickers)
            )

            results = session.execute(query).all()
            return {t212: yf for t212, yf in results}

    def get_by_exchange(self, exchange_name: str) -> List[InstrumentDTO]:
        """
        Fetch all instruments from a specific exchange.
        """
        with self._get_session() as session:
            query = (
                select(Instrument)
                .join(Exchange, Instrument.exchange_id == Exchange.id)
                .where(Exchange.exchange_name == exchange_name)
            )

            results = session.execute(query).scalars().all()
            return [self._to_dto(inst) for inst in results]

    def get_active_instruments(self) -> List[InstrumentDTO]:
        """
        Fetch all active (tradeable) instruments.
        """
        with self._get_session() as session:
            query = select(Instrument).where(Instrument.is_active == True)
            results = session.execute(query).scalars().all()
            return [self._to_dto(inst) for inst in results]

    def get_by_yfinance_ticker(self, yf_ticker: str) -> Optional[InstrumentDTO]:
        """
        Fetch instrument by Yahoo Finance ticker.
        """
        with self._get_session() as session:
            query = select(Instrument).where(Instrument.yfinance_ticker == yf_ticker)
            result = session.execute(query).scalar_one_or_none()

            if result:
                return self._to_dto(result)
            return None

    def get_instruments_by_currency(self, currency_code: str) -> List[InstrumentDTO]:
        """
        Fetch instruments by currency.
        """
        with self._get_session() as session:
            query = select(Instrument).where(Instrument.currency_code == currency_code)
            results = session.execute(query).scalars().all()
            return [self._to_dto(inst) for inst in results]

    def count_by_exchange(self) -> Dict[str, int]:
        """
        Count instruments by exchange.
        """
        from sqlalchemy import func

        with self._get_session() as session:
            query = (
                select(Exchange.exchange_name, func.count(Instrument.id))
                .join(Exchange, Instrument.exchange_id == Exchange.id)
                .group_by(Exchange.exchange_name)
            )

            results = session.execute(query).all()
            return {name: count for name, count in results}


class ExchangeRepositoryImpl(BaseRepository[Exchange]):
    """Repository for exchange data."""

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Exchange)

    def _to_dto(self, exchange: Exchange) -> ExchangeDTO:
        """Convert to DTO."""
        return ExchangeDTO(
            id=exchange.id,
            exchange_id=exchange.exchange_id,
            exchange_name=exchange.exchange_name,
            is_active=exchange.is_active,
            last_updated=exchange.last_updated,
        )

    def get_by_name(self, name: str) -> Optional[ExchangeDTO]:
        """Get exchange by name."""
        with self._get_session() as session:
            query = select(Exchange).where(Exchange.exchange_name == name)
            result = session.execute(query).scalar_one_or_none()
            return self._to_dto(result) if result else None

    def get_all_active(self) -> List[ExchangeDTO]:
        """Get all active exchanges."""
        with self._get_session() as session:
            query = select(Exchange).where(Exchange.is_active == True)
            results = session.execute(query).scalars().all()
            return [self._to_dto(e) for e in results]
