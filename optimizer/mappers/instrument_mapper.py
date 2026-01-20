"""
Instrument Mapper - Convert between Instrument models and DTOs.
"""

from typing import List, Dict

from optimizer.database.models.universe import Instrument, Exchange
from optimizer.domain.models.instrument import InstrumentDTO, ExchangeDTO


class InstrumentMapper:
    """
    Static mapper class for Instrument conversions.

    Follows the Data Mapper pattern to separate domain logic from persistence.
    """

    @staticmethod
    def to_dto(instrument: Instrument) -> InstrumentDTO:
        """
        Convert SQLAlchemy Instrument model to InstrumentDTO.

        Args:
            instrument: Database Instrument model

        Returns:
            InstrumentDTO domain object
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

    @staticmethod
    def to_dto_batch(instruments: List[Instrument]) -> List[InstrumentDTO]:
        """
        Convert multiple instruments to DTOs.

        Args:
            instruments: List of database Instrument models

        Returns:
            List of InstrumentDTO objects
        """
        return [InstrumentMapper.to_dto(i) for i in instruments]

    @staticmethod
    def to_dto_dict(instruments: List[Instrument]) -> Dict[str, InstrumentDTO]:
        """
        Convert instruments to a ticker-keyed dictionary.

        Args:
            instruments: List of database Instrument models

        Returns:
            Dictionary mapping ticker -> InstrumentDTO
        """
        return {i.ticker: InstrumentMapper.to_dto(i) for i in instruments}


class ExchangeMapper:
    """
    Static mapper class for Exchange conversions.
    """

    @staticmethod
    def to_dto(exchange: Exchange) -> ExchangeDTO:
        """
        Convert SQLAlchemy Exchange model to ExchangeDTO.

        Args:
            exchange: Database Exchange model

        Returns:
            ExchangeDTO domain object
        """
        return ExchangeDTO(
            id=exchange.id,
            exchange_id=exchange.exchange_id,
            exchange_name=exchange.exchange_name,
            is_active=exchange.is_active,
            last_updated=exchange.last_updated,
        )

    @staticmethod
    def to_dto_batch(exchanges: List[Exchange]) -> List[ExchangeDTO]:
        """
        Convert multiple exchanges to DTOs.

        Args:
            exchanges: List of database Exchange models

        Returns:
            List of ExchangeDTO objects
        """
        return [ExchangeMapper.to_dto(e) for e in exchanges]
