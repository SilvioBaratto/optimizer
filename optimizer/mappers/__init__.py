"""
Mappers - Convert between SQLAlchemy models and domain DTOs.

These mappers provide a clean separation between the database layer
and the domain layer, following the Data Mapper pattern.

Usage:
    from mappers import SignalMapper, InstrumentMapper

    # Convert SQLAlchemy model to DTO
    signal_dto = SignalMapper.to_dto(signal_model)

    # Convert DTO to SQLAlchemy model
    signal_model = SignalMapper.from_dto(signal_dto)
"""

from optimizer.mappers.signal_mapper import SignalMapper
from optimizer.mappers.instrument_mapper import InstrumentMapper
from optimizer.mappers.portfolio_mapper import PortfolioMapper

__all__ = [
    "SignalMapper",
    "InstrumentMapper",
    "PortfolioMapper",
]
