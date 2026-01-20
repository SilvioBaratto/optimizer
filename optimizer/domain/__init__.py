"""
Domain Layer - Pure Python domain models and protocols.

This layer contains:
- Protocols (interfaces) for dependency injection
- Domain models (DTOs) as dataclasses
- Value objects for immutable data types

No database or framework dependencies allowed in this layer.
"""

from optimizer.domain.protocols.repository import (
    SignalRepository,
    InstrumentRepository,
    MacroRegimeRepository,
    PortfolioRepository,
)
from optimizer.domain.protocols.filter import StockFilter, FilterPipeline
from optimizer.domain.protocols.optimizer import PortfolioOptimizer, CovarianceEstimator

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.domain.models.instrument import InstrumentDTO
from optimizer.domain.models.portfolio import PortfolioDTO, PositionDTO
from optimizer.domain.models.view import BlackLittermanViewDTO

__all__ = [
    # Protocols
    "SignalRepository",
    "InstrumentRepository",
    "MacroRegimeRepository",
    "PortfolioRepository",
    "StockFilter",
    "FilterPipeline",
    "PortfolioOptimizer",
    "CovarianceEstimator",
    # Models
    "StockSignalDTO",
    "InstrumentDTO",
    "PortfolioDTO",
    "PositionDTO",
    "BlackLittermanViewDTO",
]
