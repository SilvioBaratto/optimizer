"""
Domain Models - Pure Python dataclasses for domain objects.

These models are:
- Database-independent (no SQLAlchemy dependencies)
- Immutable where possible (frozen dataclasses)
- Serializable (JSON/dict conversion methods)
- Validated (using __post_init__ for invariants)

Naming convention: *DTO suffix for Data Transfer Objects.
"""

from optimizer.domain.models.stock_signal import StockSignalDTO, SignalType, ConfidenceLevel, RiskLevel
from optimizer.domain.models.instrument import InstrumentDTO, ExchangeDTO
from optimizer.domain.models.portfolio import PortfolioDTO, PositionDTO
from optimizer.domain.models.view import BlackLittermanViewDTO, MacroRegimeDTO

__all__ = [
    # Stock signals
    "StockSignalDTO",
    "SignalType",
    "ConfidenceLevel",
    "RiskLevel",
    # Instruments
    "InstrumentDTO",
    "ExchangeDTO",
    # Portfolio
    "PortfolioDTO",
    "PositionDTO",
    # Views
    "BlackLittermanViewDTO",
    "MacroRegimeDTO",
]
