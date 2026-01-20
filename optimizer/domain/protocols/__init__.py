"""
Domain Protocols - Interface definitions using Python Protocol.

These protocols define contracts for dependency injection, enabling:
- Testability through mocking
- Decoupling business logic from infrastructure
- Easy swapping of implementations
"""

from optimizer.domain.protocols.repository import (
    SignalRepository,
    InstrumentRepository,
    MacroRegimeRepository,
    PortfolioRepository,
)
from optimizer.domain.protocols.filter import StockFilter, FilterPipeline
from optimizer.domain.protocols.optimizer import PortfolioOptimizer, CovarianceEstimator
from optimizer.domain.protocols.universe import (
    InstrumentFilter,
    FilterPipeline as UniverseFilterPipeline,
    TickerMapper,
    TickerCache,
    UniverseRepository,
    Trading212ApiClient,
)

__all__ = [
    "SignalRepository",
    "InstrumentRepository",
    "MacroRegimeRepository",
    "PortfolioRepository",
    "StockFilter",
    "FilterPipeline",
    "PortfolioOptimizer",
    "CovarianceEstimator",
    # Universe protocols
    "InstrumentFilter",
    "UniverseFilterPipeline",
    "TickerMapper",
    "TickerCache",
    "UniverseRepository",
    "Trading212ApiClient",
]
