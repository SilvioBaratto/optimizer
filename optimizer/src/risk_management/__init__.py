"""
Risk Management Module - Stock Selection for Portfolio Optimization
====================================================================

This is the refactored version following SOLID principles:
- Filter Pipeline: Composite pattern for filter chaining
- Single Responsibility: Separate filters for quality, affordability, sector, etc.
- Open/Closed: Easy to add new filters without modifying existing code
- Dependency Injection: Filters injected via configuration

Components:
- filters: Filter protocol and implementations (Quality, Affordability, Sector, etc.)
- ConcentratedPortfolioBuilder: Stock selector orchestrator
- CorrelationAnalyzer: Correlation analysis and diversification
- SectorAllocator: Sector constraints

Usage:
    from src.risk_management import ConcentratedPortfolioBuilder
    from src.risk_management.filters import FilterPipelineImpl, QualityFilterImpl
    from config import QualityFilterConfig

    # Using filter pipeline
    pipeline = FilterPipelineImpl()
    pipeline.add_filter(QualityFilterImpl(config=QualityFilterConfig()))
    filtered = pipeline.filter(signals)

    # Or using builder (includes filters internally)
    builder = ConcentratedPortfolioBuilder(target_positions=20)
    selected_stocks = builder.build_portfolio()
"""

# New SOLID-compliant filter imports
from optimizer.src.risk_management.filters import (
    StockFilter,
    StockFilterImpl,
    FilterPipelineImpl,
    CompositeFilter,
    QualityFilterImpl,
    AffordabilityFilter,
    SectorFilter,
    SectorBalancer,
    CountryFilter,
    RegionalFilter,
    CorrelationFilterImpl,
)

# Legacy imports for backward compatibility
from .concentrated_portfolio_builder import (
    ConcentratedPortfolioBuilder
)

from .quality_filter import (
    QualityFilter,
    QualityMetrics
)

from .correlation_analyzer import (
    CorrelationAnalyzer
)

from .sector_allocator import (
    SectorAllocator,
    DEFENSIVE_SECTORS,
    CYCLICAL_SECTORS,
    ALL_SECTORS
)

from .portfolio_analytics import (
    PortfolioAnalytics
)

__all__ = [
    # Main stock selector
    "ConcentratedPortfolioBuilder",

    # New filter pipeline
    "StockFilter",
    "StockFilterImpl",
    "FilterPipelineImpl",
    "CompositeFilter",

    # Individual filters (new)
    "QualityFilterImpl",
    "AffordabilityFilter",
    "SectorFilter",
    "SectorBalancer",
    "CountryFilter",
    "RegionalFilter",
    "CorrelationFilterImpl",

    # Legacy quality filtering
    "QualityFilter",
    "QualityMetrics",

    # Correlation analysis
    "CorrelationAnalyzer",

    # Sector allocation
    "SectorAllocator",
    "DEFENSIVE_SECTORS",
    "CYCLICAL_SECTORS",
    "ALL_SECTORS",

    # Utilities
    "PortfolioAnalytics",
]

__version__ = "3.0.0"
