"""
Filters Module - Stock filtering implementations using Strategy and Composite patterns.

This module provides:
- StockFilter protocol implementation
- FilterPipeline for composing multiple filters
- Individual filter implementations (Quality, Affordability, Sector, Correlation)

Usage:
    from src.risk_management.filters import FilterPipelineImpl, QualityFilter, AffordabilityFilter

    # Create individual filters
    quality_filter = QualityFilter(config=quality_config)
    affordability_filter = AffordabilityFilter(max_price=75.0)

    # Create pipeline
    pipeline = FilterPipelineImpl()
    pipeline.add_filter(quality_filter)
    pipeline.add_filter(affordability_filter)

    # Filter signals
    passed_signals, stats = pipeline.filter_batch(signals)
"""

from optimizer.src.risk_management.filters.protocol import StockFilterImpl, CompositeFilter
from optimizer.src.risk_management.filters.pipeline import FilterPipelineImpl
from optimizer.src.risk_management.filters.quality_filter import QualityFilterImpl
from optimizer.src.risk_management.filters.affordability_filter import AffordabilityFilter
from optimizer.src.risk_management.filters.sector_filter import SectorFilter, SectorBalancer
from optimizer.src.risk_management.filters.country_filter import CountryFilter, RegionalFilter
from optimizer.src.risk_management.filters.correlation_filter import CorrelationFilterImpl

# Re-export protocol from domain for convenience
from optimizer.domain.protocols.filter import StockFilter

__all__ = [
    # Protocol
    "StockFilter",
    # Base implementations
    "StockFilterImpl",
    "CompositeFilter",
    "FilterPipelineImpl",
    # Individual filters
    "QualityFilterImpl",
    "AffordabilityFilter",
    "SectorFilter",
    "SectorBalancer",
    "CountryFilter",
    "RegionalFilter",
    "CorrelationFilterImpl",
]
