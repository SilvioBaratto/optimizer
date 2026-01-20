"""
Universe Filters Package - Institutional stock filters for universe building.

This package implements the Strategy pattern for stock filtering, with each
filter as a separate class implementing the InstrumentFilter protocol.

Filters:
- MarketCapFilter: Minimum market cap ($100M default)
- PriceFilter: Price range validation ($5-$10,000)
- LiquidityFilter: ADV thresholds by market cap segment
- DataCoverageFilter: Institutional data completeness (100% required fields)
- HistoricalDataFilter: Historical data availability (5 years)

Pipeline:
- FilterPipelineImpl: Composite filter that chains all filters

Usage:
    from optimizer.src.universe.filters import (
        FilterPipelineImpl,
        MarketCapFilter,
        PriceFilter,
        LiquidityFilter,
        DataCoverageFilter,
        HistoricalDataFilter,
    )
    from optimizer.config.universe_builder_config import UniverseBuilderConfig

    config = UniverseBuilderConfig()

    pipeline = FilterPipelineImpl()
    pipeline.add_filter(MarketCapFilter(config))
    pipeline.add_filter(PriceFilter(config))
    pipeline.add_filter(LiquidityFilter(config))
    pipeline.add_filter(DataCoverageFilter(config))
    pipeline.add_filter(HistoricalDataFilter(config))

    passed, reason = pipeline.apply(yfinance_data, "AAPL")
"""

from optimizer.src.universe.filters.market_cap import MarketCapFilter
from optimizer.src.universe.filters.price import PriceFilter
from optimizer.src.universe.filters.liquidity import LiquidityFilter
from optimizer.src.universe.filters.data_coverage import DataCoverageFilter
from optimizer.src.universe.filters.historical_data import HistoricalDataFilter
from optimizer.src.universe.filters.pipeline import FilterPipelineImpl

__all__ = [
    "MarketCapFilter",
    "PriceFilter",
    "LiquidityFilter",
    "DataCoverageFilter",
    "HistoricalDataFilter",
    "FilterPipelineImpl",
]
