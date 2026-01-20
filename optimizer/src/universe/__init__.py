"""
Universe Building Module - T212 to yfinance ticker mapping with institutional filters.

This module provides a complete pipeline for building a stock universe:
1. Fetch metadata from Trading212 API
2. Map T212 tickers to yfinance tickers
3. Apply institutional filters (market cap, liquidity, data coverage)
4. Persist to database

Components:
- UniverseBuilder: Main orchestrator
- Trading212Client: API client for T212
- YFinanceTickerMapper: Ticker discovery service
- FilterPipelineImpl: Composable filter pipeline
- UniverseRepositoryImpl: Database persistence

Usage:
    from optimizer.config.universe_builder_config import UniverseBuilderConfig
    from optimizer.src.universe import UniverseBuilder
    from optimizer.src.universe.api import Trading212Client
    from optimizer.src.universe.services import YFinanceTickerMapper
    from optimizer.src.universe.filters import (
        FilterPipelineImpl,
        MarketCapFilter,
        PriceFilter,
        LiquidityFilter,
    )
    from optimizer.database.repositories.universe_repository import UniverseRepositoryImpl

    config = UniverseBuilderConfig()
    pipeline = FilterPipelineImpl()
    pipeline.add_filter(MarketCapFilter(config))
    pipeline.add_filter(PriceFilter(config))
    pipeline.add_filter(LiquidityFilter(config))

    builder = UniverseBuilder(
        config=config,
        api_client=Trading212Client.from_env(),
        ticker_mapper=YFinanceTickerMapper(config),
        filter_pipeline=pipeline,
        repository=UniverseRepositoryImpl(),
    )

    exchanges, instruments = builder.build()
"""

from optimizer.src.universe.builder import UniverseBuilder

__all__ = ["UniverseBuilder"]
