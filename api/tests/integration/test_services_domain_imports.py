"""Integration tests: service sub-packages resolve from domain folders (issue #611).

Verifies that the three moved sub-packages (yfinance, trading212, scrapers)
are importable from their new domain locations.
"""

from __future__ import annotations

import pytest


class TestYFinanceDomainImport:
    """yfinance sub-package must resolve from market_data domain."""

    def test_yfinance_client_importable_from_market_data(self) -> None:
        from app.services.market_data.yfinance import (
            YFinanceClient,
            get_yfinance_client,
        )

        assert YFinanceClient is not None
        assert get_yfinance_client is not None

    def test_yfinance_protocols_importable_from_market_data(self) -> None:
        from app.services.market_data.yfinance import YFinanceClientProtocol

        assert YFinanceClientProtocol is not None

    def test_yfinance_infrastructure_importable_from_market_data(self) -> None:
        from app.services.market_data.yfinance import (
            CircuitBreaker,
            LRUCache,
            RateLimiter,
        )

        assert CircuitBreaker is not None
        assert LRUCache is not None
        assert RateLimiter is not None

    def test_yfinance_news_subpackage_importable(self) -> None:
        from app.services.market_data.yfinance.news import MacroNewsFetcher, MacroTheme

        assert MacroNewsFetcher is not None
        assert MacroTheme is not None

    def test_yfinance_market_subpackage_importable(self) -> None:
        from app.services.market_data.yfinance.market import (
            MarketClient,
            ScreenerClient,
            SearchClient,
            SectorIndustryClient,
        )

        assert MarketClient is not None
        assert ScreenerClient is not None
        assert SearchClient is not None
        assert SectorIndustryClient is not None

    def test_yfinance_ticker_subpackage_importable(self) -> None:
        from app.services.market_data.yfinance.ticker import (
            AnalysisClient,
            FinancialsClient,
            FundsClient,
            MetadataClient,
        )

        assert AnalysisClient is not None
        assert FinancialsClient is not None
        assert FundsClient is not None
        assert MetadataClient is not None

    def test_old_yfinance_import_path_does_not_resolve(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            from app.services.yfinance_data_service import (  # noqa: F401
                YFinanceDataService,
            )


class TestTrading212DomainImport:
    """trading212 sub-package must resolve from universe domain."""

    def test_trading212_client_importable_from_universe(self) -> None:
        from app.services.universe.trading212 import (
            Trading212Client,
            UniverseBuilder,
            UniverseBuilderConfig,
        )

        assert Trading212Client is not None
        assert UniverseBuilder is not None
        assert UniverseBuilderConfig is not None

    def test_trading212_filters_importable(self) -> None:
        from app.services.universe.trading212.filters import (
            DataCoverageFilter,
            HistoricalDataFilter,
            LiquidityFilter,
            MarketCapFilter,
            PriceFilter,
        )

        assert DataCoverageFilter is not None
        assert HistoricalDataFilter is not None
        assert LiquidityFilter is not None
        assert MarketCapFilter is not None
        assert PriceFilter is not None

    def test_trading212_cache_importable(self) -> None:
        from app.services.universe.trading212.cache.ticker_cache import (
            TickerMappingCache,
        )

        assert TickerMappingCache is not None

    def test_old_trading212_import_path_does_not_resolve(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            from app.services.trading212_client import (  # noqa: F401
                Trading212Client,
            )


class TestScrapersDomainImport:
    """scrapers sub-package must resolve from macro domain."""

    def test_scrapers_importable_from_macro(self) -> None:
        from app.services.macro.scrapers import (
            FRED_SERIES,
            PORTFOLIO_COUNTRIES,
            FredScraper,
            IlSoleScraper,
            TradingEconomicsIndicatorsScraper,
        )

        assert FredScraper is not None
        assert IlSoleScraper is not None
        assert TradingEconomicsIndicatorsScraper is not None
        assert FRED_SERIES is not None
        assert PORTFOLIO_COUNTRIES is not None

    def test_old_scrapers_import_path_does_not_resolve(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            from app.services.scrapers import FredScraper  # noqa: F401


class TestServicesInitReExports:
    """services/__init__.py re-exports must resolve from new locations."""

    def test_trading212_names_still_re_exported_from_services_package(self) -> None:
        from app.services import (
            BuildProgress,
            BuildResult,
            Trading212Client,
            UniverseBuilder,
            UniverseBuilderConfig,
            YFinanceTickerMapper,
        )

        assert Trading212Client is not None
        assert UniverseBuilder is not None
        assert UniverseBuilderConfig is not None
        assert YFinanceTickerMapper is not None
        assert BuildProgress is not None
        assert BuildResult is not None
