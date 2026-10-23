"""yfinance market-data integration: facade, sub-clients, and resilience primitives."""

from ._base import BaseClient
from ._config import configure_yfinance
from ._facade import YFinanceClient, get_yfinance_client
from .infrastructure import (
    CircuitBreaker,
    LRUCache,
    RateLimiter,
    is_no_data_error,
    is_rate_limit_error,
    is_symbol_not_found,
    retry_with_backoff,
)
from .market import SearchClient
from .news import (
    ArticleResult,
    ArticleScraper,
    CountryNewsFetcher,
    MacroNewsFetcher,
    MacroTheme,
    NewsClient,
)
from .protocols import (
    AnalysisClientProtocol,
    ArticleScraperProtocol,
    CacheProtocol,
    CircuitBreakerProtocol,
    CorporateActionsClientProtocol,
    FinancialsClientProtocol,
    HoldersClientProtocol,
    MetadataClientProtocol,
    RateLimiterProtocol,
    ScreenerClientProtocol,
    SearchClientProtocol,
    YFinanceClientProtocol,
)
from .screener import ScreenerClient
from .ticker import (
    AnalysisClient,
    CorporateActionsClient,
    FinancialsClient,
    HoldersClient,
    MetadataClient,
)

__all__ = [
    "AnalysisClient",
    "AnalysisClientProtocol",
    "ArticleResult",
    "ArticleScraper",
    "ArticleScraperProtocol",
    "BaseClient",
    "CacheProtocol",
    "CircuitBreaker",
    "CircuitBreakerProtocol",
    "CorporateActionsClient",
    "CorporateActionsClientProtocol",
    "CountryNewsFetcher",
    "FinancialsClient",
    "FinancialsClientProtocol",
    "HoldersClient",
    "HoldersClientProtocol",
    "LRUCache",
    "MacroNewsFetcher",
    "MacroTheme",
    "MetadataClient",
    "MetadataClientProtocol",
    "NewsClient",
    "RateLimiter",
    "RateLimiterProtocol",
    "ScreenerClient",
    "ScreenerClientProtocol",
    "SearchClient",
    "SearchClientProtocol",
    "YFinanceClient",
    "YFinanceClientProtocol",
    "configure_yfinance",
    "get_yfinance_client",
    "is_no_data_error",
    "is_rate_limit_error",
    "is_symbol_not_found",
    "retry_with_backoff",
]
