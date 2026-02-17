from .infrastructure import LRUCache, CircuitBreaker, RateLimiter, retry_with_backoff, is_rate_limit_error
from .protocols import (
    AnalysisClientProtocol,
    ArticleScraperProtocol,
    CacheProtocol,
    CalendarsClientProtocol,
    CircuitBreakerProtocol,
    CorporateActionsClientProtocol,
    FinancialsClientProtocol,
    FundsClientProtocol,
    HoldersClientProtocol,
    MarketClientProtocol,
    MetadataClientProtocol,
    RateLimiterProtocol,
    ScreenerClientProtocol,
    SearchClientProtocol,
    SectorIndustryClientProtocol,
    StreamingClientProtocol,
    YFinanceClientProtocol,
)
from .ticker import (
    AnalysisClient,
    CorporateActionsClient,
    FinancialsClient,
    FundsClient,
    HoldersClient,
    MetadataClient,
)
from .market import (
    CalendarsClient,
    MarketClient,
    ScreenerClient,
    SearchClient,
    SectorIndustryClient,
    StreamingClient,
    AsyncStreamingClient,
)
from .news import ArticleScraper, ArticleResult, NewsClient, CountryNewsFetcher
from ._base import BaseClient
from ._facade import YFinanceClient, get_yfinance_client

__all__ = [
    # Core client
    "YFinanceClient",
    "get_yfinance_client",
    # Base
    "BaseClient",
    # Sub-clients (ticker-based)
    "FinancialsClient",
    "AnalysisClient",
    "HoldersClient",
    "CorporateActionsClient",
    "MetadataClient",
    "FundsClient",
    # Sub-clients (module-level)
    "MarketClient",
    "SectorIndustryClient",
    "SearchClient",
    "ScreenerClient",
    "CalendarsClient",
    "StreamingClient",
    "AsyncStreamingClient",
    # Protocols
    "ArticleScraperProtocol",
    "CacheProtocol",
    "CircuitBreakerProtocol",
    "RateLimiterProtocol",
    "YFinanceClientProtocol",
    "FinancialsClientProtocol",
    "AnalysisClientProtocol",
    "HoldersClientProtocol",
    "CorporateActionsClientProtocol",
    "MetadataClientProtocol",
    "FundsClientProtocol",
    "MarketClientProtocol",
    "SectorIndustryClientProtocol",
    "SearchClientProtocol",
    "ScreenerClientProtocol",
    "CalendarsClientProtocol",
    "StreamingClientProtocol",
    # Implementations
    "LRUCache",
    "RateLimiter",
    "CircuitBreaker",
    # News/Articles
    "NewsClient",
    "ArticleScraper",
    "ArticleResult",
    "CountryNewsFetcher",
    # Utilities
    "retry_with_backoff",
    "is_rate_limit_error",
]
