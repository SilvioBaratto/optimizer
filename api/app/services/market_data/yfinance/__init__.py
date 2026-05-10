from ._base import BaseClient
from ._facade import YFinanceClient, get_yfinance_client
from .infrastructure import (
    CircuitBreaker,
    LRUCache,
    RateLimiter,
    is_rate_limit_error,
    retry_with_backoff,
)
from .market import (
    AsyncStreamingClient,
    CalendarsClient,
    MarketClient,
    ScreenerClient,
    SearchClient,
    SectorIndustryClient,
    StreamingClient,
)
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
    AsyncStreamingClientProtocol,
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

__all__ = [
    "AnalysisClient",
    "AnalysisClientProtocol",
    "ArticleResult",
    "ArticleScraper",
    # Protocols
    "ArticleScraperProtocol",
    "AsyncStreamingClient",
    "AsyncStreamingClientProtocol",
    # Base
    "BaseClient",
    "CacheProtocol",
    "CalendarsClient",
    "CalendarsClientProtocol",
    "CircuitBreaker",
    "CircuitBreakerProtocol",
    "CorporateActionsClient",
    "CorporateActionsClientProtocol",
    "CountryNewsFetcher",
    # Sub-clients (ticker-based)
    "FinancialsClient",
    "FinancialsClientProtocol",
    "FundsClient",
    "FundsClientProtocol",
    "HoldersClient",
    "HoldersClientProtocol",
    # Implementations
    "LRUCache",
    "MacroNewsFetcher",
    "MacroTheme",
    # Sub-clients (module-level)
    "MarketClient",
    "MarketClientProtocol",
    "MetadataClient",
    "MetadataClientProtocol",
    # News/Articles
    "NewsClient",
    "RateLimiter",
    "RateLimiterProtocol",
    "ScreenerClient",
    "ScreenerClientProtocol",
    "SearchClient",
    "SearchClientProtocol",
    "SectorIndustryClient",
    "SectorIndustryClientProtocol",
    "StreamingClient",
    "StreamingClientProtocol",
    # Core client
    "YFinanceClient",
    "YFinanceClientProtocol",
    "get_yfinance_client",
    "is_rate_limit_error",
    # Utilities
    "retry_with_backoff",
]
