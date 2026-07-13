from ._base import BaseClient
from ._facade import YFinanceClient, get_yfinance_client
from .infrastructure import (
    CircuitBreaker,
    LRUCache,
    RateLimiter,
    is_rate_limit_error,
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
    SearchClientProtocol,
    YFinanceClientProtocol,
)
from .ticker import (
    AnalysisClient,
    CorporateActionsClient,
    FinancialsClient,
    HoldersClient,
    MetadataClient,
)

__all__ = [
    # Sub-clients (ticker-based)
    "AnalysisClient",
    "AnalysisClientProtocol",
    "ArticleResult",
    "ArticleScraper",
    # Protocols
    "ArticleScraperProtocol",
    # Base
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
    # Implementations
    "LRUCache",
    "MacroNewsFetcher",
    "MacroTheme",
    "MetadataClient",
    "MetadataClientProtocol",
    # News/Articles
    "NewsClient",
    "RateLimiter",
    "RateLimiterProtocol",
    # Sub-clients (module-level)
    "SearchClient",
    "SearchClientProtocol",
    # Core client
    "YFinanceClient",
    "YFinanceClientProtocol",
    "get_yfinance_client",
    "is_rate_limit_error",
    # Utilities
    "retry_with_backoff",
]
