"""
YFinance Module - SOLID-Compliant yfinance Client
=================================================

Provides modular, dependency-injected components for yfinance API access.

Architecture:
- protocols.py: Protocol definitions (CacheProtocol, RateLimiterProtocol, etc.)
- cache.py: LRU cache implementation
- rate_limiter.py: Thread-safe rate limiter
- circuit_breaker.py: Exponential backoff circuit breaker
- client.py: Core YFinanceClient (info, history, bulk download)
- news_client.py: NewsClient (news fetching with optional content)
- article_scraper.py: Web scraping for full article content

Key Features:
- Dependency injection through protocols
- Single Responsibility Principle - focused classes
- Open/Closed Principle - extend via new implementations
- Interface Segregation - separate NewsClient for news operations
- Testable - easy to mock dependencies

Usage:
    from optimizer.src.yfinance import YFinanceClient, get_yfinance_client

    # Get singleton instance
    client = get_yfinance_client()

    # Fetch data
    info = client.fetch_info("AAPL")
    hist = client.fetch_history("AAPL", period="1y")
    stock_hist, spy_hist, info = client.fetch_price_and_benchmark("AAPL")

    # Get cached Ticker object (for advanced usage)
    ticker = client.get_ticker("AAPL")

For news fetching:
    from optimizer.src.yfinance import get_yfinance_client
    from optimizer.src.yfinance.news_client import NewsClient

    client = get_yfinance_client()
    news_client = NewsClient(yf_client=client)
    news = news_client.fetch("AAPL", fetch_full_content=True, max_articles=5)
"""

from optimizer.src.yfinance.article_scraper import ArticleScraper
from optimizer.src.yfinance.cache import LRUCache
from optimizer.src.yfinance.circuit_breaker import CircuitBreaker
from optimizer.src.yfinance.client import YFinanceClient, get_yfinance_client
from optimizer.src.yfinance.news_client import NewsClient
from optimizer.src.yfinance.protocols import (
    CacheProtocol,
    CircuitBreakerProtocol,
    RateLimiterProtocol,
)
from optimizer.src.yfinance.rate_limiter import RateLimiter

__all__ = [
    # Core client
    "YFinanceClient",
    "get_yfinance_client",
    # Protocols
    "CacheProtocol",
    "RateLimiterProtocol",
    "CircuitBreakerProtocol",
    # Implementations
    "LRUCache",
    "RateLimiter",
    "CircuitBreaker",
    # News/Articles
    "NewsClient",
    "ArticleScraper",
]
