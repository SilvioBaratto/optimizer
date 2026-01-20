"""
YFinance Client - Core client with dependency injection.

Slimmed-down client focusing on core yfinance operations:
- Ticker caching and retrieval
- Info fetching
- History fetching
- Bulk download

News and article fetching moved to separate NewsClient for
Interface Segregation Principle compliance.
"""

import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from optimizer.src.yfinance.cache import LRUCache
from optimizer.src.yfinance.circuit_breaker import CircuitBreaker
from optimizer.src.yfinance.protocols import (
    CacheProtocol,
    CircuitBreakerProtocol,
    RateLimiterProtocol,
)
from optimizer.src.yfinance.rate_limiter import RateLimiter

# Load environment variables from optimizer/.env
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class YFinanceClient:
    """
    Core yfinance client with dependency injection.

    Focused on core operations: ticker caching, info/history fetching,
    and bulk downloads. Uses injected dependencies for:
    - Cache (CacheProtocol)
    - Rate limiter (RateLimiterProtocol)
    - Circuit breaker (CircuitBreakerProtocol)

    For news fetching, use NewsClient which composes this client.

    Singleton pattern - use YFinanceClient.get_instance() or
    get_yfinance_client() factory function.

    Attributes:
        cache: Cache for Ticker objects
        rate_limiter: Rate limiter for API calls
        circuit_breaker: Circuit breaker for rate limit handling
        default_max_retries: Default retry attempts
    """

    _instance: "YFinanceClient | None" = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        cache_size: int = 3000,
        rate_limit_delay: float = 0.1,
        default_max_retries: int = 3,
    ) -> "YFinanceClient":
        """
        Get or create singleton instance.

        Args:
            cache_size: Number of Ticker objects to cache
            rate_limit_delay: Minimum delay between requests in seconds
            default_max_retries: Default number of retry attempts

        Returns:
            Singleton YFinanceClient instance

        Note:
            Configuration parameters only apply on first call.
            Subsequent calls return the existing instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cache=LRUCache(capacity=cache_size),
                        rate_limiter=RateLimiter(delay=rate_limit_delay),
                        circuit_breaker=CircuitBreaker(),
                        default_max_retries=default_max_retries,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (mainly for testing).

        Clears the cached instance, allowing a new one to be created
        with different configuration.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cache.clear()
                cls._instance = None

    def __init__(
        self,
        cache: CacheProtocol,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        """
        Initialize client with injected dependencies.

        Note: Use get_instance() instead of calling this directly
        for singleton behavior.

        Args:
            cache: Cache implementation (CacheProtocol)
            rate_limiter: Rate limiter implementation (RateLimiterProtocol)
            circuit_breaker: Circuit breaker implementation (CircuitBreakerProtocol)
            default_max_retries: Default number of retry attempts
        """
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Get cached Ticker object for a symbol.

        If not in cache, creates new Ticker object and caches it.
        Applies rate limiting before creating new Ticker.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "SPY")

        Returns:
            yfinance Ticker object

        Example:
            ticker = client.get_ticker("AAPL")
            info = ticker.info
        """
        ticker = self.cache.get(symbol)

        if ticker is not None:
            return ticker

        self.rate_limiter.acquire(symbol)

        ticker = yf.Ticker(symbol)
        self.cache.put(symbol, ticker)

        return ticker

    def fetch_info(
        self,
        symbol: str,
        max_retries: int | None = None,
        min_fields: int = 10,
    ) -> dict[str, Any] | None:
        """
        Fetch ticker info with retry logic.

        Args:
            symbol: Ticker symbol
            max_retries: Maximum retry attempts (uses default if None)
            min_fields: Minimum number of fields required for valid response

        Returns:
            Dictionary with ticker info or None if fetch failed

        Example:
            info = client.fetch_info("AAPL")
            if info:
                sector = info.get('sector')
        """
        max_retries = max_retries or self.default_max_retries

        for attempt in range(max_retries):
            self.circuit_breaker.check()

            try:
                ticker = self.get_ticker(symbol)
                info = ticker.info

                if not info or len(info) < min_fields:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return None

                self.circuit_breaker.reset()
                return info

            except Exception as e:
                if self._is_rate_limit_error(e):
                    self.circuit_breaker.trigger()
                    continue

                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return None

        return None

    def fetch_history(
        self,
        symbol: str,
        period: str = "2y",
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        max_retries: int | None = None,
        min_rows: int = 10,
    ) -> pd.DataFrame | None:
        """
        Fetch historical price data with retry logic.

        Args:
            symbol: Ticker symbol
            period: Period to fetch (e.g., "1y", "2y", "5y", "max")
            start: Start date string (YYYY-MM-DD) - overrides period
            end: End date string (YYYY-MM-DD)
            interval: Data interval (e.g., "1d", "1wk", "1mo")
            max_retries: Maximum retry attempts (uses default if None)
            min_rows: Minimum number of rows required for valid response

        Returns:
            DataFrame with OHLCV data or None if fetch failed

        Example:
            hist = client.fetch_history("AAPL", period="1y")
            hist = client.fetch_history("AAPL", start="2023-01-01", end="2023-12-31")
        """
        max_retries = max_retries or self.default_max_retries

        for attempt in range(max_retries):
            self.circuit_breaker.check()

            try:
                ticker = self.get_ticker(symbol)

                if start is not None:
                    hist = ticker.history(start=start, end=end, interval=interval)
                else:
                    hist = ticker.history(period=period, interval=interval)

                if hist is None or hist.empty or len(hist) < min_rows:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return None

                self.circuit_breaker.reset()
                return hist

            except Exception as e:
                if self._is_rate_limit_error(e):
                    self.circuit_breaker.trigger()
                    continue

                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return None

        return None

    def fetch_price_and_benchmark(
        self,
        symbol: str,
        benchmark: str = "SPY",
        period: str = "2y",
        max_retries: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
        """
        Fetch stock data and benchmark data in one call.

        Common pattern - fetches stock history, benchmark history,
        and stock info together with date alignment.

        Args:
            symbol: Stock ticker symbol
            benchmark: Benchmark ticker (default: SPY)
            period: Historical period to fetch
            max_retries: Maximum retry attempts (uses default if None)

        Returns:
            Tuple of (stock_hist, benchmark_hist, stock_info)
            Returns (None, None, None) if fetch fails

        Example:
            stock_hist, spy_hist, info = client.fetch_price_and_benchmark("AAPL")
        """
        max_retries = max_retries or self.default_max_retries

        stock_hist = self.fetch_history(symbol, period=period, max_retries=max_retries)
        stock_info = self.fetch_info(symbol, max_retries=max_retries)

        if stock_hist is None or stock_hist.empty:
            return None, None, None

        benchmark_hist = self.fetch_history(benchmark, period=period, max_retries=max_retries)

        if benchmark_hist is None or benchmark_hist.empty:
            return stock_hist, None, stock_info

        # Align dates (timezone-agnostic matching)
        try:
            stock_date_strs = [ts.strftime("%Y-%m-%d") for ts in stock_hist.index]
            bench_date_strs = [ts.strftime("%Y-%m-%d") for ts in benchmark_hist.index]

            common_date_strs = set(stock_date_strs) & set(bench_date_strs)

            if len(common_date_strs) > 0:
                stock_mask = pd.Series(stock_date_strs).isin(common_date_strs).values
                bench_mask = pd.Series(bench_date_strs).isin(common_date_strs).values

                stock_hist = stock_hist[stock_mask]
                benchmark_hist = benchmark_hist[bench_mask]
            else:
                benchmark_hist = None
        except Exception:
            benchmark_hist = None

        return stock_hist, benchmark_hist, stock_info

    def bulk_download(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str = "2y",
        interval: str = "1d",
        threads: bool = True,
        group_by: str = "ticker",
        auto_adjust: bool = False,
        progress: bool = False,
    ) -> pd.DataFrame | None:
        """
        Bulk download multiple tickers using yfinance.download().

        More efficient than fetching tickers individually.

        Args:
            symbols: List of ticker symbols
            start: Start date string (YYYY-MM-DD)
            end: End date string (YYYY-MM-DD)
            period: Period to fetch (ignored if start is provided)
            interval: Data interval
            threads: Use threading for parallel downloads
            group_by: How to group results ('ticker' or 'column')
            auto_adjust: Adjust all OHLC automatically
            progress: Show download progress bar

        Returns:
            DataFrame with price data (multi-index columns) or None if failed

        Example:
            data = client.bulk_download(["AAPL", "MSFT", "GOOGL"], period="1y")
            aapl_data = data["AAPL"]
        """
        try:
            self.rate_limiter.acquire("bulk_download")

            if start is not None:
                data = yf.download(
                    tickers=symbols,
                    start=start,
                    end=end,
                    interval=interval,
                    threads=threads,
                    group_by=group_by,
                    auto_adjust=auto_adjust,
                    progress=progress,
                )
            else:
                data = yf.download(
                    tickers=symbols,
                    period=period,
                    interval=interval,
                    threads=threads,
                    group_by=group_by,
                    auto_adjust=auto_adjust,
                    progress=progress,
                )

            if data is None or (hasattr(data, "empty") and data.empty):
                return None

            return data

        except Exception:
            return None

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size, capacity, and config info
        """
        return {
            "size": self.cache.size(),
            "capacity": getattr(self.cache, "capacity", "unknown"),
            "default_max_retries": self.default_max_retries,
        }

    def clear_cache(self) -> None:
        """Clear all cached Ticker objects."""
        self.cache.clear()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if an error indicates rate limiting.

        Args:
            error: Exception to check

        Returns:
            True if error indicates rate limiting
        """
        error_msg = str(error)
        rate_limit_indicators = ["Too Many Requests", "Rate limited", "429"]
        return any(indicator in error_msg for indicator in rate_limit_indicators)


def get_yfinance_client(
    cache_size: int = 3000,
    rate_limit_delay: float = 0.1,
    default_max_retries: int = 3,
) -> YFinanceClient:
    """
    Get the singleton YFinanceClient instance.

    Factory function that wires up dependencies and returns
    the singleton instance.

    Args:
        cache_size: Number of Ticker objects to cache
        rate_limit_delay: Minimum delay between requests in seconds
        default_max_retries: Default number of retry attempts

    Returns:
        Singleton YFinanceClient instance

    Example:
        from optimizer.src.yfinance import get_yfinance_client

        client = get_yfinance_client()
        info = client.fetch_info("AAPL")
    """
    return YFinanceClient.get_instance(
        cache_size=cache_size,
        rate_limit_delay=rate_limit_delay,
        default_max_retries=default_max_retries,
    )
