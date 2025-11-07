"""
YFinance Client - Centralized Client with Caching and Rate Limiting
====================================================================

Singleton client for yfinance API calls with:
- Ticker object caching (LRU)
- Rate limiting to avoid throttling
- Retry logic for transient failures
- Thread-safe operations
- Common methods for frequent operations

Usage:
    from src.yfinance import YFinanceClient

    # Get singleton instance
    client = YFinanceClient.get_instance()

    # Fetch data
    info = client.fetch_info("AAPL")
    hist = client.fetch_history("AAPL", period="1y")
    news = client.fetch_news("AAPL")

    # Get cached Ticker object
    ticker = client.get_ticker("AAPL")

Author: Portfolio Optimization System
"""

import logging
import time
import threading
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from src.yfinance.cache import LRUCache

# Load environment variables from optimizer/.env
# Works for both local development and Docker (PYTHONPATH=/app)
_env_path = Path(__file__).parent.parent.parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path)
    logger_init = logging.getLogger(__name__)
    logger_init.debug(f"Loaded environment from {_env_path}")

logger = logging.getLogger(__name__)

# HTTP headers for article fetching
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Delay between article fetches (seconds) - be respectful to servers
ARTICLE_FETCH_DELAY = 1.0


class YFinanceClient:
    """
    Centralized yfinance client with caching and rate limiting.

    Singleton pattern - use YFinanceClient.get_instance() to get the shared instance.

    Features:
    - Caches Ticker objects to avoid redundant instantiation
    - Rate limiting to avoid API throttling
    - Retry logic for transient failures
    - Thread-safe operations
    - Common methods for frequently used operations

    Configuration:
    - cache_size: Number of Ticker objects to cache (default: 1000)
    - rate_limit_delay: Minimum delay between requests in seconds (default: 0.1)
    - default_max_retries: Default number of retry attempts (default: 3)
    """

    _instance: Optional['YFinanceClient'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        cache_size: int = 3000,
        rate_limit_delay: float = 0.1,
        default_max_retries: int = 3
    ) -> 'YFinanceClient':
        """
        Get or create singleton instance.

        Args:
            cache_size: Number of Ticker objects to cache
            rate_limit_delay: Minimum delay between requests in seconds
            default_max_retries: Default number of retry attempts

        Returns:
            Singleton YFinanceClient instance

        Note:
            Configuration parameters only apply on first call (when instance is created).
            Subsequent calls ignore these parameters and return the existing instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        cache_size=cache_size,
                        rate_limit_delay=rate_limit_delay,
                        default_max_retries=default_max_retries
                    )
                    logger.info(
                        f"YFinanceClient initialized: "
                        f"cache_size={cache_size}, "
                        f"rate_limit_delay={rate_limit_delay}s, "
                        f"default_max_retries={default_max_retries}"
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
                cls._instance._ticker_cache.clear()
                cls._instance = None
                logger.info("YFinanceClient instance reset")

    def __init__(
        self,
        cache_size: int = 3000,
        rate_limit_delay: float = 0.1,
        default_max_retries: int = 3
    ):
        """
        Initialize client with caching and rate limiting.

        Note: Use get_instance() instead of calling this directly.

        Args:
            cache_size: Number of Ticker objects to cache
            rate_limit_delay: Minimum delay between requests in seconds
            default_max_retries: Default number of retry attempts
        """
        self._ticker_cache = LRUCache(capacity=cache_size)
        self._rate_limit_delay = rate_limit_delay
        self._default_max_retries = default_max_retries
        self._last_request_time: Dict[str, float] = {}
        self._rate_limit_lock = threading.Lock()

        # Global circuit breaker for rate limiting
        self._circuit_breaker_active = False
        self._circuit_breaker_until = 0.0
        self._rate_limit_attempt = 0  # Exponential backoff counter
        self._circuit_breaker_lock = threading.Lock()


    def _trigger_circuit_breaker(self) -> None:
        """
        Trigger global circuit breaker when rate limit is detected.

        Blocks ALL API calls with exponential backoff:
        - Attempt 1: 2 minutes
        - Attempt 2: 4 minutes
        - Attempt 3: 8 minutes
        - And so on...
        """
        with self._circuit_breaker_lock:
            # If circuit breaker is already active, don't increment counter again
            # Multiple threads may detect rate limit simultaneously
            if self._circuit_breaker_active:
                now = time.time()
                if now < self._circuit_breaker_until:
                    # Circuit breaker already active and waiting, don't re-trigger
                    return

            # Use exponential backoff
            self._rate_limit_attempt += 1
            wait_seconds = (2 ** self._rate_limit_attempt) * 60
            logger.error("=" * 100)
            logger.error("🚨 YAHOO FINANCE RATE LIMIT DETECTED - CIRCUIT BREAKER ACTIVATED 🚨")
            logger.error("=" * 100)
            logger.error(f"Attempt: {self._rate_limit_attempt}")
            logger.error(f"Waiting: {wait_seconds / 60:.1f} minutes before resuming")
            logger.error(f"Resume at: {time.strftime('%H:%M:%S', time.localtime(time.time() + wait_seconds))}")
            logger.error("All yfinance API calls are BLOCKED until rate limit resets")
            logger.error("=" * 100)

            self._circuit_breaker_until = time.time() + wait_seconds
            self._circuit_breaker_active = True

    def _check_circuit_breaker(self) -> None:
        """
        Check if circuit breaker is active and wait if necessary.

        Raises:
            RuntimeError: If circuit breaker has failed too many times (safety limit)
        """
        # Check if we need to wait (outside lock to avoid blocking other threads)
        should_wait = False
        wait_time = 0
        resume_time_str = ""

        with self._circuit_breaker_lock:
            if self._circuit_breaker_active:
                now = time.time()
                if now < self._circuit_breaker_until:
                    should_wait = True
                    wait_time = self._circuit_breaker_until - now
                    resume_time_str = time.strftime('%H:%M:%S', time.localtime(self._circuit_breaker_until))
                else:
                    # Circuit breaker expired
                    logger.info("✅ Circuit breaker expired. Resuming API calls...")
                    self._circuit_breaker_active = False

            # Safety limit: If we've retried too many times, fail completely
            if self._rate_limit_attempt >= 10:
                logger.error(
                    f"❌ FATAL: Circuit breaker triggered {self._rate_limit_attempt} times. "
                    "Yahoo Finance is persistently rate limiting. Aborting."
                )
                raise RuntimeError(
                    f"Yahoo Finance rate limit persists after {self._rate_limit_attempt} attempts. "
                    "Total wait time exceeded safety limit. Aborting to prevent infinite loop."
                )

        # Sleep outside the lock so other threads can check status
        if should_wait:
            logger.warning(
                f"⏳ Circuit breaker active. Waiting {wait_time / 60:.1f} minutes "
                f"(resume at {resume_time_str})"
            )
            time.sleep(wait_time)

    def _reset_circuit_breaker(self) -> None:
        """
        Reset circuit breaker after successful API calls.

        Called after a successful request to gradually reduce backoff.
        """
        with self._circuit_breaker_lock:
            if self._rate_limit_attempt > 0:
                # Gradually reduce attempt counter on success
                self._rate_limit_attempt = max(0, self._rate_limit_attempt - 1)
                if self._rate_limit_attempt == 0:
                    logger.info("✅ Circuit breaker fully reset after successful calls")

    def _apply_rate_limit(self, symbol: str) -> None:
        """
        Apply rate limiting for a symbol.

        Ensures minimum delay between requests for the same symbol.

        Args:
            symbol: Ticker symbol
        """
        with self._rate_limit_lock:
            now = time.time()
            last_request = self._last_request_time.get(symbol, 0)
            elapsed = now - last_request

            if elapsed < self._rate_limit_delay:
                sleep_time = self._rate_limit_delay - elapsed
                logger.debug(f"Rate limiting {symbol}: sleeping {sleep_time:.3f}s")
                time.sleep(sleep_time)

            self._last_request_time[symbol] = time.time()

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Get cached Ticker object for a symbol.

        If not in cache, creates new Ticker object and caches it.
        Applies rate limiting before creating new Ticker.

        Note: yfinance now handles HTTP sessions internally with curl_cffi.
        We cache Ticker objects to avoid redundant instantiation.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "SPY")

        Returns:
            yfinance Ticker object

        Example:
            ticker = client.get_ticker("AAPL")
            info = ticker.info
            hist = ticker.history(period="1y")
        """
        # Check cache first
        ticker = self._ticker_cache.get(symbol)

        if ticker is not None:
            logger.debug(f"Cache hit for {symbol}")
            return ticker

        # Cache miss - create new Ticker
        logger.debug(f"Cache miss for {symbol} - creating new Ticker")
        self._apply_rate_limit(symbol)

        # Let yfinance handle session (uses curl_cffi internally)
        ticker = yf.Ticker(symbol)
        self._ticker_cache.put(symbol, ticker)

        return ticker

    def fetch_info(
        self,
        symbol: str,
        max_retries: Optional[int] = None,
        min_fields: int = 10
    ) -> Optional[Dict[str, Any]]:
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
                print(f"Sector: {info.get('sector')}")
                print(f"Price: ${info.get('currentPrice')}")
        """
        max_retries = max_retries or self._default_max_retries

        for attempt in range(max_retries):
            # Check circuit breaker before EVERY attempt
            self._check_circuit_breaker()

            try:
                ticker = self.get_ticker(symbol)
                info = ticker.info

                # Validate response
                if not info or len(info) < min_fields:
                    logger.warning(
                        f"Insufficient info data for {symbol} "
                        f"(got {len(info) if info else 0} fields, expected >={min_fields})"
                    )
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {symbol} (attempt {attempt + 2}/{max_retries})")
                        time.sleep(1 * (attempt + 1))  # Progressive backoff
                        continue
                    return None

                # Success! Reset circuit breaker gradually
                self._reset_circuit_breaker()
                logger.debug(f"Successfully fetched info for {symbol} ({len(info)} fields)")
                return info

            except Exception as e:
                error_msg = str(e)

                # Check if this is a rate limit error
                if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
                    logger.error(f"🚨 RATE LIMIT detected for {symbol}: {error_msg}")
                    # Trigger circuit breaker immediately - stops ALL API calls
                    self._trigger_circuit_breaker()
                    # After circuit breaker expires, retry this stock
                    continue

                logger.error(f"Error fetching info for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Progressive backoff
                    continue
                return None

        return None

    def fetch_history(
        self,
        symbol: str,
        period: str = "2y",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        max_retries: Optional[int] = None,
        min_rows: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data with retry logic.

        Args:
            symbol: Ticker symbol
            period: Period to fetch (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            start: Start date string (YYYY-MM-DD) - overrides period if provided
            end: End date string (YYYY-MM-DD)
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            max_retries: Maximum retry attempts (uses default if None)
            min_rows: Minimum number of rows required for valid response

        Returns:
            DataFrame with OHLCV data or None if fetch failed

        Example:
            # Using period
            hist = client.fetch_history("AAPL", period="1y")

            # Using date range
            hist = client.fetch_history("AAPL", start="2023-01-01", end="2023-12-31")
        """
        max_retries = max_retries or self._default_max_retries

        for attempt in range(max_retries):
            # Check circuit breaker before EVERY attempt
            self._check_circuit_breaker()

            try:
                ticker = self.get_ticker(symbol)

                # Fetch history based on parameters
                if start is not None:
                    hist = ticker.history(start=start, end=end, interval=interval)
                else:
                    hist = ticker.history(period=period, interval=interval)

                # Validate response
                if hist is None or hist.empty or len(hist) < min_rows:
                    logger.warning(
                        f"Insufficient history data for {symbol} "
                        f"(got {len(hist) if hist is not None and not hist.empty else 0} rows, expected >={min_rows})"
                    )
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {symbol} (attempt {attempt + 2}/{max_retries})")
                        time.sleep(1 * (attempt + 1))  # Progressive backoff
                        continue
                    return None

                # Success! Reset circuit breaker gradually
                self._reset_circuit_breaker()
                logger.debug(f"Successfully fetched {len(hist)} days of history for {symbol}")
                return hist

            except Exception as e:
                error_msg = str(e)

                # Check if this is a rate limit error
                if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
                    logger.error(f"🚨 RATE LIMIT detected for {symbol}: {error_msg}")
                    # Trigger circuit breaker immediately - stops ALL API calls
                    self._trigger_circuit_breaker()
                    # After circuit breaker expires, retry this stock
                    continue

                logger.error(f"Error fetching history for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Progressive backoff
                    continue
                return None

        return None

    def fetch_article_content(
        self,
        url: str,
        timeout: int = 10,
        delay: float = ARTICLE_FETCH_DELAY,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch full article content from URL using web scraping.

        Generic method that works for both country news and stock news.
        Uses BeautifulSoup to extract article text from common HTML patterns.

        Args:
            url: Article URL to fetch
            timeout: Request timeout in seconds (default: 10)
            delay: Delay before request to be respectful to servers (default: 1.0s)
            headers: Custom HTTP headers (default: Mozilla/5.0 User-Agent)

        Returns:
            Dictionary with keys:
                - success (bool): Whether fetch succeeded
                - content (str|None): Full article text
                - content_length (int|None): Character count
                - error (str|None): Error message if failed

        Example:
            result = client.fetch_article_content("https://...")
            if result['success']:
                print(f"Article length: {result['content_length']} chars")
                print(result['content'][:500])  # First 500 chars
        """
        if headers is None:
            headers = DEFAULT_HEADERS

        try:
            # Add delay to be respectful to servers
            if delay > 0:
                time.sleep(delay)

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # Try to find article content using common selectors
            article_content = None

            # Try common article containers (ordered by specificity)
            selectors = [
                'article',
                'div.article-body',
                'div.article-content',
                'div.entry-content',
                'div.post-content',
                'div.content-body',
                'div.story-body',
                'div.caas-body',  # Yahoo Finance specific
                'main',
            ]

            for selector in selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    article_content = content_div
                    break

            if not article_content:
                # Fallback: get all paragraphs from body
                article_content = soup.find('body')

            if article_content:
                # Extract text from paragraphs
                paragraphs = article_content.find_all('p')
                full_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

                return {
                    'success': True,
                    'content': full_text,
                    'content_length': len(full_text),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'content': None,
                    'content_length': None,
                    'error': 'Could not find article content'
                }

        except requests.exceptions.Timeout:
            return {
                'success': False,
                'content': None,
                'content_length': None,
                'error': 'Request timeout'
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'content': None,
                'content_length': None,
                'error': f'Request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'content': None,
                'content_length': None,
                'error': f'Parsing error: {str(e)}'
            }

    def fetch_news(
        self,
        symbol: str,
        max_retries: Optional[int] = None,
        fetch_full_content: bool = False,
        max_articles_with_content: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch news for a ticker with retry logic.

        Args:
            symbol: Ticker symbol (stock ticker or index like '^GSPC')
            max_retries: Maximum retry attempts (uses default if None)
            fetch_full_content: If True, fetch full article content from URLs (default: False)
            max_articles_with_content: If set, only fetch full content for first N articles (default: all)

        Returns:
            List of news articles or None if fetch failed
            Each article dict includes: title, publisher, link, providerPublishTime, summary
            If fetch_full_content=True, also includes: full_content, content_length (or content_error)

        Example:
            # Basic usage (metadata only)
            news = client.fetch_news("AAPL")

            # With full content
            news = client.fetch_news("AAPL", fetch_full_content=True, max_articles_with_content=10)
            if news:
                for article in news:
                    if 'full_content' in article:
                        print(f"{article['title']}: {len(article['full_content'])} chars")
        """
        max_retries = max_retries or self._default_max_retries

        for attempt in range(max_retries):
            try:
                ticker = self.get_ticker(symbol)
                news = ticker.news

                if news is None:
                    logger.warning(f"No news data for {symbol}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {symbol} (attempt {attempt + 2}/{max_retries})")
                        time.sleep(1 * (attempt + 1))  # Progressive backoff
                        continue
                    return None

                logger.debug(f"Successfully fetched {len(news)} news articles for {symbol}")

                # Fetch full content if requested
                if fetch_full_content:
                    articles_to_fetch = len(news) if max_articles_with_content is None else min(len(news), max_articles_with_content)
                    logger.debug(f"Fetching full content for {articles_to_fetch} articles from {symbol}")

                    for i, article in enumerate(news[:articles_to_fetch]):
                        # Extract link from nested structure
                        content = article.get('content', article)
                        link = None

                        # Try different link formats
                        if isinstance(content.get('canonicalUrl'), dict):
                            link = content['canonicalUrl'].get('url')
                        elif 'canonicalUrl' in content:
                            link = content['canonicalUrl']
                        elif 'link' in article:
                            link = article['link']
                        elif 'link' in content:
                            link = content['link']

                        if link:
                            # Fetch full content
                            content_result = self.fetch_article_content(link)
                            if content_result['success']:
                                article['full_content'] = content_result['content']
                                article['content_length'] = content_result['content_length']
                            else:
                                article['full_content'] = None
                                article['content_error'] = content_result['error']
                        else:
                            article['full_content'] = None
                            article['content_error'] = 'No link available'

                return news

            except Exception as e:
                logger.error(f"Error fetching news for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Progressive backoff
                    continue
                return None

        return None

    def fetch_price_and_benchmark(
        self,
        symbol: str,
        benchmark: str = "SPY",
        period: str = "2y",
        max_retries: Optional[int] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
        """
        Fetch stock data and benchmark data in one call.

        Common pattern used across the codebase - fetches stock history,
        benchmark history, and stock info together.

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
            if stock_hist is not None:
                # Calculate alpha vs benchmark
                stock_returns = stock_hist['Close'].pct_change()
                spy_returns = spy_hist['Close'].pct_change()
        """
        max_retries = max_retries or self._default_max_retries

        # Fetch stock data
        stock_hist = self.fetch_history(symbol, period=period, max_retries=max_retries)
        stock_info = self.fetch_info(symbol, max_retries=max_retries)

        if stock_hist is None or stock_hist.empty:
            logger.warning(f"Failed to fetch stock data for {symbol}")
            return None, None, None

        # Fetch benchmark data
        benchmark_hist = self.fetch_history(benchmark, period=period, max_retries=max_retries)

        if benchmark_hist is None or benchmark_hist.empty:
            logger.warning(f"Failed to fetch benchmark data for {benchmark}")
            return stock_hist, None, stock_info

        # Align dates (timezone-agnostic matching)
        try:
            stock_date_strs = [ts.strftime('%Y-%m-%d') for ts in stock_hist.index]
            bench_date_strs = [ts.strftime('%Y-%m-%d') for ts in benchmark_hist.index]

            common_date_strs = set(stock_date_strs) & set(bench_date_strs)

            if len(common_date_strs) > 0:
                stock_mask = pd.Series(stock_date_strs).isin(common_date_strs).values
                bench_mask = pd.Series(bench_date_strs).isin(common_date_strs).values

                stock_hist = stock_hist[stock_mask]
                benchmark_hist = benchmark_hist[bench_mask]

                logger.debug(
                    f"Date alignment: {len(common_date_strs)} common trading days "
                    f"(stock: {len(stock_hist)}, benchmark: {len(benchmark_hist)})"
                )
            else:
                logger.warning(f"No common trading dates between {symbol} and {benchmark}")
                benchmark_hist = None
        except Exception as e:
            logger.error(f"Error aligning dates: {e}")
            benchmark_hist = None

        return stock_hist, benchmark_hist, stock_info

    def bulk_download(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d",
        threads: bool = True,
        group_by: str = 'ticker',
        auto_adjust: bool = False,
        progress: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Bulk download multiple tickers using yfinance.download().

        More efficient than fetching tickers individually when you need
        many tickers at once.

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
            # Download multiple stocks
            data = client.bulk_download(["AAPL", "MSFT", "GOOGL"], period="1y")

            # Access individual stock data
            aapl_data = data["AAPL"]
        """
        try:
            logger.info(f"Bulk downloading {len(symbols)} symbols")

            # Apply rate limiting for bulk download
            self._apply_rate_limit("bulk_download")

            if start is not None:
                data = yf.download(
                    tickers=symbols,
                    start=start,
                    end=end,
                    interval=interval,
                    threads=threads,
                    group_by=group_by,
                    auto_adjust=auto_adjust,
                    progress=progress
                )
            else:
                data = yf.download(
                    tickers=symbols,
                    period=period,
                    interval=interval,
                    threads=threads,
                    group_by=group_by,
                    auto_adjust=auto_adjust,
                    progress=progress
                )

            if data is None or (hasattr(data, 'empty') and data.empty):
                logger.error(f"Bulk download returned no data for {len(symbols)} symbols")
                return None

            logger.info(f"Successfully downloaded data for {len(symbols)} symbols")
            return data

        except Exception as e:
            logger.error(f"Error in bulk download: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size, capacity, and hit/miss info

        Example:
            stats = client.get_cache_stats()
            print(f"Cache size: {stats['size']}/{stats['capacity']}")
        """
        return {
            'size': self._ticker_cache.size(),
            'capacity': self._ticker_cache.capacity,
            'rate_limit_delay': self._rate_limit_delay,
            'default_max_retries': self._default_max_retries
        }

    def clear_cache(self) -> None:
        """
        Clear all cached Ticker objects.

        Useful for forcing fresh data or freeing memory.

        Example:
            client.clear_cache()
        """
        self._ticker_cache.clear()
        logger.info("Ticker cache cleared")


# Convenience function for backward compatibility
def get_yfinance_client(**kwargs) -> YFinanceClient:
    """
    Get the singleton YFinanceClient instance.

    Convenience wrapper around YFinanceClient.get_instance().

    Args:
        **kwargs: Configuration options (only used on first call)

    Returns:
        Singleton YFinanceClient instance

    Example:
        from src.yfinance import get_yfinance_client

        client = get_yfinance_client()
        info = client.fetch_info("AAPL")
    """
    return YFinanceClient.get_instance(**kwargs)


if __name__ == "__main__":
    """Test the YFinanceClient"""

    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("YFINANCE CLIENT - TEST")
    print("=" * 80)

    # Initialize client
    print("\nInitializing client...")
    client = YFinanceClient.get_instance(cache_size=100, rate_limit_delay=0.2)

    # Test fetch_info
    print("\n" + "=" * 80)
    print("TEST 1: Fetch Info")
    print("=" * 80)
    info = client.fetch_info("AAPL")
    if info:
        print(f"✓ AAPL Info:")
        print(f"  Company: {info.get('longName')}")
        print(f"  Sector: {info.get('sector')}")
        print(f"  Price: ${info.get('currentPrice')}")

    # Test fetch_history
    print("\n" + "=" * 80)
    print("TEST 2: Fetch History")
    print("=" * 80)
    hist = client.fetch_history("AAPL", period="1mo")
    if hist is not None:
        print(f"✓ AAPL History: {len(hist)} days")
        print(f"  Date range: {hist.index[0].date()} to {hist.index[-1].date()}")

    # Test fetch_price_and_benchmark
    print("\n" + "=" * 80)
    print("TEST 3: Fetch Price and Benchmark")
    print("=" * 80)
    stock_hist, spy_hist, info = client.fetch_price_and_benchmark("MSFT", period="1mo")
    if stock_hist is not None:
        print(f"✓ MSFT + SPY: {len(stock_hist)} days")
        print(f"  MSFT return: {((stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[0]) - 1) * 100:.2f}%")
        if spy_hist is not None:
            print(f"  SPY return: {((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1) * 100:.2f}%")

    # Test caching
    print("\n" + "=" * 80)
    print("TEST 4: Cache Performance")
    print("=" * 80)
    print("Fetching AAPL again (should use cache)...")
    start_time = time.time()
    info2 = client.fetch_info("AAPL")
    elapsed = time.time() - start_time
    print(f"✓ Cached fetch completed in {elapsed:.3f}s")

    # Test bulk download
    print("\n" + "=" * 80)
    print("TEST 5: Bulk Download")
    print("=" * 80)
    data = client.bulk_download(["AAPL", "MSFT", "GOOGL"], period="1mo")
    if data is not None:
        print(f"✓ Bulk download successful")
        print(f"  Shape: {data.shape}")

    # Cache stats
    print("\n" + "=" * 80)
    print("CACHE STATISTICS")
    print("=" * 80)
    stats = client.get_cache_stats()
    print(f"Cache size: {stats['size']}/{stats['capacity']}")
    print(f"Rate limit delay: {stats['rate_limit_delay']}s")
    print(f"Default max retries: {stats['default_max_retries']}")

    print("\n" + "=" * 80)
    print("✓ All tests completed")
    print("=" * 80)
