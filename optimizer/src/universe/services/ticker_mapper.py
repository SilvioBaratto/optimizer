"""
YFinance Ticker Mapper - Maps Trading212 symbols to yfinance tickers.

Single Responsibility: Discover correct yfinance ticker for T212 symbols.

Note: This module is intentionally silent (no logging/print).
All user-facing output is handled by the CLI layer.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List

from optimizer.config.universe_builder_config import UniverseBuilderConfig
from optimizer.src.universe.cache.ticker_cache import TickerMappingCache
from optimizer.src.yfinance import YFinanceClient


@dataclass
class YFinanceTickerMapper:
    """
    Maps Trading212 symbols to Yahoo Finance tickers.

    Discovery Strategy:
    1. Check cache for previously discovered mapping
    2. Try exchange-specific suffix (e.g., '.DE' for Deutsche Börse)
    3. Try US format (no suffix)
    4. Validate ticker returns valid data

    Handles:
    - Symbol normalization (slashes to dashes)
    - Rate limiting with retry logic
    - Caching of successful mappings

    Attributes:
        config: UniverseBuilderConfig with yahoo_suffix_map
        cache: TickerMappingCache for persistent caching
        _yf_client: Optional YFinanceClient (created lazily if not provided)
    """

    config: UniverseBuilderConfig
    cache: Optional[TickerMappingCache] = None
    _yf_client: Optional[YFinanceClient] = field(default=None, repr=False)
    max_retries: int = 5

    def __post_init__(self):
        """Initialize cache if not provided."""
        if self.cache is None:
            self.cache = TickerMappingCache()

    @property
    def yf_client(self) -> YFinanceClient:
        """Get or create YFinanceClient instance."""
        if self._yf_client is None:
            self._yf_client = YFinanceClient.get_instance()
        return self._yf_client

    def discover(
        self, symbol: str, exchange_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Discover the correct yfinance ticker for a Trading212 symbol.

        Args:
            symbol: Trading212 short name (e.g., 'AAPL', 'VOW3')
            exchange_name: Exchange name for suffix determination

        Returns:
            yfinance ticker string if found (e.g., 'AAPL', 'VOW3.DE')
            None if ticker cannot be discovered
        """
        try:
            # Check cache first
            if exchange_name and self.cache:
                cached = self.cache.get_mapping(symbol, exchange_name)
                if cached:
                    # Verify cached ticker still works
                    if self._verify_ticker(cached):
                        return cached
                    # Cache invalid, continue discovery

            # Yahoo Finance uses dashes instead of slashes for share classes
            clean_symbol = symbol.replace("/", "-")

            # Build list of tickers to try
            ticker_attempts = self._build_ticker_attempts(clean_symbol, exchange_name)

            # Try each ticker
            for attempt_ticker in ticker_attempts:
                if self._verify_ticker(attempt_ticker):
                    # Cache successful mapping
                    if exchange_name and self.cache:
                        self.cache.save_mapping(symbol, exchange_name, attempt_ticker)
                    return attempt_ticker

            return None

        except Exception:
            return None

    def _build_ticker_attempts(
        self, clean_symbol: str, exchange_name: Optional[str]
    ) -> List[str]:
        """
        Build list of tickers to try in order of preference.

        Args:
            clean_symbol: Normalized symbol (slashes replaced with dashes)
            exchange_name: Exchange name for suffix lookup

        Returns:
            List of ticker strings to try
        """
        attempts = []

        # Add exchange-specific suffix first (most likely to be correct)
        if exchange_name:
            suffix = self.config.get_yahoo_suffix(exchange_name)
            if suffix is not None:
                preferred_ticker = clean_symbol + suffix
                attempts.append(preferred_ticker)

        # Add US format (no suffix) - common fallback
        if clean_symbol not in attempts:
            attempts.append(clean_symbol)

        return attempts

    def _verify_ticker(self, ticker: str) -> bool:
        """
        Verify a ticker returns valid data from yfinance.

        Args:
            ticker: yfinance ticker to verify

        Returns:
            True if ticker returns valid info with price data
        """
        for retry in range(self.max_retries):
            try:
                info = self.yf_client.fetch_info(ticker, max_retries=1, min_fields=5)

                if info and len(info) > 5:
                    # Check for price data (confirms valid tradeable ticker)
                    if "currentPrice" in info or "regularMarketPrice" in info:
                        return True

                # No valid data
                return False

            except Exception as e:
                error_str = str(e).lower()

                # Rate limit - progressive backoff
                if any(
                    x in error_str
                    for x in ["rate limit", "too many requests", "timeout", "timed out"]
                ):
                    if retry < self.max_retries - 1:
                        wait_times = [60, 300, 900, 1800, 3600]
                        wait_time = (
                            wait_times[retry] if retry < len(wait_times) else 3600
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        return False

                # Ticker not found errors
                if any(
                    x in error_str
                    for x in ["not found", "404", "invalid crumb", "unauthorized"]
                ):
                    return False

                # Other errors - give up
                return False

        return False

    def fetch_basic_data(
        self, yf_ticker: str, max_retries: int = 3
    ) -> Optional[dict]:
        """
        Fetch comprehensive financial data for filtering.

        Args:
            yf_ticker: Yahoo Finance ticker symbol
            max_retries: Maximum retry attempts

        Returns:
            Dictionary with institutional data fields or None if failed
        """
        for attempt in range(max_retries):
            try:
                info = self.yf_client.fetch_info(
                    yf_ticker, max_retries=1, min_fields=10
                )

                if not info or len(info) < 10:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    return None

                # Extract all institutional data fields
                return {
                    # Market cap and price
                    "marketCap": info.get("marketCap"),
                    "currentPrice": info.get("currentPrice"),
                    "regularMarketPrice": info.get("regularMarketPrice"),
                    # Volume and shares
                    "volume": info.get("volume"),
                    "averageVolume": info.get("averageVolume"),
                    "averageVolume10days": info.get("averageVolume10days"),
                    "sharesOutstanding": info.get("sharesOutstanding"),
                    # Beta and risk
                    "beta": info.get("beta"),
                    # Sector and industry
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    # Exchange
                    "exchange": info.get("exchange"),
                    # Financial ratios
                    "trailingPE": info.get("trailingPE"),
                    "priceToBook": info.get("priceToBook"),
                    "priceToSalesTrailing12Months": info.get(
                        "priceToSalesTrailing12Months"
                    ),
                    # Profitability
                    "returnOnEquity": info.get("returnOnEquity"),
                    "returnOnAssets": info.get("returnOnAssets"),
                    "profitMargins": info.get("profitMargins"),
                    "operatingMargins": info.get("operatingMargins"),
                    "grossMargins": info.get("grossMargins"),
                    # Debt metrics
                    "debtToEquity": info.get("debtToEquity"),
                    "totalDebt": info.get("totalDebt"),
                    "totalAssets": info.get("totalAssets"),
                    "currentRatio": info.get("currentRatio"),
                    # Dividend data
                    "dividendYield": info.get("dividendYield"),
                    "dividendRate": info.get("dividendRate"),
                    # 52-week range
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                    "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                    # Additional metrics
                    "operatingCashflow": info.get("operatingCashflow"),
                    "revenueGrowth": info.get("revenueGrowth"),
                    "earningsGrowth": info.get("earningsGrowth"),
                    "52WeekChange": info.get("52WeekChange"),
                }

            except Exception as e:
                error_str = str(e).lower()

                if any(
                    x in error_str
                    for x in ["rate limit", "too many requests", "timeout", "timed out"]
                ):
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 10
                        time.sleep(wait_time)
                        continue
                    return None
                else:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    return None

        return None
