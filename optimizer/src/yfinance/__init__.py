"""
YFinance Module - Centralized yfinance Client
==============================================

Provides a singleton YFinanceClient for all yfinance API calls across the codebase.

Key Features:
- Ticker object caching (LRU with configurable size)
- Rate limiting to avoid API throttling
- Retry logic for transient failures
- Thread-safe operations
- Common methods: fetch_info, fetch_history, fetch_news, bulk_download

Usage:
    from src.yfinance import YFinanceClient

    # Get singleton instance
    client = YFinanceClient.get_instance()

    # Fetch data
    info = client.fetch_info("AAPL")
    hist = client.fetch_history("AAPL", period="1y")
    stock_hist, spy_hist, info = client.fetch_price_and_benchmark("AAPL")

    # Get cached Ticker object (for advanced usage)
    ticker = client.get_ticker("AAPL")

Or use convenience function:
    from src.yfinance import get_yfinance_client

    client = get_yfinance_client()

Author: Portfolio Optimization System
"""

from src.yfinance.client import YFinanceClient, get_yfinance_client

__all__ = [
    'YFinanceClient',
    'get_yfinance_client',
]
