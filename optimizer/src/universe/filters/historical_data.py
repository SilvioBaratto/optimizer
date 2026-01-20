"""
Historical Data Filter - Filter stocks by historical data availability.

Single Responsibility: Validates sufficient historical data for analysis.

Note: This module is intentionally silent (no logging/print).
All user-facing output is handled by the CLI layer.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

from optimizer.config.universe_builder_config import UniverseBuilderConfig
from optimizer.src.yfinance import YFinanceClient


@dataclass
class HistoricalDataFilter:
    """
    Filter stocks by historical data availability.

    Requires sufficient historical data (5 years via period='5y')
    for institutional signal generation:
    - Long-term alpha/beta estimation
    - Multi-year momentum patterns
    - Secular trend analysis
    - Robust statistical significance testing

    Uses a sanity check minimum (default ~3 years = 750 trading days)
    to ensure yfinance returned reasonable data.

    Attributes:
        config: UniverseBuilderConfig with min_trading_days
        _yf_client: Optional YFinanceClient (created lazily if not provided)
    """

    config: UniverseBuilderConfig
    _yf_client: Optional[YFinanceClient] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Human-readable filter name."""
        return "HistoricalDataFilter"

    @property
    def yf_client(self) -> YFinanceClient:
        """Get or create YFinanceClient instance."""
        if self._yf_client is None:
            self._yf_client = YFinanceClient.get_instance()
        return self._yf_client

    def filter(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply historical data filter to instrument.

        Args:
            data: yfinance info dictionary (unused but required by protocol)
            yf_ticker: Yahoo Finance ticker symbol for history lookup

        Returns:
            Tuple of (passed, reason):
            - passed: True if historical data meets requirements
            - reason: Human-readable explanation with days available
        """
        # data parameter unused but required by protocol
        _ = data

        has_coverage, days_available = self._check_historical_data(yf_ticker)

        if not has_coverage:
            years = days_available / 252 if days_available > 0 else 0
            return (
                False,
                f"Insufficient history: {days_available} days ({years:.1f}y) from period='5y'",
            )

        years = days_available / 252
        return True, f"Historical data: {days_available} days ({years:.1f}y)"

    def _check_historical_data(
        self, yf_ticker: str, max_retries: int = 3
    ) -> Tuple[bool, int]:
        """
        Check if stock has sufficient historical data using yfinance period='5y'.

        Trust yfinance's definition of "5 years" and just validate we got reasonable data.
        No arbitrary day count thresholds - if yfinance returns data for '5y', accept it.

        Args:
            yf_ticker: Yahoo Finance ticker symbol
            max_retries: Maximum retry attempts for rate limiting

        Returns:
            (has_coverage: bool, days_available: int)
        """
        for attempt in range(max_retries):
            try:
                # Fetch 5 years of historical data
                hist = self.yf_client.fetch_history(
                    yf_ticker,
                    period="5y",
                    max_retries=1,  # We handle retries ourselves
                    min_rows=1,  # Accept any amount initially
                )

                if hist is None or hist.empty:
                    return False, 0

                days_available = len(hist)

                # Sanity check: Just ensure we got a reasonable amount of data
                # (at least ~3 years worth = 750 trading days)
                # If yfinance returned data for period="5y", trust it's approximately correct
                has_coverage = days_available >= self.config.min_trading_days

                return has_coverage, days_available

            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting and timeouts
                if any(
                    x in error_str
                    for x in ["rate limit", "too many requests", "timeout", "timed out"]
                ):
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 10
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, 0
                else:
                    # Other errors
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        return False, 0

        return False, 0
