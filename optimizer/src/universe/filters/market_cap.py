"""
Market Cap Filter - Filter stocks by market capitalization.

Single Responsibility: Validates market cap meets minimum threshold.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple

from optimizer.config.universe_builder_config import UniverseBuilderConfig


@dataclass
class MarketCapFilter:
    """
    Filter stocks by market capitalization.

    Rejects stocks below the minimum market cap threshold (default $100M).
    This is a fundamental institutional filter to ensure investability.

    Attributes:
        config: UniverseBuilderConfig with min_market_cap threshold
    """

    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        """Human-readable filter name."""
        return "MarketCapFilter"

    def filter(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply market cap filter to instrument data.

        Args:
            data: yfinance info dictionary with 'marketCap' field
            yf_ticker: Yahoo Finance ticker symbol (unused but required by protocol)

        Returns:
            Tuple of (passed, reason):
            - passed: True if market cap >= min_market_cap
            - reason: Human-readable explanation
        """
        if not data:
            return False, "No data available"

        market_cap = data.get("marketCap")

        if market_cap is None:
            return False, "No market cap data"

        if market_cap < self.config.min_market_cap:
            mcap_str = self._format_market_cap(market_cap)
            min_str = self._format_market_cap(self.config.min_market_cap)
            return False, f"Market cap {mcap_str} < {min_str}"

        mcap_str = self._format_market_cap(market_cap)
        return True, f"Market cap {mcap_str}"

    def _format_market_cap(self, value: float) -> str:
        """Format market cap for human-readable display."""
        if value >= 1_000_000_000:
            return f"${value / 1e9:.2f}B"
        else:
            return f"${value / 1e6:.1f}M"
