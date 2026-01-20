"""
Price Filter - Filter stocks by share price range.

Single Responsibility: Validates price within acceptable range.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple

from optimizer.config.universe_builder_config import UniverseBuilderConfig


@dataclass
class PriceFilter:
    """
    Filter stocks by share price.

    Rejects stocks:
    - Below minimum price (default $5) - penny stock exclusion
    - Above maximum price (default $10,000) - data error detection

    Attributes:
        config: UniverseBuilderConfig with min_price and max_price thresholds
    """

    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        """Human-readable filter name."""
        return "PriceFilter"

    def filter(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply price filter to instrument data.

        Args:
            data: yfinance info dictionary with 'currentPrice' or 'regularMarketPrice'
            yf_ticker: Yahoo Finance ticker symbol (unused but required by protocol)

        Returns:
            Tuple of (passed, reason):
            - passed: True if min_price <= price <= max_price
            - reason: Human-readable explanation
        """
        if not data:
            return False, "No data available"

        # Try both price fields (some stocks use one or the other)
        price = data.get("currentPrice") or data.get("regularMarketPrice")

        if price is None:
            return False, "No current price available"

        if price < self.config.min_price:
            return (
                False,
                f"Price ${price:.2f} < ${self.config.min_price:.0f} (penny stock)",
            )

        if price > self.config.max_price:
            return (
                False,
                f"Price ${price:,.2f} > ${self.config.max_price:,.0f} (data error)",
            )

        return True, f"Price ${price:.2f}"
