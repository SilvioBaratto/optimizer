"""
Liquidity Filter - Filter stocks by average daily volume.

Single Responsibility: Validates liquidity meets market cap-adjusted thresholds.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple

from optimizer.config.universe_builder_config import UniverseBuilderConfig


@dataclass
class LiquidityFilter:
    """
    Filter stocks by liquidity (average daily volume).

    Uses market cap-adjusted thresholds:
    - Large-cap ($10B+): $10M ADV, 500K shares
    - Mid-cap ($2B-$10B): $5M ADV, 250K shares
    - Small-cap ($100M-$2B): $1M ADV, 100K shares

    Both dollar volume and share volume must meet thresholds.

    Attributes:
        config: UniverseBuilderConfig with liquidity_tiers
    """

    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        """Human-readable filter name."""
        return "LiquidityFilter"

    def filter(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply liquidity filter to instrument data.

        Args:
            data: yfinance info dictionary with:
                - marketCap
                - averageVolume or averageVolume10days
                - currentPrice or regularMarketPrice

            yf_ticker: Yahoo Finance ticker symbol (unused but required by protocol)

        Returns:
            Tuple of (passed, reason):
            - passed: True if ADV meets segment requirements
            - reason: Human-readable explanation
        """
        if not data:
            return False, "No data available"

        # Get market cap to determine segment
        market_cap = data.get("marketCap")
        if market_cap is None:
            return False, "No market cap for liquidity check"

        # Get price for dollar volume calculation
        price = data.get("currentPrice") or data.get("regularMarketPrice")
        if price is None:
            return False, "No price for dollar volume calculation"

        # Get average volume
        avg_volume = data.get("averageVolume") or data.get("averageVolume10days")
        if avg_volume is None:
            return False, "No average volume data"

        # Determine market cap segment and get requirements
        segment = self.config.determine_market_cap_segment(market_cap)
        liquidity_req = self.config.liquidity_tiers[segment]

        # Calculate dollar volume
        avg_dollar_volume = avg_volume * price

        # Check dollar volume
        if avg_dollar_volume < liquidity_req.min_adv_dollars:
            return (
                False,
                f"{self._format_segment(segment)}: ADV ${avg_dollar_volume / 1e6:.1f}M < "
                f"${liquidity_req.min_adv_dollars / 1e6:.0f}M required",
            )

        # Check share volume
        if avg_volume < liquidity_req.min_adv_shares:
            return (
                False,
                f"{self._format_segment(segment)}: ADV {avg_volume:,.0f} shares < "
                f"{liquidity_req.min_adv_shares:,} required",
            )

        return (
            True,
            f"{self._format_segment(segment)}: ADV ${avg_dollar_volume / 1e6:.1f}M, "
            f"{avg_volume:,.0f} shares",
        )

    def _format_segment(self, segment: str) -> str:
        """Format market cap segment for display."""
        return segment.replace("_", "-").title()
