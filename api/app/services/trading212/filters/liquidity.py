from dataclasses import dataclass
from typing import Any

from app.services.trading212.config import UniverseBuilderConfig


@dataclass
class LiquidityFilter:
    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        return "LiquidityFilter"

    def filter(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
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
        return segment.replace("_", "-").title()
