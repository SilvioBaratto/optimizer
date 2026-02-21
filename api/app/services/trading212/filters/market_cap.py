from dataclasses import dataclass
from typing import Any

from app.services.trading212.config import UniverseBuilderConfig


@dataclass
class MarketCapFilter:
    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        return "MarketCapFilter"

    def filter(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
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
        if value >= 1_000_000_000:
            return f"${value / 1e9:.2f}B"
        else:
            return f"${value / 1e6:.1f}M"
