from dataclasses import dataclass
from typing import Any

from app.services.trading212.config import UniverseBuilderConfig


@dataclass
class PriceFilter:
    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        return "PriceFilter"

    def filter(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
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
