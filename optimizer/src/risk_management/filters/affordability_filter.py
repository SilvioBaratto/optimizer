"""
Affordability Filter - Filter stocks by price constraints.
"""

from typing import Tuple, Optional, Dict, Any

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.src.risk_management.filters.protocol import StockFilterImpl


class AffordabilityFilter(StockFilterImpl):
    """
    Filter stocks based on price affordability.
    """

    def __init__(
        self,
        max_price: float = 75.0,
        min_price: float = 5.0,
    ):
        """
        Initialize affordability filter.
        """
        super().__init__()
        self._max_price = max_price
        self._min_price = min_price

    @property
    def name(self) -> str:
        """Filter name."""
        return "AffordabilityFilter"

    @property
    def max_price(self) -> float:
        """Maximum allowed price."""
        return self._max_price

    @property
    def min_price(self) -> float:
        """Minimum allowed price."""
        return self._min_price

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Check if signal passes price constraints.

        Args:
            signal: StockSignalDTO to evaluate

        Returns:
            Tuple of (passes, reason)
        """
        if signal.close_price is None:
            return False, "missing_price"

        if signal.close_price > self._max_price:
            return False, f"price_too_high(${signal.close_price:.2f}>${self._max_price:.2f})"

        if signal.close_price < self._min_price:
            return False, f"penny_stock(${signal.close_price:.2f}<${self._min_price:.2f})"

        return True, None

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of filter configuration."""
        return {
            "name": self.name,
            "max_price": f"${self._max_price:.2f}",
            "min_price": f"${self._min_price:.2f}",
        }

    @classmethod
    def for_budget(cls, budget: float, target_positions: int) -> "AffordabilityFilter":
        """
        Create affordability filter based on budget.
        """
        # Allow some headroom (divide budget by positions * 1.2)
        max_price = budget / (target_positions * 1.2)
        # Cap at reasonable level
        max_price = min(max_price, 150.0)

        return cls(max_price=max_price, min_price=5.0)
