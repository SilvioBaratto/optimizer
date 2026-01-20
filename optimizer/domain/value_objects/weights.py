"""
Portfolio Weights Value Object - Immutable weight allocation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np


@dataclass(frozen=True)
class PortfolioWeights:
    """
    Immutable portfolio weight allocation.

    Ensures weights are valid (sum to 1, non-negative for long-only).

    Attributes:
        weights: Dictionary mapping ticker -> weight
        allow_short: Whether to allow negative weights
    """

    weights: Dict[str, float] = field(default_factory=dict)
    allow_short: bool = False

    def __post_init__(self):
        """Validate weights after initialization."""
        if not self.weights:
            return

        # Validate non-negative for long-only
        if not self.allow_short:
            negative = {t: w for t, w in self.weights.items() if w < 0}
            if negative:
                raise ValueError(f"Long-only portfolio has negative weights: {negative}")

        # Validate sum (allow small tolerance)
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

    def __getitem__(self, ticker: str) -> float:
        """Get weight for a ticker."""
        return self.weights.get(ticker, 0.0)

    def __contains__(self, ticker: str) -> bool:
        """Check if ticker has a weight."""
        return ticker in self.weights

    def __len__(self) -> int:
        """Number of positions."""
        return len(self.weights)

    def __iter__(self) -> Iterator[str]:
        """Iterate over tickers."""
        return iter(self.weights)

    def items(self) -> Iterator[Tuple[str, float]]:
        """Iterate over (ticker, weight) pairs."""
        return iter(self.weights.items())

    @property
    def tickers(self) -> List[str]:
        """Get list of tickers."""
        return list(self.weights.keys())

    @property
    def values(self) -> List[float]:
        """Get list of weights in ticker order."""
        return list(self.weights.values())

    @property
    def total_weight(self) -> float:
        """Calculate total weight."""
        return sum(self.weights.values())

    @property
    def max_weight(self) -> float:
        """Get maximum weight."""
        return max(self.weights.values()) if self.weights else 0.0

    @property
    def min_weight(self) -> float:
        """Get minimum weight."""
        return min(self.weights.values()) if self.weights else 0.0

    @property
    def effective_n(self) -> float:
        """
        Calculate effective number of positions (diversification measure).

        Uses the Herfindahl-Hirschman Index (HHI) inverse.
        A portfolio with N equal weights has effective_n = N.
        More concentrated portfolios have lower effective_n.
        """
        if not self.weights:
            return 0.0
        hhi = sum(w ** 2 for w in self.weights.values())
        return 1.0 / hhi if hhi > 0 else 0.0

    def to_numpy(self, tickers: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert to numpy array.

        Args:
            tickers: Optional ordered list of tickers. If provided, array
                    will be in this order with 0 for missing tickers.

        Returns:
            1D numpy array of weights
        """
        if tickers is None:
            return np.array(list(self.weights.values()))

        return np.array([self.weights.get(t, 0.0) for t in tickers])

    def to_dict(self) -> Dict[str, float]:
        """Return weights as dictionary."""
        return dict(self.weights)

    def get_sector_weights(
        self,
        ticker_to_sector: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate weights by sector.

        Args:
            ticker_to_sector: Mapping of ticker -> sector

        Returns:
            Dictionary of sector -> total weight
        """
        sector_weights: Dict[str, float] = {}
        for ticker, weight in self.weights.items():
            sector = ticker_to_sector.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        return sector_weights

    @classmethod
    def from_equal_weights(cls, tickers: List[str]) -> "PortfolioWeights":
        """Create equal-weighted portfolio."""
        if not tickers:
            return cls({})
        weight = 1.0 / len(tickers)
        return cls({t: weight for t in tickers})

    @classmethod
    def from_numpy(
        cls,
        weights: np.ndarray,
        tickers: List[str],
        allow_short: bool = False
    ) -> "PortfolioWeights":
        """
        Create from numpy array.

        Args:
            weights: 1D array of weights
            tickers: Ordered list of tickers matching weights
            allow_short: Whether to allow negative weights

        Returns:
            PortfolioWeights object
        """
        if len(weights) != len(tickers):
            raise ValueError(f"Length mismatch: {len(weights)} weights, {len(tickers)} tickers")

        return cls(
            weights=dict(zip(tickers, weights.tolist())),
            allow_short=allow_short
        )
