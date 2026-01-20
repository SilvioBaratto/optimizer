"""
Portfolio Configuration - Externalized settings for portfolio construction.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass(frozen=True)
class PortfolioConfig:
    """
    Configuration for portfolio construction.

    All parameters for portfolio building and constraints are externalized here.

    Attributes:
        # Portfolio size
        target_positions: Target number of positions
        min_positions: Minimum positions (abort if below)
        max_positions: Maximum positions allowed

        # Weight constraints
        max_position_weight: Maximum weight per position
        min_position_weight: Minimum weight per position
        max_sector_weight: Maximum weight per sector
        max_country_weight: Maximum weight per country

        # Diversification
        min_sectors: Minimum number of sectors
        max_correlation: Maximum pairwise correlation
        max_cluster_size: Maximum stocks per correlation cluster

        # Price constraints (Trading212 specific)
        max_price: Maximum stock price (affordability filter)
        min_budget: Minimum investment budget

        # Trading212 constraints
        max_open_quantity: Maximum position size (Trading212 limit)

        # Selection tiers
        tier1_count: Number of highest conviction positions
        tier2_count: Number of medium conviction positions
        tier1_weight: Base weight for tier 1 positions
        tier2_weight: Base weight for tier 2 positions
    """

    # Portfolio size
    target_positions: int = 15
    min_positions: int = 10
    max_positions: int = 20

    # Weight constraints
    max_position_weight: float = 0.10  # 10%
    min_position_weight: float = 0.02  # 2%
    max_sector_weight: float = 0.15  # 15%
    max_country_weight: float = 0.30  # 30%

    # Diversification
    min_sectors: int = 5
    max_correlation: float = 0.75
    max_cluster_size: int = 3

    # Price constraints
    max_price: float = 75.0  # $75 max for affordability
    min_budget: float = 2000.0  # $2000 minimum budget

    # Trading212 constraints
    max_open_quantity: Optional[float] = None

    # Selection tiers
    tier1_count: int = 5
    tier2_count: int = 7
    tier1_weight: float = 0.08  # 8%
    tier2_weight: float = 0.055  # 5.5%

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "target_positions": self.target_positions,
            "min_positions": self.min_positions,
            "max_positions": self.max_positions,
            "max_position_weight": self.max_position_weight,
            "min_position_weight": self.min_position_weight,
            "max_sector_weight": self.max_sector_weight,
            "max_country_weight": self.max_country_weight,
            "min_sectors": self.min_sectors,
            "max_correlation": self.max_correlation,
            "max_cluster_size": self.max_cluster_size,
            "max_price": self.max_price,
            "min_budget": self.min_budget,
            "max_open_quantity": self.max_open_quantity,
            "tier1_count": self.tier1_count,
            "tier2_count": self.tier2_count,
            "tier1_weight": self.tier1_weight,
            "tier2_weight": self.tier2_weight,
        }

    @classmethod
    def from_env(cls) -> "PortfolioConfig":
        """Create configuration from environment variables."""

        def get_int(name: str, default: int) -> int:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def get_float(name: str, default: Optional[float]) -> Optional[float]:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        return cls(
            target_positions=get_int("PORTFOLIO_TARGET_POSITIONS", 15),
            min_positions=get_int("PORTFOLIO_MIN_POSITIONS", 10),
            max_positions=get_int("PORTFOLIO_MAX_POSITIONS", 20),
            max_position_weight=get_float("PORTFOLIO_MAX_POSITION_WEIGHT", 0.10) or 0.10,
            min_position_weight=get_float("PORTFOLIO_MIN_POSITION_WEIGHT", 0.02) or 0.02,
            max_sector_weight=get_float("PORTFOLIO_MAX_SECTOR_WEIGHT", 0.15) or 0.15,
            max_country_weight=get_float("PORTFOLIO_MAX_COUNTRY_WEIGHT", 0.30) or 0.30,
            min_sectors=get_int("PORTFOLIO_MIN_SECTORS", 5),
            max_correlation=get_float("PORTFOLIO_MAX_CORRELATION", 0.75) or 0.75,
            max_cluster_size=get_int("PORTFOLIO_MAX_CLUSTER_SIZE", 3),
            max_price=get_float("PORTFOLIO_MAX_PRICE", 75.0) or 75.0,
            min_budget=get_float("PORTFOLIO_MIN_BUDGET", 2000.0) or 2000.0,
            max_open_quantity=get_float("PORTFOLIO_MAX_OPEN_QUANTITY", None),
            tier1_count=get_int("PORTFOLIO_TIER1_COUNT", 5),
            tier2_count=get_int("PORTFOLIO_TIER2_COUNT", 7),
            tier1_weight=get_float("PORTFOLIO_TIER1_WEIGHT", 0.08) or 0.08,
            tier2_weight=get_float("PORTFOLIO_TIER2_WEIGHT", 0.055) or 0.055,
        )

    @classmethod
    def concentrated(cls) -> "PortfolioConfig":
        """Create concentrated portfolio configuration (fewer positions, higher conviction)."""
        return cls(
            target_positions=10,
            min_positions=8,
            max_positions=12,
            max_position_weight=0.15,
            min_position_weight=0.05,
            max_sector_weight=0.20,
            max_country_weight=0.40,
            min_sectors=4,
            max_correlation=0.70,
            max_cluster_size=2,
            tier1_count=4,
            tier2_count=4,
            tier1_weight=0.12,
            tier2_weight=0.08,
        )

    @classmethod
    def diversified(cls) -> "PortfolioConfig":
        """Create diversified portfolio configuration (more positions, lower concentration)."""
        return cls(
            target_positions=25,
            min_positions=20,
            max_positions=30,
            max_position_weight=0.06,
            min_position_weight=0.02,
            max_sector_weight=0.12,
            max_country_weight=0.25,
            min_sectors=8,
            max_correlation=0.65,
            max_cluster_size=4,
            tier1_count=8,
            tier2_count=10,
            tier1_weight=0.05,
            tier2_weight=0.035,
        )


@dataclass(frozen=True)
class SectorConstraints:
    """
    Sector-specific weight constraints.

    Allows different max weights per sector based on regime or preferences.
    """

    default_max_weight: float = 0.15
    sector_overrides: Dict[str, float] = field(default_factory=dict)

    def get_max_weight(self, sector: str) -> float:
        """Get maximum weight for a sector."""
        return self.sector_overrides.get(sector, self.default_max_weight)

    @classmethod
    def for_regime(cls, regime: str) -> "SectorConstraints":
        """Create regime-appropriate sector constraints."""
        if regime == "recession":
            return cls(
                default_max_weight=0.15,
                sector_overrides={
                    "Consumer Staples": 0.20,
                    "Healthcare": 0.20,
                    "Utilities": 0.18,
                    "Technology": 0.10,
                    "Consumer Discretionary": 0.08,
                }
            )
        elif regime == "early_cycle":
            return cls(
                default_max_weight=0.15,
                sector_overrides={
                    "Consumer Discretionary": 0.20,
                    "Technology": 0.18,
                    "Industrials": 0.18,
                    "Utilities": 0.08,
                    "Consumer Staples": 0.10,
                }
            )
        else:
            return cls()
