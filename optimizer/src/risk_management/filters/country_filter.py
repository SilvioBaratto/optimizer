"""
Country Filter - Filter stocks based on geographic diversification.
"""

from typing import Tuple, Optional, Dict, Any, List, Set

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.src.risk_management.filters.protocol import StockFilterImpl


class CountryFilter(StockFilterImpl):
    """
    Filter stocks to ensure geographic diversification.
    """

    def __init__(
        self,
        max_per_country: int = 10,
        allowed_countries: Optional[List[str]] = None,
        blocked_countries: Optional[List[str]] = None,
        max_country_weight: float = 0.30,
    ):
        """
        Initialize country filter.

        Args:
            max_per_country: Maximum stocks allowed per country
            allowed_countries: If set, only these countries are allowed
            blocked_countries: These countries are excluded
            max_country_weight: Maximum portfolio weight per country
        """
        super().__init__()
        self._max_per_country = max_per_country
        self._allowed_countries = set(allowed_countries) if allowed_countries else None
        self._blocked_countries = set(blocked_countries or [])
        self._max_country_weight = max_country_weight

        # Stateful tracking
        self._country_counts: Dict[str, int] = {}

    @property
    def name(self) -> str:
        """Filter name."""
        return "CountryFilter"

    def reset(self) -> None:
        """Reset stateful tracking for a new filtering run."""
        self._country_counts.clear()

    def _infer_country(self, signal: StockSignalDTO) -> str:
        """
        Infer country from signal data.
        """
        # First check if country is directly available
        # (would be populated from instrument data)

        # Infer from yfinance ticker suffix
        if signal.yfinance_ticker:
            suffix_to_country = {
                ".L": "UK",
                ".DE": "Germany",
                ".PA": "France",
                ".AS": "Netherlands",
                ".MI": "Italy",
                ".MC": "Spain",
                ".SW": "Switzerland",
                ".TO": "Canada",
                ".AX": "Australia",
                ".HK": "Hong Kong",
                ".T": "Japan",
            }
            for suffix, country in suffix_to_country.items():
                if signal.yfinance_ticker.endswith(suffix):
                    return country

        # Infer from T212 ticker pattern
        if signal.ticker:
            if "_US_" in signal.ticker:
                return "USA"
            if "_UK_" in signal.ticker or "_LSE_" in signal.ticker:
                return "UK"
            if "_DE_" in signal.ticker:
                return "Germany"

        return "USA"  # Default to USA

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Check if signal passes country constraints.

        This is stateful - it tracks which countries have been filled.
        """
        country = self._infer_country(signal)

        # Check blocklist
        if country in self._blocked_countries:
            return False, f"blocked_country({country})"

        # Check allowlist
        if self._allowed_countries is not None and country not in self._allowed_countries:
            return False, f"country_not_allowed({country})"

        # Check country count
        current_count = self._country_counts.get(country, 0)
        if current_count >= self._max_per_country:
            return False, f"country_full({country}:{current_count}/{self._max_per_country})"

        # Signal passes - update tracking
        self._country_counts[country] = current_count + 1

        return True, None

    def filter_batch(
        self,
        signals: List[StockSignalDTO],
    ) -> Tuple[List[StockSignalDTO], Dict[str, int]]:
        """
        Filter batch with automatic reset.
        """
        self.reset()
        return super().filter_batch(signals)

    def get_country_distribution(self) -> Dict[str, int]:
        """Get current country distribution."""
        return dict(self._country_counts)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of filter configuration."""
        return {
            "name": self.name,
            "max_per_country": self._max_per_country,
            "allowed_countries": (
                list(self._allowed_countries) if self._allowed_countries else "all"
            ),
            "blocked_countries": list(self._blocked_countries),
            "max_country_weight": f"{self._max_country_weight:.0%}",
        }


class RegionalFilter(StockFilterImpl):
    """
    Filter stocks based on regional allocation.

    Groups countries into regions and applies regional constraints.
    """

    # Default regional groupings
    REGIONS = {
        "North America": ["USA", "Canada"],
        "Europe": ["UK", "Germany", "France", "Netherlands", "Italy", "Spain", "Switzerland"],
        "Asia Pacific": ["Japan", "Australia", "Hong Kong", "Singapore"],
        "Emerging Markets": ["Brazil", "China", "India", "South Korea", "Taiwan"],
    }

    def __init__(
        self,
        max_per_region: Optional[Dict[str, int]] = None,
        regional_targets: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize regional filter.

        Args:
            max_per_region: Maximum stocks per region
            regional_targets: Target weight per region (0.0-1.0)
        """
        super().__init__()
        self._max_per_region = max_per_region or {
            "North America": 10,
            "Europe": 8,
            "Asia Pacific": 5,
            "Emerging Markets": 3,
        }
        self._regional_targets = regional_targets

        # Build country to region mapping
        self._country_to_region: Dict[str, str] = {}
        for region, countries in self.REGIONS.items():
            for country in countries:
                self._country_to_region[country] = region

        # Stateful tracking
        self._region_counts: Dict[str, int] = {}

    @property
    def name(self) -> str:
        """Filter name."""
        return "RegionalFilter"

    def reset(self) -> None:
        """Reset stateful tracking."""
        self._region_counts.clear()

    def _get_region(self, country: str) -> str:
        """Get region for a country."""
        return self._country_to_region.get(country, "Other")

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """Check if signal passes regional constraints."""
        # This would need country inference similar to CountryFilter
        # For now, return True as placeholder
        return True, None

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of filter configuration."""
        return {
            "name": self.name,
            "max_per_region": self._max_per_region,
            "regional_targets": self._regional_targets,
        }
