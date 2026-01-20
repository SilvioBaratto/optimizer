"""
Data Coverage Filter - Filter stocks by institutional data completeness.

Single Responsibility: Validates all required data categories are present.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from optimizer.config.universe_builder_config import (
    UniverseBuilderConfig,
    InstitutionalFieldSpec,
)


@dataclass
class DataCoverageFilter:
    """
    Filter stocks by institutional data coverage.

    Requires 100% coverage of required data categories:
    - market_cap: Market capitalization
    - price: Current price
    - volume: Average daily volume
    - shares_outstanding: Total shares
    - sector_industry: GICS classification
    - exchange: Primary exchange
    - financial_ratios: P/E, P/B
    - profitability: ROE, ROA, margins
    - debt_metrics: D/E ratio, total debt
    - 52week_range: 52-week high/low

    Each category has specific field requirements:
    - all_required: All fields must be present
    - at_least_one: At least one field must be present
    - default: All fields must be present

    Attributes:
        config: UniverseBuilderConfig with institutional_fields
    """

    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        """Human-readable filter name."""
        return "DataCoverageFilter"

    def filter(self, data: Dict[str, Any], yf_ticker: str) -> Tuple[bool, str]:
        """
        Apply data coverage filter to instrument data.

        Args:
            data: yfinance info dictionary with institutional fields
            yf_ticker: Yahoo Finance ticker symbol (unused but required by protocol)

        Returns:
            Tuple of (passed, reason):
            - passed: True if 100% of required categories are satisfied
            - reason: Human-readable explanation (missing categories if failed)
        """
        if not data:
            return False, "No data available"

        missing_categories: List[str] = []

        # Check each required category
        for category_name, spec in self.config.institutional_fields.items():
            if not spec.required:
                continue  # Skip optional categories

            passed, _ = self._check_category(data, category_name, spec)
            if not passed:
                missing_categories.append(category_name)

        if missing_categories:
            missing_str = ", ".join(missing_categories[:3])
            if len(missing_categories) > 3:
                missing_str += f" +{len(missing_categories) - 3} more"
            return False, f"Incomplete data: missing {missing_str}"

        required_count = sum(
            1 for spec in self.config.institutional_fields.values() if spec.required
        )
        return True, f"100% coverage ({required_count} required categories)"

    def _check_category(
        self, data: Dict[str, Any], category_name: str, spec: InstitutionalFieldSpec
    ) -> Tuple[bool, List[str]]:
        """
        Check if a data category meets requirements.

        Args:
            data: Stock data dictionary
            category_name: Name of the category
            spec: Category specification

        Returns:
            (passed, missing_fields)
        """
        available = []
        missing = []

        for field in spec.fields:
            if field in data and data[field] is not None:
                available.append(field)
            else:
                missing.append(field)

        # Determine if category passed
        if spec.all_required:
            # All fields must be present
            passed = len(missing) == 0
        elif spec.at_least_one:
            # At least one field must be present
            passed = len(available) > 0
        else:
            # Default: all fields must be present
            passed = len(missing) == 0

        return passed, missing
