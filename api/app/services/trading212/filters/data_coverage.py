from dataclasses import dataclass
from typing import Any

from app.services.trading212.config import (
    InstitutionalFieldSpec,
    UniverseBuilderConfig,
)


@dataclass
class DataCoverageFilter:
    config: UniverseBuilderConfig

    @property
    def name(self) -> str:
        return "DataCoverageFilter"

    def filter(self, data: dict[str, Any], yf_ticker: str) -> tuple[bool, str]:
        if not data:
            return False, "No data available"

        missing_categories: list[str] = []

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
        self, data: dict[str, Any], category_name: str, spec: InstitutionalFieldSpec
    ) -> tuple[bool, list[str]]:
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
