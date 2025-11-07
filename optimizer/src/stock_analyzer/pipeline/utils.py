"""
Utility Functions
=================

Helper functions for signal analysis pipeline including:
- Enum mappings (Signal types, confidence levels, risk levels)
- Safe type conversions (floats, numbers)
- Data validation helpers
"""

from typing import Optional
import logging
from app.models.stock_signals import SignalEnum, ConfidenceLevelEnum, RiskLevelEnum
from baml_client.types import SignalType, ConfidenceLevel

# Import safe_float and safe_int from parent utils to avoid circular imports
from src.stock_analyzer.utils import safe_float, safe_int  # noqa: F401

logger = logging.getLogger(__name__)


def map_signal_type_to_enum(signal_type: SignalType) -> SignalEnum:
    """
    Map BAML SignalType to database SignalEnum.

    Args:
        signal_type: BAML SignalType enum

    Returns:
        SignalEnum for database storage
    """
    signal_type_mapping = {
        SignalType.LARGE_GAIN: SignalEnum.LARGE_GAIN,
        SignalType.SMALL_GAIN: SignalEnum.SMALL_GAIN,
        SignalType.NEUTRAL: SignalEnum.NEUTRAL,
        SignalType.SMALL_DECLINE: SignalEnum.SMALL_DECLINE,
        SignalType.LARGE_DECLINE: SignalEnum.LARGE_DECLINE,
    }
    return signal_type_mapping[signal_type]


def map_confidence_level_to_enum(confidence_level: ConfidenceLevel) -> ConfidenceLevelEnum:
    """
    Map BAML ConfidenceLevel to database ConfidenceLevelEnum.

    Args:
        confidence_level: BAML ConfidenceLevel enum

    Returns:
        ConfidenceLevelEnum for database storage
    """
    confidence_level_mapping = {
        ConfidenceLevel.LOW: ConfidenceLevelEnum.LOW,
        ConfidenceLevel.MEDIUM: ConfidenceLevelEnum.MEDIUM,
        ConfidenceLevel.HIGH: ConfidenceLevelEnum.HIGH,
    }
    return confidence_level_mapping[confidence_level]


def map_risk_level_to_enum(risk_level_str: Optional[str]) -> Optional[RiskLevelEnum]:
    """
    Convert string risk level to RiskLevelEnum.

    Args:
        risk_level_str: String risk level ("LOW", "MEDIUM", "HIGH", "UNKNOWN", "MINIMAL", "EXTREME")

    Returns:
        RiskLevelEnum or None if input is None
    """
    if risk_level_str is None:
        return None

    risk_level_mapping = {
        "LOW": RiskLevelEnum.LOW,
        "MEDIUM": RiskLevelEnum.MEDIUM,
        "HIGH": RiskLevelEnum.HIGH,
        "UNKNOWN": RiskLevelEnum.UNKNOWN,
        "MINIMAL": RiskLevelEnum.LOW,  # Map MINIMAL to LOW
        "EXTREME": RiskLevelEnum.HIGH,  # Map EXTREME to HIGH
    }

    # Handle both uppercase and mixed case
    return risk_level_mapping.get(risk_level_str.upper(), RiskLevelEnum.UNKNOWN)


def validate_numeric_list(values: list, metric_name: str) -> list:
    """
    Validate and clean numeric list by removing None, NaN, and inf values.

    Args:
        values: List of values to validate
        metric_name: Name of metric (for logging)

    Returns:
        Cleaned list of valid numeric values
    """
    import numpy as np

    if not values:
        return []

    # Remove None values
    values = [v for v in values if v is not None]

    if not values:
        return []

    # Convert to numpy array for NaN/inf filtering
    arr = np.array(values, dtype=float)

    # Filter out NaN and inf
    valid_mask = np.isfinite(arr)
    valid_values = arr[valid_mask].tolist()

    removed = len(values) - len(valid_values)
    if removed > 0:
        logger.debug(
            f"Removed {removed} invalid values from {metric_name} "
            f"(NaN/inf) - {len(valid_values)} valid values remaining"
        )

    return valid_values


# Export all public functions including re-exported utilities
__all__ = [
    'map_signal_type_to_enum',
    'map_confidence_level_to_enum',
    'map_risk_level_to_enum',
    'validate_numeric_list',
    'safe_float',  # Re-exported from parent
    'safe_int',    # Re-exported from parent
]
