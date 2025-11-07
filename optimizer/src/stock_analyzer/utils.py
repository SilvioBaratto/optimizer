"""
Stock Analyzer Utilities
=========================

Shared utility functions used across the stock_analyzer package.

This module is separate from pipeline.utils to avoid circular imports.
"""

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with default fallback.

    Handles special cases:
    - None → default
    - 'Infinity', 'inf', '-inf' → default
    - NaN → default
    - Invalid strings → default

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default

    Examples:
        >>> safe_float(3.14)
        3.14
        >>> safe_float('Infinity')
        0.0
        >>> safe_float(None, -1.0)
        -1.0
        >>> safe_float('invalid')
        0.0
    """
    if value is None:
        return default
    try:
        result = float(value)
        # Check if result is finite (not inf or nan)
        import math
        if not math.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int with default fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Int value or default

    Examples:
        >>> safe_int(42)
        42
        >>> safe_int('123')
        123
        >>> safe_int(None, -1)
        -1
        >>> safe_int('invalid')
        0
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
