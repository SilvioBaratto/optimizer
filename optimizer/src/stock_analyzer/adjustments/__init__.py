"""
Adjustments Module
==================

Provides macro and risk adjustments for signal calculation:
- Macro adjustments (PMI, unemployment, regime)
- Risk adjustments and distress detection
"""

from .macro import apply_macro_adjustments
from .risk import calculate_risk_factors, calculate_confidence

__all__ = [
    'apply_macro_adjustments',
    'calculate_risk_factors',
    'calculate_confidence',
]
