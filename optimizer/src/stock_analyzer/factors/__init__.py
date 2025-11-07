"""
Factor Calculation Module
=========================

Implements the four-factor institutional framework per Section 5.2:

1. Value Factor (25%): B/P, E/P, FCF/P (Section 5.2.2)
2. Momentum Factor (25%): 12-1 month returns (Section 5.2.5)
3. Quality Factor (25%): ROE, margins, Sharpe (Section 5.2.4)
4. Growth Factor (25%): Revenue/earnings growth (Section 5.2.3)

Z-score standardization relative to market norms (Section 5.2.1).
"""

from .calculators import (
    calculate_value_factor,
    calculate_momentum_factor,
    calculate_quality_factor,
    calculate_growth_factor,
    calculate_ic_weights,
)

__all__ = [
    'calculate_value_factor',
    'calculate_momentum_factor',
    'calculate_quality_factor',
    'calculate_growth_factor',
    'calculate_ic_weights',
]
