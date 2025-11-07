"""
Classification Module
=====================

Provides signal classification based on percentile thresholds and scoring:
- Distribution tracking and percentile-based classification
- Upside/downside potential calculation
- Data quality scoring
"""

from .distribution import (
    SignalClassifier,
    load_saved_distribution,
    save_distribution_snapshot,
)
from .scoring import (
    calculate_upside_potential,
    calculate_downside_risk,
    calculate_data_quality,
    generate_analysis_notes,
)

__all__ = [
    'SignalClassifier',
    'load_saved_distribution',
    'save_distribution_snapshot',
    'calculate_upside_potential',
    'calculate_downside_risk',
    'calculate_data_quality',
    'generate_analysis_notes',
]
