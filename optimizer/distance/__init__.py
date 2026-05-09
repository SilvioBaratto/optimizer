"""Distance estimator selection.

All distance estimators are sklearn-compatible — they expose ``.fit(X)``
and store a square ``distance_`` matrix.
"""

from optimizer.distance._config import DistanceConfig, DistanceEstimatorType
from optimizer.distance._factory import build_distance

__all__ = [
    "DistanceConfig",
    "DistanceEstimatorType",
    "build_distance",
]
