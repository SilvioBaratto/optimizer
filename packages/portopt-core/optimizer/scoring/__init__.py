"""Performance scoring for model selection and hyperparameter tuning.

Wraps skfolio ratio / performance / risk measures and custom scoring
functions into callables compatible with sklearn cross-validation, plus a
helper to resolve a config to the bare skfolio measure required by the
online model-selection utilities.
"""

from optimizer.scoring._config import PerfMeasureType, ScorerConfig
from optimizer.scoring._factory import build_online_measure, build_scorer

__all__ = [
    "PerfMeasureType",
    "ScorerConfig",
    "build_online_measure",
    "build_scorer",
]
