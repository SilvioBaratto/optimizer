"""Performance scoring for model selection and hyperparameter tuning.

Wraps skfolio ratio measures and custom scoring functions into
callables compatible with sklearn cross-validation.
"""

from optimizer.scoring._config import ScorerConfig
from optimizer.scoring._factory import build_scorer

__all__ = [
    "ScorerConfig",
    "build_scorer",
]
