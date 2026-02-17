"""Hyperparameter tuning with temporal cross-validation.

Wraps sklearn GridSearchCV and RandomizedSearchCV with temporal
cross-validation defaults that prevent look-ahead bias.
"""

from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig
from optimizer.tuning._factory import (
    build_grid_search_cv,
    build_randomized_search_cv,
)

__all__ = [
    "GridSearchConfig",
    "RandomizedSearchConfig",
    "build_grid_search_cv",
    "build_randomized_search_cv",
]
