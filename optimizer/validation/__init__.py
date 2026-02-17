"""Model selection and cross-validation for portfolio backtesting.

Includes Walk-Forward backtesting, Combinatorial Purged Cross-Validation
(CPCV), and Multiple Randomized Cross-Validation.
"""

from optimizer.validation._config import (
    CPCVConfig,
    MultipleRandomizedCVConfig,
    WalkForwardConfig,
)
from optimizer.validation._factory import (
    build_cpcv,
    build_multiple_randomized_cv,
    build_walk_forward,
    compute_optimal_folds,
    run_cross_val,
)

__all__ = [
    "CPCVConfig",
    "MultipleRandomizedCVConfig",
    "WalkForwardConfig",
    "build_cpcv",
    "build_multiple_randomized_cv",
    "build_walk_forward",
    "compute_optimal_folds",
    "run_cross_val",
]
