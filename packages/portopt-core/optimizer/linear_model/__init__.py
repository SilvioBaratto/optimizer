"""Cross-sectional linear regression for factor IC and risk regressions.

Vectorised cross-sectional weighted-least-squares over panel data (one
independent regression per period), plus an adapter that fits an arbitrary
scikit-learn regressor per period.

Input shapes: ``X: (T, N, K)``, ``y: (T, N)``, ``cs_weights: (T, N)`` (the
weight kwarg is ``cs_weights``, not ``sample_weight``). Zero-weight pairs are
excluded from the fit and may contain NaN.
"""

from optimizer.linear_model._config import (
    CSLinearRegressionConfig,
    CSLinearRegressorWrapperConfig,
)
from optimizer.linear_model._factory import (
    build_cs_linear_regression,
    build_cs_linear_regressor_wrapper,
)

__all__ = [
    "CSLinearRegressionConfig",
    "CSLinearRegressorWrapperConfig",
    "build_cs_linear_regression",
    "build_cs_linear_regressor_wrapper",
]
