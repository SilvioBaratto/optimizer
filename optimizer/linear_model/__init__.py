"""Cross-sectional linear regression for factor IC and risk regressions.

Input shapes: ``X: (T, N, K)``, ``y: (T, N)``, ``sample_weight: (T, N)``.
Zero-weight pairs are excluded and may contain NaN.
"""

from optimizer.linear_model._config import CSLinearRegressionConfig
from optimizer.linear_model._factory import build_cs_linear_regression

__all__ = [
    "CSLinearRegressionConfig",
    "build_cs_linear_regression",
]
