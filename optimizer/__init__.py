"""Portfolio optimization library built on skfolio and scikit-learn.

Modules
-------
universe
    Investability screening with hysteresis-based entry/exit
    thresholds for market cap, liquidity, price, and data availability.
factors
    Factor construction, cross-sectional standardization, composite
    scoring, stock selection with buffer zones, macro regime tilts,
    statistical validation (IC, VIF, FDR), and bridge to optimization
    inputs (expected returns, Black-Litterman views, constraints).
preprocessing
    Custom sklearn-compatible transformers for return data cleaning.
pre_selection
    Pipeline assembly and asset pre-selection.
moments
    Moment estimation and prior construction.
distance
    Distance estimator selection (Pearson, Kendall, Spearman,
    Covariance, Distance Correlation, Mutual Information) for
    hierarchical clustering optimizers.
cluster
    Hierarchical clustering wrapper for HRP/HERC/NCO/Schur
    optimizers (linkage methods + max-cluster cap).
uncertainty_set
    Mu and covariance uncertainty-set estimators (empirical and
    stationary-bootstrap variants) for robust mean-risk
    optimization.
linear_model
    Cross-sectional linear regression (CSLinearRegression wrapper)
    for factor IC computation and cross-sectional risk regressions.
views
    View integration frameworks (Black-Litterman, Entropy Pooling,
    Opinion Pooling).
optimization
    Portfolio optimization models (Mean-Risk, regime-blended
    Mean-Risk with factor prior).
synthetic
    Synthetic data generation, vine copula models, and conditional
    stress testing.
validation
    Model selection and cross-validation (Walk-Forward, Combinatorial
    Purged CV, Multiple Randomized CV).
scoring
    Performance scoring for model selection and hyperparameter tuning.
tuning
    Hyperparameter tuning with temporal cross-validation
    (GridSearchCV, RandomizedSearchCV).
online
    Incremental-fit (online) workflows around skfolio's
    ``partial_fit``: ``online_predict``, ``online_score``,
    ``OnlineGridSearch``, ``OnlineRandomizedSearch``. Online
    instances are NOT thread-safe — use one per thread.
rebalancing
    Rebalancing frameworks (calendar-based, threshold-based,
    turnover computation, transaction cost estimation).
pipeline
    End-to-end portfolio orchestration: prices → validated weights.
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

logging.getLogger("optimizer").addHandler(logging.NullHandler())

from optimizer.exceptions import (
    ConfigurationError,
    ConvergenceError,
    DataError,
    OptimizationError,
    OptimizerError,
    ValidationError,
)

__all__ = [
    "ConfigurationError",
    "ConvergenceError",
    "DataError",
    "OptimizationError",
    "OptimizerError",
    "ValidationError",
]

try:
    __version__ = _pkg_version("portopt")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
