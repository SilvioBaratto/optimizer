"""Portfolio optimization library built on skfolio and scikit-learn.

Submodules:

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
    Purged CV, Multiple Randomized CV) and the probabilistic / deflated
    Sharpe ratio.
scoring
    Performance scoring for model selection and hyperparameter tuning.
tuning
    Hyperparameter tuning with temporal cross-validation
    (GridSearchCV, RandomizedSearchCV).
online
    Incremental-fit (online) workflows around skfolio's
    ``partial_fit``: ``online_predict``, ``online_score``,
    ``OnlineGridSearch``, ``OnlineRandomizedSearch``, plus the
    covariance-forecast-evaluation surface
    (``run_covariance_forecast_evaluation``,
    ``run_online_covariance_forecast_evaluation``,
    ``build_covariance_forecast_comparison``). Online instances are
    NOT thread-safe — use one per thread.
rebalancing
    Rebalancing frameworks (calendar-based, threshold-based,
    turnover computation, transaction cost estimation).

Research tools that sit beside the optimization pipeline and are not part of
it:

risk
    Value-at-Risk and expected shortfall (historical, parametric,
    Cornish-Fisher, Monte Carlo) and a Kupiec / Christoffersen backtest.
simulation
    GBM and bootstrap price paths, path summaries and the Kelly criterion.
strategies
    Moving-average and momentum signals, and a weight backtest with
    proportional costs and no look-ahead.
regression
    OLS with Newey-West errors, CAPM, the security market line and
    Fama-French-style factor regressions.
plotting
    plotly figures for correlation, the efficient frontier, returns and
    volatility, and the security market line.
derivatives
    Black-Scholes prices, Greeks and implied volatility, a binomial tree and
    a delta-hedging simulator.
credit
    The Merton structural credit model.
statarb
    Cointegration, a Kalman hedge ratio, z-score signals and PCA residual
    models. Research only.
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

logging.getLogger("optimizer").addHandler(logging.NullHandler())

# Top-level convenience re-exports for the public API; every symbol is also
# reachable via its owning submodule (``optimizer.<module>``). Purely additive —
# does not alter the submodule-access surface.
from optimizer.distance import NBinsMethod
from optimizer.exceptions import (
    ConfigurationError,
    ConvergenceError,
    DataError,
    OptimizationError,
    OptimizerError,
    ValidationError,
)
from optimizer.linear_model import (
    CSLinearRegressorWrapperConfig,
    build_cs_linear_regressor_wrapper,
)
from optimizer.optimization import FallbackPolicy
from optimizer.pre_selection import SelectKMeasure
from optimizer.scoring import PerfMeasureType, build_online_measure
from optimizer.synthetic import build_conditional_synthetic_data
from optimizer.tuning import search_results_dataframe
from optimizer.uncertainty_set import (
    CrossSectionalWeighting,
    OrthogonalUncertaintyShape,
)
from optimizer.universe import (
    InvestabilityScreenSelector,
    build_investability_screen,
)
from optimizer.validation import CrossValFailureReport, summarize_cv_failures

__all__ = [
    "CSLinearRegressorWrapperConfig",
    "ConfigurationError",
    "ConvergenceError",
    "CrossSectionalWeighting",
    "CrossValFailureReport",
    "DataError",
    "FallbackPolicy",
    "InvestabilityScreenSelector",
    "NBinsMethod",
    "OptimizationError",
    "OptimizerError",
    "OrthogonalUncertaintyShape",
    "PerfMeasureType",
    "SelectKMeasure",
    "ValidationError",
    "build_conditional_synthetic_data",
    "build_cs_linear_regressor_wrapper",
    "build_investability_screen",
    "build_online_measure",
    "search_results_dataframe",
    "summarize_cv_failures",
]

try:
    __version__ = _pkg_version("portopt-core")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
