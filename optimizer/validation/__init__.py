"""Model selection and cross-validation for portfolio backtesting.

Includes Walk-Forward backtesting, Combinatorial Purged Cross-Validation
(CPCV), Multiple Randomized Cross-Validation, and covariance forecast
evaluation (offline + online paths).

Covariance forecast evaluation is **optimizer-agnostic** — use it to
rank covariance estimators **before** plugging the winner into a
prior. It is not a substitute for a full backtest. Prefer the online
variant when every candidate exposes ``partial_fit`` — it is materially
faster and matches live-trading semantics.
"""

from skfolio.model_selection import CovarianceForecastComparison

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
from optimizer.validation._forecast import (
    CovarianceForecastConfig,
    OnlineCovarianceForecastConfig,
    run_covariance_forecast_evaluation,
    run_online_covariance_forecast_evaluation,
)
from optimizer.validation._regime import (
    RegimeValidationConfig,
    RegimeValidationResult,
    run_regime_validation,
)

__all__ = [
    "CPCVConfig",
    "CovarianceForecastComparison",
    "CovarianceForecastConfig",
    "MultipleRandomizedCVConfig",
    "OnlineCovarianceForecastConfig",
    "RegimeValidationConfig",
    "RegimeValidationResult",
    "WalkForwardConfig",
    "build_cpcv",
    "build_multiple_randomized_cv",
    "build_walk_forward",
    "compute_optimal_folds",
    "run_covariance_forecast_evaluation",
    "run_cross_val",
    "run_online_covariance_forecast_evaluation",
    "run_regime_validation",
]
