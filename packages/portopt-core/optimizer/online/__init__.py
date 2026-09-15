"""Incremental-fit (online) workflows around skfolio's ``partial_fit``.

Online instances are NOT thread-safe — ``partial_fit`` accumulates
mutable state and ``OnlineGridSearch`` mutates the wrapped estimator
in place. Callers running scheduled jobs in multiple threads must
construct one wrapper per thread.

``Pipeline`` is rejected at the wrapper boundary because skfolio's
``online_predict`` / ``OnlineGridSearch`` cannot route ``partial_fit``
through a ``Pipeline``. Apply pre-selection to ``X`` upstream.

Input contract: ``X`` must be a float **linear**-returns frame (run
``prices_to_returns`` upstream — never log returns). DB ``Numeric``
columns read back as ``Decimal``; cast to ``float`` before calling
these wrappers, because an object-dtype frame breaks skfolio's numpy
covariance/mean math. Ragged history (newly listed tickers) yields NaN
rows/columns — the wrappers forward ``X`` unchanged, so drop or align
NaN upstream (or supply a skfolio NaN-aware estimator); they do not
clean it. Calendar ``freq`` modes additionally require a
``DatetimeIndex``.
"""

import optimizer.optimization  # noqa: F401
from optimizer.online._config import (
    CovarianceForecastConfig,
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
)
from optimizer.online._factory import (
    build_covariance_forecast_comparison,
    build_online_grid_search,
    build_online_randomized_search,
    run_covariance_forecast_evaluation,
    run_online_covariance_forecast_evaluation,
    run_online_predict,
    run_online_score,
)

# ``optimizer.optimization`` is eager-imported above so that
# ``optimization._config`` lands in ``sys.modules`` before anything
# else triggers the
# ``tuning → scoring → optimization → pipeline → tuning`` circular
# chain. Without this, the lazy default factories on
# ``OnlineGridSearchConfig`` would re-enter that cycle on first use.

__all__ = [
    "CovarianceForecastConfig",
    "OnlineGridSearchConfig",
    "OnlinePredictConfig",
    "OnlineRandomizedSearchConfig",
    "build_covariance_forecast_comparison",
    "build_online_grid_search",
    "build_online_randomized_search",
    "run_covariance_forecast_evaluation",
    "run_online_covariance_forecast_evaluation",
    "run_online_predict",
    "run_online_score",
]
