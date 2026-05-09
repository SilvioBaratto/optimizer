"""Incremental-fit (online) workflows around skfolio's ``partial_fit``.

Online instances are NOT thread-safe — ``partial_fit`` accumulates
mutable state and ``OnlineGridSearch`` mutates the wrapped estimator
in place. Callers running scheduled jobs in multiple threads must
construct one wrapper per thread.

``Pipeline`` is rejected at the wrapper boundary because skfolio's
``online_predict`` / ``OnlineGridSearch`` cannot route ``partial_fit``
through a ``Pipeline``. Apply pre-selection to ``X`` upstream.
"""

import optimizer.optimization  # noqa: F401
from optimizer.online._config import (
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
)
from optimizer.online._factory import (
    build_online_grid_search,
    build_online_randomized_search,
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
    "OnlineGridSearchConfig",
    "OnlinePredictConfig",
    "OnlineRandomizedSearchConfig",
    "build_online_grid_search",
    "build_online_randomized_search",
    "run_online_predict",
    "run_online_score",
]
