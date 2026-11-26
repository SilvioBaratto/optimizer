"""Statistical-arbitrage research tools: pairs and PCA residuals.

* ``engle_granger`` tests whether two series are cointegrated and returns the
  hedge ratio, the spread and its half-life; ``ou_half_life`` measures the speed of
  mean reversion of any series.
* ``kalman_hedge_ratio`` tracks a time-varying hedge ratio and returns the
  one-step-ahead spread and its z-score.
* ``zscore_positions`` turns a z-score path into positions with entry, exit and
  stop bands.
* ``fit_pca_residual_model`` and ``pca_signals`` implement the Avellaneda and Lee
  (2010) eigenportfolio residual model and its trading thresholds. With the default
  ``PcaResidualConfig(min_kappa=0.0)`` a residual that is close to a random walk
  is kept as mean reverting, with a small ``kappa`` and a long half-life, because
  the regression constant keeps its AR(1) coefficient below one. Avellaneda and Lee
  trade only ``kappa > 8.4``; set ``PcaResidualConfig(min_kappa=8.4)`` for that
  screen.

Research tools only: nothing in this package models transaction costs, borrow
costs or short-sale limits, and the fund mandate is long-only and paper-only, so
none of it is wired to execution.
"""

from optimizer.statarb._cointegration import (
    EngleGrangerResult,
    engle_granger,
    ou_half_life,
)
from optimizer.statarb._config import (
    EngleGrangerConfig,
    KalmanConfig,
    PcaResidualConfig,
)
from optimizer.statarb._kalman import KalmanResult, kalman_hedge_ratio
from optimizer.statarb._pca import (
    PcaResidualModel,
    fit_pca_residual_model,
    pca_signals,
)
from optimizer.statarb._signals import zscore_positions

__all__ = [
    "EngleGrangerConfig",
    "EngleGrangerResult",
    "KalmanConfig",
    "KalmanResult",
    "PcaResidualConfig",
    "PcaResidualModel",
    "engle_granger",
    "fit_pca_residual_model",
    "kalman_hedge_ratio",
    "ou_half_life",
    "pca_signals",
    "zscore_positions",
]
