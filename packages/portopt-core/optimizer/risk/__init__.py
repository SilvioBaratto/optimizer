"""Value-at-Risk, expected shortfall and the backtest of their forecasts.

``compute_var`` estimates the loss a portfolio exceeds with a given probability
over a holding period, and the average loss beyond it, by one of four methods
(historical, parametric normal, Cornish-Fisher and bootstrap Monte Carlo).
``rolling_var_forecasts`` produces one-step forecasts that use only past returns, and
``backtest_var`` checks them with the Kupiec proportion-of-failures test, the
Christoffersen independence and conditional coverage tests, and the Basel
traffic-light zones.

Every result is a **loss reported as a positive fraction of portfolio value**, and
every input is a series of simple per-period returns.
"""

from optimizer.risk._backtest import VarBacktestResult, backtest_var
from optimizer.risk._config import VarConfig, VarMethod
from optimizer.risk._var import (
    VarResult,
    compute_var,
    cornish_fisher_quantile,
    rolling_var_forecasts,
)

__all__ = [
    "VarBacktestResult",
    "VarConfig",
    "VarMethod",
    "VarResult",
    "backtest_var",
    "compute_var",
    "cornish_fisher_quantile",
    "rolling_var_forecasts",
]
