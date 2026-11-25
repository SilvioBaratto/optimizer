"""Regression tools: ordinary least squares, CAPM and factor models.

``ols`` is the core: a least squares fit with t-statistics, p-values and optional
Newey-West standard errors for autocorrelated, heteroskedastic residuals. On top of
it sit the CAPM report (``capm_regression``, ``rolling_beta`` and
``security_market_line``) and the multifactor regressions
(``multifactor_regression`` and ``fama_french_regression``).

The module does not load data. Factor returns, the market and the risk-free rate are
passed in by the caller, per period and on a shared index. Missing, infinite or
misaligned values raise ``DataError`` instead of being dropped, so a statistic never
silently describes a different sample than the one that was passed.
"""

from optimizer.regression._capm import (
    CapmResult,
    SmlResult,
    capm_regression,
    rolling_beta,
    security_market_line,
)
from optimizer.regression._config import FamaFrenchFactor
from optimizer.regression._factors import (
    FactorRegressionResult,
    fama_french_regression,
    multifactor_regression,
)
from optimizer.regression._ols import OlsResult, ols

__all__ = [
    "CapmResult",
    "FactorRegressionResult",
    "FamaFrenchFactor",
    "OlsResult",
    "SmlResult",
    "capm_regression",
    "fama_french_regression",
    "multifactor_regression",
    "ols",
    "rolling_beta",
    "security_market_line",
]
