"""Factory for skfolio cross-sectional linear regression."""

from __future__ import annotations

from skfolio.linear_model import CSLinearRegression

from optimizer.linear_model._config import CSLinearRegressionConfig


def build_cs_linear_regression(
    config: CSLinearRegressionConfig,
) -> CSLinearRegression:
    """Build a skfolio :class:`CSLinearRegression` from *config*.

    Parameters
    ----------
    config : CSLinearRegressionConfig
        CS linear regression configuration.

    Returns
    -------
    CSLinearRegression
        A fitted-ready estimator. Expects ``X: (T, N, K)``, ``y: (T, N)``
        and an optional ``cs_weights: (T, N)`` at ``fit`` time.
    """
    return CSLinearRegression(fit_intercept=config.fit_intercept)
