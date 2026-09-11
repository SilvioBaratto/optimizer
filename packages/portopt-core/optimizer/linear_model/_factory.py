"""Factory for skfolio cross-sectional linear regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skfolio.linear_model import CSLinearRegression, CSLinearRegressorWrapper

from optimizer.exceptions import ConfigurationError
from optimizer.linear_model._config import (
    CSLinearRegressionConfig,
    CSLinearRegressorWrapperConfig,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


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
        and an optional ``cs_weights: (T, N)`` at ``fit`` time. Zero-weight
        pairs are excluded and may contain NaN.
    """
    return CSLinearRegression(fit_intercept=config.fit_intercept)


def build_cs_linear_regressor_wrapper(
    config: CSLinearRegressorWrapperConfig,
    *,
    regressor: BaseEstimator,
) -> CSLinearRegressorWrapper:
    """Build a skfolio :class:`CSLinearRegressorWrapper` from *config*.

    Adapts an arbitrary scikit-learn ``Regressor`` to the cross-sectional
    contract, fitting it independently per observation. Use when a
    regularised or non-linear per-period estimator (Ridge, Lasso, a tree
    ensemble) is needed instead of plain OLS.

    Parameters
    ----------
    config : CSLinearRegressorWrapperConfig
        Wrapper configuration (parallelism + caller-side hints).
    regressor : sklearn.base.BaseEstimator
        The per-period estimator (non-serialisable, hence a keyword-only
        factory argument rather than a config field). Must implement the
        scikit-learn ``fit`` / ``predict`` regressor API.

    Returns
    -------
    CSLinearRegressorWrapper
        A fitted-ready estimator. Expects ``X: (T, N, K)``, ``y: (T, N)``
        and an optional ``cs_weights: (T, N)`` at ``fit`` time.

    Raises
    ------
    ConfigurationError
        If ``regressor`` is ``None``.
    """
    if regressor is None:
        raise ConfigurationError("regressor must be provided (got None)")
    return CSLinearRegressorWrapper(regressor=regressor, n_jobs=config.n_jobs)
