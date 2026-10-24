"""Factory for skfolio cross-sectional linear regression."""

from __future__ import annotations

import inspect
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
    """Build a skfolio CSLinearRegression from config.

    Args:
        config: CS linear regression configuration.

    Returns:
        A fitted-ready estimator. Expects X of shape (T, N, K), y of shape
        (T, N), and an optional cs_weights of shape (T, N) at fit time.
        Zero-weight pairs are excluded and may contain NaN.
    """
    return CSLinearRegression(fit_intercept=config.fit_intercept)


def build_cs_linear_regressor_wrapper(
    config: CSLinearRegressorWrapperConfig,
    *,
    regressor: BaseEstimator,
) -> CSLinearRegressorWrapper:
    """Build a skfolio CSLinearRegressorWrapper from config.

    Adapts a scikit-learn linear regressor to the cross-sectional contract,
    fitting it independently per observation. Use when a regularised or robust
    per-period estimator (Ridge, Lasso, ElasticNet, Huber) is needed instead
    of plain OLS. Estimators without ``fit_intercept`` / ``coef_`` /
    ``intercept_`` (tree ensembles, boosting) are not supported.

    Args:
        config: Wrapper configuration (parallelism + caller-side hints).
        regressor: The per-period estimator (non-serialisable, hence a
            keyword-only factory argument rather than a config field). Must
            define ``fit_intercept`` and accept ``sample_weight`` in ``fit``;
            skfolio additionally requires it to expose ``coef_`` and
            ``intercept_`` after fitting, which can only be checked at fit
            time.

    Returns:
        A fitted-ready estimator. Expects X of shape (T, N, K), y of shape
        (T, N), and an optional cs_weights of shape (T, N) at fit time.

    Raises:
        ConfigurationError: If regressor is None, does not define
            ``fit_intercept``, or its ``fit`` does not accept ``sample_weight``.
    """
    if regressor is None:
        raise ConfigurationError("regressor must be provided (got None)")
    if not hasattr(regressor, "fit_intercept"):
        raise ConfigurationError(
            f"{type(regressor).__name__} does not define fit_intercept; "
            "only linear regressors exposing fit_intercept, coef_ and "
            "intercept_ (e.g. Ridge, Lasso, ElasticNet) are supported"
        )
    if not _accepts_sample_weight(regressor):
        raise ConfigurationError(
            f"{type(regressor).__name__}.fit must accept sample_weight, "
            "which the per-period cs_weights are passed through"
        )
    return CSLinearRegressorWrapper(regressor=regressor, n_jobs=config.n_jobs)


def _accepts_sample_weight(regressor: BaseEstimator) -> bool:
    fit = getattr(regressor, "fit", None)
    if fit is None:
        return False
    try:
        params = inspect.signature(fit).parameters
    except (TypeError, ValueError):
        # Signature is unavailable for some C-level callables; skfolio will
        # surface any real incompatibility at fit time.
        return True
    return "sample_weight" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
