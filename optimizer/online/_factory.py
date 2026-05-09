"""Factory functions for skfolio online learning workflows.

Online instances are NOT thread-safe — caller is responsible for
constructing one wrapper per thread.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
from skfolio.model_selection import (
    OnlineGridSearch,
    OnlineRandomizedSearch,
    online_predict,
    online_score,
)
from sklearn.pipeline import Pipeline

from optimizer.exceptions import ConfigurationError
from optimizer.online._config import (
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
)


def _build_scorer(scorer_config: Any) -> Any:
    """Lazy import of :func:`build_scorer` to avoid an import cycle."""
    from optimizer.scoring._factory import build_scorer

    return build_scorer(scorer_config)

_PARTIAL_FIT_ERROR = (
    "Online workflows require an estimator implementing `partial_fit`. "
    "Use `EmpiricalPrior(mu_estimator=EWMu(half_life=...), "
    "covariance_estimator=EWCovariance(half_life=...))` inside "
    "`MeanRisk(...)`. `Pipeline` is not supported here — apply "
    "pre-selection upstream."
)


def _validate_partial_fit(estimator: Any) -> None:
    """Reject ``Pipeline`` and any estimator without ``partial_fit``."""
    if isinstance(estimator, Pipeline) or not hasattr(estimator, "partial_fit"):
        raise ConfigurationError(_PARTIAL_FIT_ERROR)


def run_online_predict(
    estimator: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    *,
    config: OnlinePredictConfig,
) -> Any:
    """Forward to :func:`skfolio.model_selection.online_predict`.

    The returned :class:`MultiPeriodPortfolio` is fitted out-of-sample
    by repeatedly calling ``estimator.partial_fit`` after the warmup.
    Online state is mutated on ``estimator``; pass a fresh instance
    per thread.
    """
    _validate_partial_fit(estimator)
    return online_predict(
        estimator,
        X,
        y=y,
        warmup_size=config.warmup_size,
    )


def run_online_score(
    estimator: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    *,
    scorer: Any,
    config: OnlinePredictConfig,
) -> Any:
    """Forward to :func:`skfolio.model_selection.online_score`."""
    _validate_partial_fit(estimator)
    return online_score(
        estimator,
        X,
        y=y,
        warmup_size=config.warmup_size,
        scoring=scorer,
    )


def build_online_grid_search(
    config: OnlineGridSearchConfig,
    estimator: Any,
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
) -> OnlineGridSearch:
    """Build an :class:`OnlineGridSearch` with online cross-validation.

    Online instances are NOT thread-safe; construct one per thread.
    """
    _validate_partial_fit(estimator)
    return OnlineGridSearch(
        estimator=estimator,
        param_grid=param_grid,
        scoring=_build_scorer(config.base.scorer_config),
        warmup_size=config.online.warmup_size,
        n_jobs=config.online.n_jobs if config.online.n_jobs is not None
        else config.base.n_jobs,
        verbose=int(config.online.verbose),
    )


def build_online_randomized_search(
    config: OnlineRandomizedSearchConfig,
    estimator: Any,
    param_distributions: dict[str, Any],
) -> OnlineRandomizedSearch:
    """Build an :class:`OnlineRandomizedSearch` with online CV.

    Online instances are NOT thread-safe; construct one per thread.
    """
    _validate_partial_fit(estimator)
    return OnlineRandomizedSearch(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=config.base.n_iter,
        scoring=_build_scorer(config.base.scorer_config),
        warmup_size=config.online.warmup_size,
        random_state=config.base.random_state,
        n_jobs=config.online.n_jobs if config.online.n_jobs is not None
        else config.base.n_jobs,
        verbose=int(config.online.verbose),
    )
