"""Covariance forecast evaluation (offline + online).

Covariance forecast evaluation is **optimizer-agnostic** — use it to
rank covariance estimators **before** plugging the winner into a
prior. It is not a substitute for a full backtest. Prefer the online
variant when every candidate exposes ``partial_fit`` — it is materially
faster and matches live-trading semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt
from skfolio.model_selection import (
    CovarianceForecastComparison,
    covariance_forecast_evaluation,
    online_covariance_forecast_evaluation,
)
from skfolio.moments.covariance._base import BaseCovariance

from optimizer.exceptions import ConfigurationError


@dataclass(frozen=True)
class CovarianceForecastConfig:
    """Immutable configuration for offline covariance forecast evaluation.

    Fields mirror :func:`skfolio.model_selection.covariance_forecast_evaluation`
    (skfolio 0.20.1) — verified at implementation time via
    ``inspect.signature``.

    Parameters
    ----------
    train_size : int
        Walk-forward training window size. Default ``252``.
    test_size : int
        Walk-forward evaluation window size. Default ``1``.
    expand_train : bool
        ``True`` for expanding window, ``False`` for rolling. Default
        ``False``.
    purged_size : int
        Embargo gap between train and test (days).
    """

    train_size: int = 252
    test_size: int = 1
    expand_train: bool = False
    purged_size: int = 0


@dataclass(frozen=True)
class OnlineCovarianceForecastConfig:
    """Immutable configuration for online covariance forecast evaluation.

    Fields mirror
    :func:`skfolio.model_selection.online_covariance_forecast_evaluation`.

    Parameters
    ----------
    warmup_size : int
        Number of initial observations consumed by the first
        ``partial_fit`` call. Default ``252``.
    test_size : int
        Walk-forward evaluation window size. Default ``1``.
    purged_size : int
        Embargo gap between warmup-tail and test (days).
    """

    warmup_size: int = 252
    test_size: int = 1
    purged_size: int = 0


def _evaluate_offline(
    name: str,
    estimator: BaseCovariance,
    X: npt.ArrayLike,
    config: CovarianceForecastConfig,
) -> Any:
    """Run offline evaluation on a single named estimator."""
    evaluation = covariance_forecast_evaluation(
        estimator,
        X,
        train_size=config.train_size,
        test_size=config.test_size,
        expand_train=config.expand_train,
        purged_size=config.purged_size,
    )
    _ = name  # forwarded via CovarianceForecastComparison.names
    return evaluation


def run_covariance_forecast_evaluation(
    estimators: Sequence[tuple[str, BaseCovariance]],
    X: npt.ArrayLike,
    *,
    config: CovarianceForecastConfig | None = None,
) -> CovarianceForecastComparison:
    """Rank a set of covariance estimators using walk-forward evaluation.

    Parameters
    ----------
    estimators : sequence of (name, BaseCovariance)
        Named candidate estimators. Pass positionally — ``BaseCovariance``
        instances are not frozen-serialisable.
    X : array-like
        Asset returns of shape ``(n_observations, n_assets)``.
    config : CovarianceForecastConfig or None
        Evaluation configuration. ``None`` triggers default.
    """
    if config is None:
        config = CovarianceForecastConfig()
    evaluations = [
        _evaluate_offline(name, estimator, X, config) for name, estimator in estimators
    ]
    names = [name for name, _ in estimators]
    return CovarianceForecastComparison(evaluations=evaluations, names=names)


def _evaluate_online(
    name: str,
    estimator: BaseCovariance,
    X: npt.ArrayLike,
    config: OnlineCovarianceForecastConfig,
) -> Any:
    """Run online evaluation on a single named estimator."""
    evaluation = online_covariance_forecast_evaluation(
        estimator,
        X,
        warmup_size=config.warmup_size,
        test_size=config.test_size,
        purged_size=config.purged_size,
    )
    _ = name  # forwarded via CovarianceForecastComparison.names
    return evaluation


def _validate_partial_fit(
    estimators: Sequence[tuple[str, BaseCovariance]],
) -> None:
    """Reject estimators missing ``partial_fit``."""
    offending = [name for name, est in estimators if not hasattr(est, "partial_fit")]
    if offending:
        joined = ", ".join(offending)
        raise ConfigurationError(
            f"Online covariance forecast evaluation requires every "
            f"estimator to implement `partial_fit`. Offending: {joined}"
        )


def run_online_covariance_forecast_evaluation(
    estimators: Sequence[tuple[str, BaseCovariance]],
    X: npt.ArrayLike,
    *,
    config: OnlineCovarianceForecastConfig | None = None,
) -> CovarianceForecastComparison:
    """Rank a set of ``partial_fit``-capable covariance estimators.

    Parameters
    ----------
    estimators : sequence of (name, BaseCovariance)
        Named candidate estimators. Each must implement ``partial_fit``.
    X : array-like
        Asset returns of shape ``(n_observations, n_assets)``.
    config : OnlineCovarianceForecastConfig or None
        Evaluation configuration. ``None`` triggers default.

    Raises
    ------
    ConfigurationError
        If any estimator lacks ``partial_fit``. The message lists every
        offending name so the caller can fix them in one pass.
    """
    if config is None:
        config = OnlineCovarianceForecastConfig()
    _validate_partial_fit(estimators)
    evaluations = [
        _evaluate_online(name, estimator, X, config) for name, estimator in estimators
    ]
    names = [name for name, _ in estimators]
    return CovarianceForecastComparison(evaluations=evaluations, names=names)
