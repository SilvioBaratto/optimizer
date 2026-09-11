"""Factory functions for skfolio online learning workflows.

Online instances are NOT thread-safe — caller is responsible for
constructing one wrapper per thread.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
import pandas as pd
from skfolio.model_selection import (
    CovarianceForecastComparison,
    CovarianceForecastEvaluation,
    OnlineGridSearch,
    OnlineRandomizedSearch,
    covariance_forecast_evaluation,
    online_covariance_forecast_evaluation,
    online_predict,
    online_score,
)
from sklearn.pipeline import Pipeline

from optimizer.exceptions import ConfigurationError
from optimizer.online._config import (
    CovarianceForecastConfig,
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
)


def _resolve_measure(scorer_config: Any) -> Any:
    """Resolve a :class:`ScorerConfig` to a skfolio ``BaseMeasure``.

    Online *portfolio* evaluation (``OnlineGridSearch`` / ``online_score``
    on optimizers) rejects ``make_scorer`` objects and requires a
    ``BaseMeasure`` passed directly. Custom callables and the
    benchmark-relative Information Ratio cannot be expressed as a single
    ``BaseMeasure``, so they are unsupported here.
    """
    from optimizer.optimization._config import RatioMeasureType
    from optimizer.scoring._factory import _RATIO_MEASURE_MAP

    measure = getattr(scorer_config, "ratio_measure", None)
    if measure is None:
        raise ConfigurationError(
            "Online portfolio search requires a built-in ratio measure. "
            "ScorerConfig.ratio_measure=None (custom callable) is not "
            "supported — online evaluation scores the aggregated "
            "MultiPeriodPortfolio and rejects make_scorer objects. "
            "Set ratio_measure=RatioMeasureType.SHARPE_RATIO (or similar)."
        )
    if measure == RatioMeasureType.INFORMATION_RATIO:
        raise ConfigurationError(
            "Information Ratio is benchmark-relative and not expressible as "
            "a skfolio BaseMeasure for online portfolio search. Use a "
            "built-in ratio measure (e.g. SHARPE_RATIO, SORTINO_RATIO)."
        )
    return _RATIO_MEASURE_MAP[measure]


def _to_offset(freq_offset: str | None) -> Any:
    """Parse a pandas offset alias into a ``BaseOffset`` (or ``None``)."""
    if freq_offset is None:
        return None
    return pd.tseries.frequencies.to_offset(freq_offset)


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
    portfolio_params: dict[str, Any] | None = None,
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
        test_size=config.test_size,
        purged_size=config.purged_size,
        freq=config.freq,
        freq_offset=_to_offset(config.freq_offset),
        previous=config.previous,
        reduce_test=config.reduce_test,
        portfolio_params=portfolio_params,
    )


def run_online_score(
    estimator: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    *,
    scorer: Any,
    config: OnlinePredictConfig,
    per_step: bool = False,
) -> Any:
    """Forward to :func:`skfolio.model_selection.online_score`.

    ``scorer`` must be a skfolio ``BaseMeasure`` (e.g.
    ``RatioMeasure.SHARPE_RATIO``) or ``None`` for portfolio optimizers —
    online portfolio evaluation rejects ``make_scorer`` objects. When
    ``per_step`` is ``True`` a per-rebalance array is returned instead of
    a single aggregated scalar.
    """
    _validate_partial_fit(estimator)
    return online_score(
        estimator,
        X,
        y=y,
        warmup_size=config.warmup_size,
        test_size=config.test_size,
        purged_size=config.purged_size,
        freq=config.freq,
        freq_offset=_to_offset(config.freq_offset),
        previous=config.previous,
        reduce_test=config.reduce_test,
        scoring=scorer,
        per_step=per_step,
    )


def build_online_grid_search(
    config: OnlineGridSearchConfig,
    estimator: Any,
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
) -> OnlineGridSearch:
    """Build an :class:`OnlineGridSearch` with online cross-validation.

    Scoring is resolved from ``config.base.scorer_config`` to a skfolio
    ``BaseMeasure`` (online portfolio search rejects ``make_scorer``).
    Online instances are NOT thread-safe; construct one per thread.
    """
    _validate_partial_fit(estimator)
    return OnlineGridSearch(
        estimator=estimator,
        param_grid=param_grid,
        scoring=_resolve_measure(config.base.scorer_config),
        warmup_size=config.online.warmup_size,
        test_size=config.online.test_size,
        purged_size=config.online.purged_size,
        freq=config.online.freq,
        freq_offset=_to_offset(config.online.freq_offset),
        previous=config.online.previous,
        reduce_test=config.online.reduce_test,
        n_jobs=config.online.n_jobs
        if config.online.n_jobs is not None
        else config.base.n_jobs,
        verbose=int(config.online.verbose),
    )


def build_online_randomized_search(
    config: OnlineRandomizedSearchConfig,
    estimator: Any,
    param_distributions: dict[str, Any],
) -> OnlineRandomizedSearch:
    """Build an :class:`OnlineRandomizedSearch` with online CV.

    Scoring is resolved from ``config.base.scorer_config`` to a skfolio
    ``BaseMeasure`` (online portfolio search rejects ``make_scorer``).
    Online instances are NOT thread-safe; construct one per thread.
    """
    _validate_partial_fit(estimator)
    return OnlineRandomizedSearch(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=config.base.n_iter,
        scoring=_resolve_measure(config.base.scorer_config),
        warmup_size=config.online.warmup_size,
        test_size=config.online.test_size,
        purged_size=config.online.purged_size,
        freq=config.online.freq,
        freq_offset=_to_offset(config.online.freq_offset),
        previous=config.online.previous,
        reduce_test=config.online.reduce_test,
        random_state=config.base.random_state,
        n_jobs=config.online.n_jobs
        if config.online.n_jobs is not None
        else config.base.n_jobs,
        verbose=int(config.online.verbose),
    )


# ---------------------------------------------------------------------------
# Covariance forecast evaluation (optimizer-independent diagnostics)
# ---------------------------------------------------------------------------


def run_covariance_forecast_evaluation(
    estimator: Any,
    X: npt.ArrayLike,
    *,
    config: CovarianceForecastConfig | None = None,
    portfolio_weights: npt.ArrayLike | None = None,
) -> CovarianceForecastEvaluation:
    """Walk-forward covariance-forecast evaluation (refit each split).

    Diagnoses a covariance estimator's out-of-sample calibration
    independently of any optimizer. Accepts a plain covariance estimator
    or a :class:`~sklearn.pipeline.Pipeline` (unlike the online path).
    ``portfolio_weights`` optionally scores portfolio-direction
    calibration; ``None`` defaults to an inverse-volatility direction.
    """
    if config is None:
        config = CovarianceForecastConfig()
    return covariance_forecast_evaluation(
        estimator,
        X,
        train_size=config.train_size,
        test_size=config.test_size,
        expand_train=config.expand_train,
        purged_size=config.purged_size,
        portfolio_weights=portfolio_weights,
    )


def run_online_covariance_forecast_evaluation(
    estimator: Any,
    X: npt.ArrayLike,
    *,
    config: CovarianceForecastConfig | None = None,
    portfolio_weights: npt.ArrayLike | None = None,
) -> CovarianceForecastEvaluation:
    """Online (``partial_fit``-based) covariance-forecast evaluation.

    Faster than the walk-forward path — a single stateful estimator is
    updated incrementally. Requires ``partial_fit`` and rejects
    ``Pipeline`` (mirrors the online-search boundary). ``config.train_size``
    is forwarded as the warmup window; ``expand_train`` is ignored because
    ``partial_fit`` is inherently cumulative.
    """
    _validate_partial_fit(estimator)
    if config is None:
        config = CovarianceForecastConfig()
    return online_covariance_forecast_evaluation(
        estimator,
        X,
        warmup_size=config.train_size,
        test_size=config.test_size,
        purged_size=config.purged_size,
        portfolio_weights=portfolio_weights,
    )


def build_covariance_forecast_comparison(
    evaluations: list[CovarianceForecastEvaluation],
    names: list[str] | None = None,
) -> CovarianceForecastComparison:
    """Build a side-by-side :class:`CovarianceForecastComparison`.

    Ranks several fitted :class:`CovarianceForecastEvaluation` results
    (e.g. EW vs regime-adjusted covariance) before embedding one in a
    prior.
    """
    if not evaluations:
        raise ConfigurationError(
            "build_covariance_forecast_comparison requires at least one "
            "CovarianceForecastEvaluation."
        )
    if names is not None and len(names) != len(evaluations):
        raise ConfigurationError(
            f"names length ({len(names)}) must match evaluations length "
            f"({len(evaluations)})."
        )
    return CovarianceForecastComparison(evaluations, names=names)
