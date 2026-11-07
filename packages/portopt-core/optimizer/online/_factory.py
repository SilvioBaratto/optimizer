"""Factory functions for skfolio online learning workflows.

Online instances are NOT thread-safe — caller is responsible for
constructing one wrapper per thread.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
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

from optimizer.exceptions import ConfigurationError, OptimizationError
from optimizer.online._config import (
    CovarianceForecastConfig,
    OnlineGridSearchConfig,
    OnlinePredictConfig,
    OnlineRandomizedSearchConfig,
)
from optimizer.validation._factory import check_index_alignment


def _resolve_measure(scorer_config: Any) -> Any:
    """Resolve a `ScorerConfig` to a skfolio ``BaseMeasure``.

    Online *portfolio* evaluation (``OnlineGridSearch`` / ``online_score``
    on optimizers) rejects ``make_scorer`` objects and requires a
    ``BaseMeasure`` passed directly. Ratio, perf and risk measure families
    all map onto one; custom callables and the benchmark-relative
    Information Ratio cannot be expressed as a single ``BaseMeasure``, so
    they are unsupported here.
    """
    from optimizer.optimization._config import RatioMeasureType
    from optimizer.scoring._factory import _resolve_native_measure

    ratio_measure = getattr(scorer_config, "ratio_measure", None)
    if ratio_measure == RatioMeasureType.INFORMATION_RATIO:
        raise ConfigurationError(
            "Information Ratio is benchmark-relative and not expressible as "
            "a skfolio BaseMeasure for online portfolio search. Use a "
            "built-in ratio measure (e.g. SHARPE_RATIO, SORTINO_RATIO)."
        )
    if (
        ratio_measure is None
        and getattr(scorer_config, "perf_measure", None) is None
        and getattr(scorer_config, "risk_measure", None) is None
    ):
        raise ConfigurationError(
            "Online portfolio search requires a built-in ratio measure, perf "
            "measure or risk measure. A ScorerConfig with no measure family "
            "(custom callable) is not supported — online evaluation scores "
            "the aggregated MultiPeriodPortfolio and rejects make_scorer "
            "objects. Set ratio_measure=RatioMeasureType.SHARPE_RATIO (or "
            "similar), or use perf_measure / risk_measure."
        )
    return _resolve_native_measure(scorer_config)


def _resolve_portfolio_params(scorer_config: Any) -> dict[str, Any]:
    """Translate a `ScorerConfig`'s risk-free rate and annualization to skfolio.

    ``OnlineGridSearch`` / ``OnlineRandomizedSearch`` evaluate the aggregated
    `MultiPeriodPortfolio`, which only sees ``risk_free_rate`` and
    ``annualization_factor`` through ``portfolio_params``. The scorer's rate is
    always forwarded, an explicit ``0.0`` included: leaving it out would score
    each candidate at its own optimizer's ``risk_free_rate``, so a grid over
    that parameter would be ranked against a moving hurdle.

    Returns:
        The ``portfolio_params`` that apply the scorer's rate and, when set,
        its annualization factor.
    """
    from optimizer.scoring._factory import build_online_portfolio_params

    return build_online_portfolio_params(scorer_config)


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


def _raise_if_no_candidate_scored(
    search: OnlineGridSearch | OnlineRandomizedSearch,
) -> None:
    """Fail loudly when a fitted online search has no finite candidate score.

    With ``raise_on_failure=False`` (or a finite ``error_score``) a candidate
    that cannot be optimised is scored NaN rather than aborting the search.
    When every candidate is NaN they all tie at rank 1 and ``best_params_``
    silently becomes the first candidate, which is then refit and used as if it
    had won.
    """
    scores = np.asarray(search.cv_results_["mean_score"], dtype=float)
    if scores.size and np.isnan(scores).all():
        msg = (
            f"every candidate scored NaN: all {scores.size} parameter "
            "combinations failed to fit or score on the online walk, so no "
            "best candidate can be chosen. Check that the candidate "
            "constraints are feasible for the universe and that the data has "
            "no ragged or NaN history, or set error_score='raise' to surface "
            "the underlying error"
        )
        raise OptimizationError(msg)


class _CheckedOnlineGridSearch(OnlineGridSearch):  # type: ignore[misc]
    """``OnlineGridSearch`` that refuses to report a best candidate nobody scored."""

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> _CheckedOnlineGridSearch:
        super().fit(X, y, **fit_params)
        _raise_if_no_candidate_scored(self)
        return self


class _CheckedOnlineRandomizedSearch(OnlineRandomizedSearch):  # type: ignore[misc]
    """``OnlineRandomizedSearch`` that refuses to report an unscored best candidate."""

    def fit(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> _CheckedOnlineRandomizedSearch:
        super().fit(X, y, **fit_params)
        _raise_if_no_candidate_scored(self)
        return self


def run_online_predict(
    estimator: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    *,
    config: OnlinePredictConfig,
    portfolio_params: dict[str, Any] | None = None,
) -> Any:
    """Forward to `online_predict`.

    The returned `MultiPeriodPortfolio` is fitted out-of-sample
    by repeatedly calling ``estimator.partial_fit`` after the warmup.
    Online state is mutated on ``estimator``; pass a fresh instance
    per thread.

    Raises:
        DataError: If a pandas ``y`` has the same length as ``X`` but another
            index. Rows are paired by position, so it would be matched to the
            wrong dates.
    """
    _validate_partial_fit(estimator)
    check_index_alignment(X, "y", y)
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
    portfolio_params: dict[str, Any] | None = None,
) -> Any:
    """Forward to `online_score`.

    ``scorer`` must be a skfolio ``BaseMeasure`` (e.g.
    ``RatioMeasure.SHARPE_RATIO``) or ``None`` for portfolio optimizers —
    online portfolio evaluation rejects ``make_scorer`` objects. When
    ``per_step`` is ``True`` a per-rebalance array is returned instead of
    a single aggregated scalar.

    A bare ``scorer`` cannot carry a ``risk_free_rate`` or
    ``annualization_factor``; pass ``portfolio_params`` (see
    `optimizer.scoring.build_online_portfolio_params`) so the aggregated
    portfolio is scored at the same rate as the online searches.

    Raises:
        DataError: If a pandas ``y`` has the same length as ``X`` but another
            index.
    """
    _validate_partial_fit(estimator)
    check_index_alignment(X, "y", y)
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
        portfolio_params=portfolio_params,
    )


def build_online_grid_search(
    config: OnlineGridSearchConfig,
    estimator: Any,
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
) -> OnlineGridSearch:
    """Build an `OnlineGridSearch` with online cross-validation.

    Scoring is resolved from ``config.base.scorer_config`` to a skfolio
    ``BaseMeasure`` (online portfolio search rejects ``make_scorer``).
    Online instances are NOT thread-safe; construct one per thread.

    Returns:
        A fitted-ready grid search. Its ``fit`` raises `OptimizationError` when
        no candidate produced a finite score.
    """
    _validate_partial_fit(estimator)
    return _CheckedOnlineGridSearch(
        estimator=estimator,
        param_grid=param_grid,
        scoring=_resolve_measure(config.base.scorer_config),
        portfolio_params=_resolve_portfolio_params(config.base.scorer_config),
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
        refit=config.base.refit,
        # sklearn accepts "raise" or a float for error_score; its stub types it float.
        error_score=config.base.error_score,  # pyright: ignore[reportArgumentType]
        # Either level may switch verbosity on; the stricter (louder) one wins.
        verbose=max(config.base.verbose, int(config.online.verbose)),
    )


def build_online_randomized_search(
    config: OnlineRandomizedSearchConfig,
    estimator: Any,
    param_distributions: dict[str, Any],
) -> OnlineRandomizedSearch:
    """Build an `OnlineRandomizedSearch` with online CV.

    Scoring is resolved from ``config.base.scorer_config`` to a skfolio
    ``BaseMeasure`` (online portfolio search rejects ``make_scorer``).
    Online instances are NOT thread-safe; construct one per thread.

    Returns:
        A fitted-ready randomized search. Its ``fit`` raises
        `OptimizationError` when no candidate produced a finite score.
    """
    _validate_partial_fit(estimator)
    return _CheckedOnlineRandomizedSearch(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=config.base.n_iter,
        scoring=_resolve_measure(config.base.scorer_config),
        portfolio_params=_resolve_portfolio_params(config.base.scorer_config),
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
        refit=config.base.refit,
        # sklearn accepts "raise" or a float for error_score; its stub types it float.
        error_score=config.base.error_score,  # pyright: ignore[reportArgumentType]
        # Either level may switch verbosity on; the stricter (louder) one wins.
        verbose=max(config.base.verbose, int(config.online.verbose)),
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
    or a `Pipeline` (unlike the online path).
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
    """Build a side-by-side `CovarianceForecastComparison`.

    Ranks several fitted `CovarianceForecastEvaluation` results
    (e.g. EW vs regime-adjusted covariance) before embedding one in a
    prior.

    Raises:
        ConfigurationError: If ``evaluations`` is empty, ``names`` has a
            different length, or two candidates resolve to the same name.
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
    # skfolio keys its summaries by name (falling back to each evaluation's own
    # name), so a collision silently drops a candidate from the ranking.
    resolved = names or [
        ev.name if ev.name is not None else f"Estimator {i}"
        for i, ev in enumerate(evaluations)
    ]
    duplicated = [name for name, n in Counter(resolved).items() if n > 1]
    if duplicated:
        raise ConfigurationError(
            "Covariance forecast comparison requires unique candidate names. "
            f"Duplicated: {', '.join(duplicated)}"
        )
    return CovarianceForecastComparison(evaluations, names=names)
