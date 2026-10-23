"""Factory functions for building hyperparameter search estimators."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from optimizer.exceptions import OptimizationError
from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig

# ``build_scorer`` (from ``optimizer.scoring``) and ``build_walk_forward`` (from
# ``optimizer.validation``) are imported lazily inside the factory bodies to
# avoid a module-load import cycle between ``tuning``, ``optimizer.scoring`` and
# ``optimizer.validation``; importing them at module top level would make the
# package import-order-dependent.

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _raise_if_no_candidate_scored(search: GridSearchCV | RandomizedSearchCV) -> None:
    """Fail loudly when a fitted search has no finite candidate score.

    With ``error_score=nan`` sklearn demotes a candidate that fails on some
    fold, which is the intended resilience.  But when *every* candidate is NaN
    (a scorer that cannot run, e.g. a benchmark missing a scored date, or a
    solver failing on a shared fold) they all tie at rank 1 and ``best_params_``
    silently becomes the first candidate, which is then refit and used as if it
    had won.
    """
    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    if scores.size and np.isnan(scores).all():
        msg = (
            f"every candidate scored NaN: all {scores.size} parameter "
            "combinations failed to fit or score on at least one "
            "cross-validation fold, so no best candidate can be chosen. "
            "Check that benchmark_returns covers every scored window and that "
            "the data has no ragged or NaN history, or set error_score='raise' "
            "to surface the underlying error"
        )
        raise OptimizationError(msg)


class _CheckedGridSearchCV(GridSearchCV):
    """``GridSearchCV`` that refuses to report a best candidate nobody scored."""

    def fit(self, X: Any, y: Any = None, **params: Any) -> _CheckedGridSearchCV:
        super().fit(X, y, **params)
        _raise_if_no_candidate_scored(self)
        return self


class _CheckedRandomizedSearchCV(RandomizedSearchCV):
    """``RandomizedSearchCV`` that refuses to report a best candidate nobody scored."""

    def fit(self, X: Any, y: Any = None, **params: Any) -> _CheckedRandomizedSearchCV:
        super().fit(X, y, **params)
        _raise_if_no_candidate_scored(self)
        return self


def build_grid_search_cv(
    estimator: Any,
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    config: GridSearchConfig | None = None,
    *,
    score_func: Callable[..., float] | None = None,
    benchmark_returns: pd.Series | None = None,
) -> GridSearchCV:
    """Build a GridSearchCV with temporal cross-validation.

    Args:
        estimator: The skfolio optimiser or pipeline to tune.
        param_grid: Parameter grid. Keys use sklearn double-underscore notation
            for nested estimators (e.g.
            ``"prior_estimator__mu_estimator__half_life"``). A list of dicts
            defines disjoint sub-spaces.
        config: Tuning configuration. Defaults to ``GridSearchConfig()``
            (quarterly walk-forward, Sharpe ratio scoring).
        score_func: Custom per-portfolio scorer, required when
            ``config.scorer_config.ratio_measure`` is ``None``. Forwarded to
            ``build_scorer``.
        benchmark_returns: Benchmark return series, required when the scorer
            config selects the Information Ratio. Forwarded to ``build_scorer``.

    Returns:
        A fitted-ready grid search estimator. Its ``fit`` raises
        `OptimizationError` when no candidate produced a finite score.
    """
    from optimizer.scoring._factory import build_scorer
    from optimizer.validation._factory import build_walk_forward

    if config is None:
        config = GridSearchConfig()

    cv = build_walk_forward(config.cv_config)
    scoring = build_scorer(
        config.scorer_config,
        score_func=score_func,
        benchmark_returns=benchmark_returns,
    )

    return _CheckedGridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=config.n_jobs,
        refit=config.refit,
        verbose=config.verbose,
        # sklearn accepts "raise" or a float for error_score; its stub types it float.
        error_score=config.error_score,  # pyright: ignore[reportArgumentType]
        return_train_score=config.return_train_score,
    )


def build_randomized_search_cv(
    estimator: Any,
    param_distributions: dict[str, Any] | list[dict[str, Any]],
    config: RandomizedSearchConfig | None = None,
    *,
    score_func: Callable[..., float] | None = None,
    benchmark_returns: pd.Series | None = None,
) -> RandomizedSearchCV:
    """Build a RandomizedSearchCV with temporal cross-validation.

    Args:
        estimator: The skfolio optimiser or pipeline to tune.
        param_distributions: Parameter distributions. Values may be lists
            (discrete) or ``scipy.stats`` distributions (continuous, e.g.
            ``scipy.stats.loguniform(0.01, 1)``).
        config: Tuning configuration. Defaults to ``RandomizedSearchConfig()``
            (50 iterations, quarterly walk-forward, Sharpe ratio scoring).
        score_func: Custom per-portfolio scorer, forwarded to ``build_scorer``.
        benchmark_returns: Benchmark return series for Information-Ratio
            scoring, forwarded to ``build_scorer``.

    Returns:
        A fitted-ready randomised search estimator. Its ``fit`` raises
        `OptimizationError` when no candidate produced a finite score.
    """
    from optimizer.scoring._factory import build_scorer
    from optimizer.validation._factory import build_walk_forward

    if config is None:
        config = RandomizedSearchConfig()

    cv = build_walk_forward(config.cv_config)
    scoring = build_scorer(
        config.scorer_config,
        score_func=score_func,
        benchmark_returns=benchmark_returns,
    )

    return _CheckedRandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=config.n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=config.n_jobs,
        refit=config.refit,
        verbose=config.verbose,
        # sklearn accepts "raise" or a float for error_score; its stub types it float.
        error_score=config.error_score,  # pyright: ignore[reportArgumentType]
        random_state=config.random_state,
        return_train_score=config.return_train_score,
    )


def search_results_dataframe(
    search: GridSearchCV | RandomizedSearchCV,
) -> pd.DataFrame:
    """Return a tidy, rank-sorted view of a fitted search's ``cv_results_``.

    Materialises ``search.cv_results_`` into a DataFrame sorted by
    ``rank_test_score`` (best first), so the outcome of a hyperparameter sweep
    can be inspected or logged without hand-indexing the raw results dict.

    Args:
        search: A **fitted** search estimator (``.fit`` already called).

    Returns:
        ``cv_results_`` as a DataFrame, ascending by ``rank_test_score`` when
        that column is present (it is absent only for degenerate
        single-candidate multimetric setups).

    Raises:
        AttributeError: If *search* has not been fitted (no ``cv_results_``).
    """
    import pandas as pd

    if not hasattr(search, "cv_results_"):
        raise AttributeError(
            "search has no cv_results_; call .fit(X) before search_results_dataframe()"
        )
    df = pd.DataFrame(search.cv_results_)
    if "rank_test_score" in df.columns:
        df = df.sort_values("rank_test_score", kind="stable").reset_index(drop=True)
    return df
