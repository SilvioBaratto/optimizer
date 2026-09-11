"""Factory functions for building hyperparameter search estimators."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig

# ``build_scorer`` (from ``optimizer.scoring``) and ``build_walk_forward`` (from
# ``optimizer.validation``) are imported lazily inside the factory bodies:
# ``tuning`` sits on an import cycle (scoring -> optimization -> pipeline ->
# tuning -> scoring), so importing them at module top level would make the
# package import-order-dependent.

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def build_grid_search_cv(
    estimator: Any,
    param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
    config: GridSearchConfig | None = None,
    *,
    score_func: Callable[..., float] | None = None,
    benchmark_returns: pd.Series | None = None,
) -> GridSearchCV:
    """Build a :class:`GridSearchCV` with temporal cross-validation.

    Parameters
    ----------
    estimator : BaseEstimator
        The skfolio optimiser or pipeline to tune.
    param_grid : dict or list of dict
        Parameter grid.  Keys use sklearn double-underscore
        notation for nested estimators (e.g.
        ``"prior_estimator__mu_estimator__half_life"``).  A list of
        dicts defines disjoint sub-spaces.
    config : GridSearchConfig or None
        Tuning configuration.  Defaults to ``GridSearchConfig()``
        (quarterly walk-forward, Sharpe ratio scoring).
    score_func : callable or None
        Custom per-portfolio scorer, required when
        ``config.scorer_config.ratio_measure`` is ``None``.  Forwarded to
        :func:`~optimizer.scoring.build_scorer`.
    benchmark_returns : pd.Series or None
        Benchmark return series, required when the scorer config selects
        the Information Ratio.  Forwarded to
        :func:`~optimizer.scoring.build_scorer`.

    Returns
    -------
    GridSearchCV
        A fitted-ready grid search estimator.
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

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=config.n_jobs,
        refit=config.refit,
        verbose=config.verbose,
        error_score=config.error_score,
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
    """Build a :class:`RandomizedSearchCV` with temporal cross-validation.

    Parameters
    ----------
    estimator : BaseEstimator
        The skfolio optimiser or pipeline to tune.
    param_distributions : dict
        Parameter distributions.  Values may be lists (discrete)
        or ``scipy.stats`` distributions (continuous, e.g.
        ``scipy.stats.loguniform(0.01, 1)``).
    config : RandomizedSearchConfig or None
        Tuning configuration.  Defaults to
        ``RandomizedSearchConfig()`` (50 iterations, quarterly
        walk-forward, Sharpe ratio scoring).
    score_func : callable or None
        Custom per-portfolio scorer, forwarded to
        :func:`~optimizer.scoring.build_scorer`.
    benchmark_returns : pd.Series or None
        Benchmark return series for Information-Ratio scoring, forwarded
        to :func:`~optimizer.scoring.build_scorer`.

    Returns
    -------
    RandomizedSearchCV
        A fitted-ready randomised search estimator.
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

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=config.n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=config.n_jobs,
        refit=config.refit,
        verbose=config.verbose,
        error_score=config.error_score,
        random_state=config.random_state,
        return_train_score=config.return_train_score,
    )


def search_results_dataframe(
    search: GridSearchCV | RandomizedSearchCV,
) -> pd.DataFrame:
    """Return a tidy, rank-sorted view of a fitted search's ``cv_results_``.

    Convenience wrapper that materialises ``search.cv_results_`` into a
    :class:`pandas.DataFrame` sorted by ``rank_test_score`` (best first),
    so the outcome of a hyperparameter sweep can be inspected or logged
    without hand-indexing the raw results dict.

    Parameters
    ----------
    search : GridSearchCV or RandomizedSearchCV
        A **fitted** search estimator (``.fit`` already called).

    Returns
    -------
    pd.DataFrame
        ``cv_results_`` as a DataFrame, ascending by ``rank_test_score``
        when that column is present (it is absent only for degenerate
        single-candidate multimetric setups).

    Raises
    ------
    AttributeError
        If *search* has not been fitted (no ``cv_results_``).
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
