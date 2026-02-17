"""Factory functions for building hyperparameter search estimators."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from optimizer.scoring._factory import build_scorer
from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig
from optimizer.validation._factory import build_walk_forward


def build_grid_search_cv(
    estimator: Any,
    param_grid: dict[str, list[Any]],
    config: GridSearchConfig | None = None,
) -> GridSearchCV:
    """Build a :class:`GridSearchCV` with temporal cross-validation.

    Parameters
    ----------
    estimator : BaseEstimator
        The skfolio optimiser or pipeline to tune.
    param_grid : dict
        Parameter grid.  Keys use sklearn double-underscore
        notation for nested estimators (e.g.
        ``"prior_estimator__mu_estimator__alpha"``).
    config : GridSearchConfig or None
        Tuning configuration.  Defaults to ``GridSearchConfig()``
        (quarterly walk-forward, Sharpe ratio scoring).

    Returns
    -------
    GridSearchCV
        A fitted-ready grid search estimator.
    """
    if config is None:
        config = GridSearchConfig()

    cv = build_walk_forward(config.cv_config)
    scoring = build_scorer(config.scorer_config)

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=config.n_jobs,
        return_train_score=config.return_train_score,
    )


def build_randomized_search_cv(
    estimator: Any,
    param_distributions: dict[str, Any],
    config: RandomizedSearchConfig | None = None,
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

    Returns
    -------
    RandomizedSearchCV
        A fitted-ready randomised search estimator.
    """
    if config is None:
        config = RandomizedSearchConfig()

    cv = build_walk_forward(config.cv_config)
    scoring = build_scorer(config.scorer_config)

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=config.n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        return_train_score=config.return_train_score,
    )
