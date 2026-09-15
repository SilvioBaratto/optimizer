"""Hyperparameter tuning with temporal cross-validation.

Wraps sklearn ``GridSearchCV`` and ``RandomizedSearchCV`` with temporal
cross-validation defaults that prevent look-ahead bias.

Causality guarantee
    The cross-validator is always a skfolio
    :class:`~skfolio.model_selection.WalkForward` built from
    ``config.cv_config``: it splits strictly in time order and exposes no
    ``shuffle`` knob, so a candidate is never scored on training-period data.
    ``RandomizedSearchConfig.random_state`` seeds only the hyperparameter
    *sampling* (sklearn ``ParameterSampler``), never the data splitter, so it
    cannot reorder observations or leak future returns.

Single-regime limitation
    A search tunes hyperparameters of one estimator across all walk-forward
    folds; portfolio *constraints* (e.g. sector bands) are fixed attributes of
    that estimator and therefore constant for the whole run.  Walk-forward CV
    cannot vary constraints per fold -- a backtest is single-regime.

Data note
    Feed linear (simple) return DataFrames as ``X`` (see
    :func:`skfolio.preprocessing.prices_to_returns`); cast DB ``Decimal``
    columns to float upstream.  Ragged history leaves ``NaN`` that can make an
    optimiser fail on some folds -- the ``error_score=nan`` default demotes
    those candidates instead of aborting the whole search.
"""

from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig
from optimizer.tuning._factory import (
    build_grid_search_cv,
    build_randomized_search_cv,
    search_results_dataframe,
)

__all__ = [
    "GridSearchConfig",
    "RandomizedSearchConfig",
    "build_grid_search_cv",
    "build_randomized_search_cv",
    "search_results_dataframe",
]
