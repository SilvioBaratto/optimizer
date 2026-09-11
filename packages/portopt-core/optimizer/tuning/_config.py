"""Configuration for hyperparameter tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizer.scoring._config import ScorerConfig
    from optimizer.validation._config import WalkForwardConfig


# Cross-module defaults are resolved lazily inside these factories rather than
# imported at module top level.  ``tuning`` sits on a legitimate import path
# (scoring -> optimization -> pipeline -> tuning -> scoring); deferring the
# imports keeps this config module import-order-independent and free of that
# cycle while still yielding real ``ScorerConfig`` / ``WalkForwardConfig``
# defaults at instance-construction time.
def _default_scorer_config() -> ScorerConfig:
    from optimizer.scoring._config import ScorerConfig

    return ScorerConfig()


def _default_walk_forward_config() -> WalkForwardConfig:
    from optimizer.validation._config import WalkForwardConfig

    return WalkForwardConfig()


def _validate_error_score(error_score: float | str) -> None:
    """Validate the ``error_score`` field.

    sklearn accepts either the string ``"raise"`` or a numeric score
    (typically ``nan``).  Any other string is rejected eagerly so a typo
    surfaces at config-construction time rather than deep inside a fit.
    """
    if isinstance(error_score, str):
        if error_score != "raise":
            raise ValueError(
                f"error_score must be 'raise' or a float, got {error_score!r}"
            )
    elif not isinstance(error_score, (int, float)):
        raise ValueError(
            f"error_score must be 'raise' or a float, got {type(error_score).__name__}"
        )


@dataclass(frozen=True)
class GridSearchConfig:
    """Immutable configuration for :class:`sklearn.model_selection.GridSearchCV`.

    Enforces temporal cross-validation by default (walk-forward)
    to prevent look-ahead bias in financial time series.

    Parameters
    ----------
    cv_config : WalkForwardConfig
        Temporal cross-validation configuration.  Defaults to
        quarterly rolling with one-year training window.
    scorer_config : ScorerConfig
        Scoring function configuration.  Defaults to Sharpe ratio.
    n_jobs : int or None
        Number of parallel jobs.  ``-1`` uses all cores.
    return_train_score : bool
        Whether to compute training scores (increases runtime).
    error_score : float or str
        Score assigned to a parameter candidate when fitting a fold
        raises.  Defaults to ``float("nan")`` (sklearn default): a
        candidate whose optimizer fails to solve on some folds is scored
        ``nan`` for those folds and demoted rather than aborting the whole
        search.  This is the search-level analogue of skfolio 1.0's
        per-fold resilience layer (``raise_on_failure=False`` /
        ``fallback=``).  Set to ``"raise"`` to fail fast instead.
    refit : bool
        Whether to refit the best estimator on the whole dataset after the
        search.  ``True`` (default) exposes ``best_estimator_``.
    verbose : int
        Verbosity level forwarded to sklearn.
    """

    cv_config: WalkForwardConfig = field(default_factory=_default_walk_forward_config)
    scorer_config: ScorerConfig = field(default_factory=_default_scorer_config)
    n_jobs: int | None = None
    return_train_score: bool = False
    error_score: float | str = math.nan
    refit: bool = True
    verbose: int = 0

    def __post_init__(self) -> None:
        _validate_error_score(self.error_score)

    @classmethod
    def for_quick_search(cls) -> GridSearchConfig:
        """Fast grid search with monthly windows."""
        from optimizer.validation._config import WalkForwardConfig

        return cls(
            cv_config=WalkForwardConfig.for_monthly_rolling(),
            n_jobs=-1,
        )

    @classmethod
    def for_thorough_search(cls) -> GridSearchConfig:
        """Thorough grid search with quarterly expanding windows."""
        from optimizer.validation._config import WalkForwardConfig

        return cls(
            cv_config=WalkForwardConfig.for_quarterly_expanding(),
            n_jobs=-1,
            return_train_score=True,
        )


@dataclass(frozen=True)
class RandomizedSearchConfig:
    """Immutable configuration for :class:`sklearn.model_selection.RandomizedSearchCV`.

    Samples parameter configurations from specified distributions
    rather than exhaustive grid enumeration.  Enforces temporal
    cross-validation by default.

    Parameters
    ----------
    n_iter : int
        Number of random parameter samples.
    cv_config : WalkForwardConfig
        Temporal cross-validation configuration.
    scorer_config : ScorerConfig
        Scoring function configuration.
    n_jobs : int or None
        Number of parallel jobs.
    random_state : int or None
        Seed for reproducibility.
    return_train_score : bool
        Whether to compute training scores.
    error_score : float or str
        Score assigned to a parameter candidate when fitting a fold
        raises.  Defaults to ``float("nan")``; set to ``"raise"`` to fail
        fast.  See :class:`GridSearchConfig` for the rationale.
    refit : bool
        Whether to refit the best estimator on the whole dataset.
    verbose : int
        Verbosity level forwarded to sklearn.
    """

    n_iter: int = 50
    cv_config: WalkForwardConfig = field(default_factory=_default_walk_forward_config)
    scorer_config: ScorerConfig = field(default_factory=_default_scorer_config)
    n_jobs: int | None = None
    random_state: int | None = None
    return_train_score: bool = False
    error_score: float | str = math.nan
    refit: bool = True
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}")
        _validate_error_score(self.error_score)

    @classmethod
    def for_quick_search(cls, n_iter: int = 20) -> RandomizedSearchConfig:
        """Fast randomised search with few iterations."""
        from optimizer.validation._config import WalkForwardConfig

        return cls(
            n_iter=n_iter,
            cv_config=WalkForwardConfig.for_monthly_rolling(),
            n_jobs=-1,
            random_state=42,
        )

    @classmethod
    def for_thorough_search(cls, n_iter: int = 100) -> RandomizedSearchConfig:
        """Thorough randomised search with many iterations."""
        from optimizer.validation._config import WalkForwardConfig

        return cls(
            n_iter=n_iter,
            cv_config=WalkForwardConfig.for_quarterly_expanding(),
            n_jobs=-1,
            random_state=42,
            return_train_score=True,
        )
