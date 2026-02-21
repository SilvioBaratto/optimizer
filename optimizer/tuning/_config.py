"""Configuration for hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass, field

from optimizer.scoring._config import ScorerConfig
from optimizer.validation._config import WalkForwardConfig


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
    """

    cv_config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    scorer_config: ScorerConfig = field(default_factory=ScorerConfig)
    n_jobs: int | None = None
    return_train_score: bool = False

    @classmethod
    def for_quick_search(cls) -> GridSearchConfig:
        """Fast grid search with monthly windows."""
        return cls(
            cv_config=WalkForwardConfig.for_monthly_rolling(),
            n_jobs=-1,
        )

    @classmethod
    def for_thorough_search(cls) -> GridSearchConfig:
        """Thorough grid search with quarterly expanding windows."""
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
    """

    n_iter: int = 50
    cv_config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    scorer_config: ScorerConfig = field(default_factory=ScorerConfig)
    n_jobs: int | None = None
    random_state: int | None = None
    return_train_score: bool = False

    @classmethod
    def for_quick_search(cls, n_iter: int = 20) -> RandomizedSearchConfig:
        """Fast randomised search with few iterations."""
        return cls(
            n_iter=n_iter,
            cv_config=WalkForwardConfig.for_monthly_rolling(),
            n_jobs=-1,
            random_state=42,
        )

    @classmethod
    def for_thorough_search(cls, n_iter: int = 100) -> RandomizedSearchConfig:
        """Thorough randomised search with many iterations."""
        return cls(
            n_iter=n_iter,
            cv_config=WalkForwardConfig.for_quarterly_expanding(),
            n_jobs=-1,
            random_state=42,
            return_train_score=True,
        )
