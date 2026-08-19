"""Configuration for skfolio online learning workflows.

Online instances are NOT thread-safe. ``partial_fit`` accumulates
mutable state across calls and ``OnlineGridSearch`` mutates the
wrapped estimator in place. Callers running scheduled jobs in
multiple daemon threads (e.g. via ``optimizer/ingestion/`` background
services) MUST construct one instance per thread.

The ``GridSearchConfig`` / ``RandomizedSearchConfig`` defaults are
constructed lazily to avoid an import cycle through
``optimizer.optimization`` → ``optimizer.pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig


def _default_grid_search() -> Any:
    """Lazily construct the default :class:`GridSearchConfig`."""
    from optimizer.tuning._config import GridSearchConfig

    return GridSearchConfig()


def _default_randomized_search() -> Any:
    """Lazily construct the default :class:`RandomizedSearchConfig`."""
    from optimizer.tuning._config import RandomizedSearchConfig

    return RandomizedSearchConfig()


@dataclass(frozen=True)
class OnlinePredictConfig:
    """Immutable configuration for ``online_predict`` / ``online_score``.

    Online estimators are NOT thread-safe — caller is responsible for
    using a fresh instance per thread.

    Parameters
    ----------
    warmup_size : int
        Number of initial observations consumed by the first
        ``partial_fit`` call. skfolio default ``252``.
    n_jobs : int or None
        Parallelism hint passed to higher-level wrappers. Not
        consumed by ``online_predict`` / ``online_score`` themselves.
    verbose : bool
        Verbosity hint passed to higher-level wrappers.
    """

    warmup_size: int = 252
    n_jobs: int | None = None
    verbose: bool = False


@dataclass(frozen=True)
class OnlineGridSearchConfig:
    """Immutable configuration for :class:`OnlineGridSearch`.

    Online instances are NOT thread-safe — caller is responsible for
    using a fresh instance per thread.
    """

    base: GridSearchConfig = field(default_factory=_default_grid_search)
    online: OnlinePredictConfig = field(default_factory=OnlinePredictConfig)


@dataclass(frozen=True)
class OnlineRandomizedSearchConfig:
    """Immutable configuration for :class:`OnlineRandomizedSearch`.

    Online instances are NOT thread-safe — caller is responsible for
    using a fresh instance per thread.
    """

    base: RandomizedSearchConfig = field(default_factory=_default_randomized_search)
    online: OnlinePredictConfig = field(default_factory=OnlinePredictConfig)
