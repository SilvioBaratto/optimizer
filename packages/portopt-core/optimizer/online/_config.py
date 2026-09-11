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

    Walk-forward semantics mirror :class:`WalkForward`: after an initial
    ``warmup_size`` observation window, the estimator predicts the next
    ``test_size`` observations, then ``partial_fit`` folds them in and the
    window advances. ``purged_size`` excises a gap between the fitted window
    and the test window to block autocorrelation leakage. Calendar
    frequencies (``freq``/``freq_offset``/``previous``) rebalance on real
    trading-calendar dates instead of a fixed observation count and require
    the input ``X`` to carry a ``DatetimeIndex``.

    Parameters
    ----------
    warmup_size : int
        Number of initial observations consumed by the first
        ``partial_fit`` call (or number of ``freq`` periods when ``freq``
        is set). skfolio default ``252``.
    test_size : int
        Number of observations advanced per rebalance step (or number of
        ``freq`` periods when ``freq`` is set). skfolio default ``1``.
    purged_size : int
        Number of observations excised between the fitted window and the
        test window to prevent look-ahead bias from autocorrelated
        returns. Defaults to ``0``.
    freq : str or None
        Optional pandas frequency/offset alias (e.g. ``"MS"``, ``"QS"``,
        ``"WOM-3FRI"``). When set, ``warmup_size`` and ``test_size`` count
        ``freq`` periods and ``X`` must have a ``DatetimeIndex``. ``None``
        (default) counts raw observations.
    freq_offset : str or None
        Optional pandas offset alias shifting each ``freq`` period boundary
        (e.g. ``"2D"``). Only used when ``freq`` is set; parsed with
        :func:`pandas.tseries.frequencies.to_offset`.
    previous : bool
        Only used when ``freq`` is set. When ``True`` and a period boundary
        is absent from the ``DatetimeIndex``, the previous observation is
        used; otherwise the next observation is used.
    reduce_test : bool
        When ``True``, the final test window may be shorter than
        ``test_size`` to avoid discarding trailing observations.
    n_jobs : int or None
        Parallelism hint forwarded to :class:`OnlineGridSearch` /
        :class:`OnlineRandomizedSearch`. Not consumed by
        ``online_predict`` / ``online_score`` themselves.
    verbose : bool
        Verbosity hint forwarded to the online search wrappers.
    """

    warmup_size: int = 252
    test_size: int = 1
    purged_size: int = 0
    freq: str | None = None
    freq_offset: str | None = None
    previous: bool = False
    reduce_test: bool = False
    n_jobs: int | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.warmup_size < 1:
            raise ValueError(f"warmup_size must be >= 1, got {self.warmup_size}")
        if self.test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {self.test_size}")
        if self.purged_size < 0:
            raise ValueError(f"purged_size must be >= 0, got {self.purged_size}")
        if self.freq_offset is not None and self.freq is None:
            raise ValueError("freq_offset requires freq to be set")

    @classmethod
    def for_daily_rebalance(cls, warmup_size: int = 252) -> OnlinePredictConfig:
        """Daily (per-observation) rebalancing after a one-year warmup."""
        return cls(warmup_size=warmup_size, test_size=1)

    @classmethod
    def for_calendar_monthly(cls, warmup_size: int = 12) -> OnlinePredictConfig:
        """Calendar monthly rebalancing (``freq="MS"``).

        Counts ``warmup_size`` and ``test_size`` in calendar months and
        rebalances on the first trading day of each month. Requires a
        ``DatetimeIndex`` on the input returns.
        """
        return cls(warmup_size=warmup_size, test_size=1, freq="MS")


@dataclass(frozen=True)
class OnlineGridSearchConfig:
    """Immutable configuration for :class:`OnlineGridSearch`.

    Online instances are NOT thread-safe — caller is responsible for
    using a fresh instance per thread.

    The ``scoring`` for online portfolio search is resolved from
    ``base.scorer_config`` to a skfolio ``BaseMeasure`` (e.g. Sharpe),
    because online portfolio evaluation rejects ``make_scorer`` objects.
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


@dataclass(frozen=True)
class CovarianceForecastConfig:
    """Immutable configuration for covariance-forecast evaluation.

    Diagnoses a covariance estimator's out-of-sample calibration
    *independently of any optimizer* (Mahalanobis / diagonal calibration
    ratios, portfolio standardized returns, QLIKE loss). Drives both the
    walk-forward evaluator (:func:`covariance_forecast_evaluation`, refit
    each split) and the online evaluator
    (:func:`online_covariance_forecast_evaluation`, ``partial_fit``-based).

    Parameters
    ----------
    train_size : int
        Length of the initial training / warmup window. For the online
        evaluator this is forwarded as ``warmup_size``. skfolio default
        ``252``.
    test_size : int
        Number of observations per forecast-evaluation step. Defaults to
        ``1``.
    expand_train : bool
        Walk-forward evaluator only: when ``True`` the training window
        expands, otherwise it rolls. Ignored by the online evaluator
        (``partial_fit`` is inherently cumulative).
    purged_size : int
        Number of observations excised between the training window and the
        forecast window. Defaults to ``0``.
    """

    train_size: int = 252
    test_size: int = 1
    expand_train: bool = False
    purged_size: int = 0

    def __post_init__(self) -> None:
        if self.train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {self.train_size}")
        if self.test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {self.test_size}")
        if self.purged_size < 0:
            raise ValueError(f"purged_size must be >= 0, got {self.purged_size}")
