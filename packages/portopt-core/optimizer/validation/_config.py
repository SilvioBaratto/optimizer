"""Configuration for model selection and cross-validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WalkForwardConfig:
    """Immutable configuration for :class:`skfolio.model_selection.WalkForward`.

    Walk-forward backtesting partitions time series into successive
    train/test windows that respect the causal arrow of time.

    **Purging**: A ``purged_size`` gap is excised between the end of the
    training window and the start of the test window.  Without this buffer,
    autocorrelated returns (volatility clustering, momentum) can leak
    information from training observations into the first test observations,
    inflating out-of-sample scores.  For daily equity returns a purge of
    21 observations (one trading month) is the standard minimum; increase
    it to match the longest look-back window used by any feature in the
    estimator pipeline.

    **Calendar frequencies** (skfolio 1.0): set ``freq`` (a pandas offset
    alias such as ``"MS"`` or ``"WOM-3FRI"``) to measure ``test_size`` /
    ``train_size`` in calendar *periods* instead of raw observations, so
    rebalancing lands on real trading-calendar dates rather than a fixed
    observation count.  When ``freq`` is set the input ``X`` must carry a
    ``DatetimeIndex``.

    Parameters
    ----------
    test_size : int
        Number of observations in each test window (or number of ``freq``
        periods when ``freq`` is set).
    train_size : int
        Number of observations in each training window (or number of
        ``freq`` periods when ``freq`` is set).  When ``expand_train`` is
        ``True``, this is the *initial* training window size.
    purged_size : int
        Number of observations purged between the end of the training
        window and the start of the test window to prevent look-ahead
        bias from autocorrelated returns.  Defaults to 5 (one trading
        week).  Presets use 21 (one trading month).
    expend_train : bool
        When ``True``, the training window expands as new data arrives
        (expanding window).  When ``False``, the training window rolls
        forward (rolling window).  ``expand_train`` (the skfolio 1.0
        spelling) is available as a read-only alias.
    reduce_test : bool
        When ``True``, the last test window may be shorter than
        ``test_size`` to avoid discarding data.
    freq : str or None
        Optional pandas frequency/offset alias (e.g. ``"MS"``,
        ``"QS"``, ``"WOM-3FRI"``).  When set, ``test_size`` and
        ``train_size`` count ``freq`` periods and ``X`` must have a
        ``DatetimeIndex``.  ``None`` (default) counts raw observations.
    freq_offset : str or None
        Optional pandas offset alias applied to shift each ``freq``
        period boundary (e.g. ``"2D"``).  Only used when ``freq`` is set;
        parsed with :func:`pandas.tseries.frequencies.to_offset`.
    previous : bool
        Only used when ``freq`` is set.  When ``True`` and a period
        boundary is not present in the ``DatetimeIndex``, the previous
        observation is used; otherwise the next observation is used.
    """

    test_size: int = 63
    train_size: int = 252
    purged_size: int = 5
    expend_train: bool = False
    reduce_test: bool = False
    freq: str | None = None
    freq_offset: str | None = None
    previous: bool = False

    def __post_init__(self) -> None:
        if self.test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {self.test_size}")
        if self.train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {self.train_size}")
        if self.purged_size < 0:
            raise ValueError(f"purged_size must be >= 0, got {self.purged_size}")
        if self.freq_offset is not None and self.freq is None:
            raise ValueError("freq_offset requires freq to be set")

    @property
    def expand_train(self) -> bool:
        """Correct-spelling read alias for :attr:`expend_train` (skfolio 1.0)."""
        return self.expend_train

    @classmethod
    def for_monthly_calendar(cls) -> WalkForwardConfig:
        """Calendar-based monthly rebalancing with a 12-month training window.

        Rebalances on the first trading day of each month (``freq="MS"``)
        with a one-month test window and a twelve-month rolling training
        window, both counted in calendar periods.  Requires a
        ``DatetimeIndex`` on the input returns.
        """
        return cls(test_size=1, train_size=12, purged_size=0, freq="MS")

    @classmethod
    def for_monthly_rolling(cls) -> WalkForwardConfig:
        """Monthly test windows with one-year rolling training.

        Uses ``purged_size=21`` (one trading month) to eliminate
        autocorrelation leakage at the train/test boundary.
        """
        return cls(test_size=21, train_size=252, purged_size=21)

    @classmethod
    def for_quarterly_rolling(cls) -> WalkForwardConfig:
        """Quarterly test windows with one-year rolling training.

        Uses ``purged_size=21`` (one trading month) to eliminate
        autocorrelation leakage at the train/test boundary.
        """
        return cls(test_size=63, train_size=252, purged_size=21)

    @classmethod
    def for_quarterly_expanding(cls) -> WalkForwardConfig:
        """Quarterly test windows with expanding training.

        Uses ``purged_size=21`` (one trading month) to eliminate
        autocorrelation leakage at the train/test boundary.
        """
        return cls(test_size=63, train_size=252, expend_train=True, purged_size=21)


@dataclass(frozen=True)
class CPCVConfig:
    """Configuration for :class:`skfolio.model_selection.CombinatorialPurgedCV`.

    Generates a population of backtest paths from all combinatorial
    selections of test folds, with purging and embargoing to prevent
    information leakage.

    Parameters
    ----------
    n_folds : int
        Number of non-overlapping temporal blocks.
    n_test_folds : int
        Number of blocks assigned to the test set in each combination.
    purged_size : int
        Number of observations excised on each side of the
        train-test boundary.
    embargo_size : int
        Number of observations embargoed immediately following
        each test block to avoid autocorrelation contamination.
    """

    n_folds: int = 10
    n_test_folds: int = 8
    purged_size: int = 0
    embargo_size: int = 0

    def __post_init__(self) -> None:
        if self.n_folds < 3:
            raise ValueError(f"n_folds must be >= 3, got {self.n_folds}")
        if not (1 <= self.n_test_folds < self.n_folds):
            raise ValueError(
                f"n_test_folds must satisfy 1 <= n_test_folds < n_folds "
                f"({self.n_folds}), got {self.n_test_folds}"
            )
        if self.purged_size < 0:
            raise ValueError(f"purged_size must be >= 0, got {self.purged_size}")
        if self.embargo_size < 0:
            raise ValueError(f"embargo_size must be >= 0, got {self.embargo_size}")

    @classmethod
    def for_statistical_testing(cls) -> CPCVConfig:
        """High-path-count configuration for significance testing.

        Uses C(12, 2) = 66 paths with 10 training folds per split,
        providing high statistical power for backtest overfitting tests.
        """
        return cls(n_folds=12, n_test_folds=2)

    @classmethod
    def for_small_sample(cls) -> CPCVConfig:
        """Fewer folds for shorter time series."""
        return cls(n_folds=6, n_test_folds=2)


@dataclass(frozen=True)
class MultipleRandomizedCVConfig:
    """Configuration for :class:`skfolio.model_selection.MultipleRandomizedCV`.

    Dual randomisation across temporal windows and asset subsets
    to test robustness of the strategy to both dimensions.

    Parameters
    ----------
    walk_forward_config : WalkForwardConfig
        Inner walk-forward configuration for temporal splitting.
    n_subsamples : int
        Number of random trials.
    asset_subset_size : int
        Number of assets drawn per trial.
    window_size : int or None
        Length of the random temporal window drawn per trial.
        ``None`` uses the full sample.
    random_state : int or None
        Seed for reproducibility.
    """

    walk_forward_config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    n_subsamples: int = 10
    asset_subset_size: int = 10
    window_size: int | None = None
    random_state: int | None = None

    def __post_init__(self) -> None:
        if self.n_subsamples < 1:
            raise ValueError(f"n_subsamples must be >= 1, got {self.n_subsamples}")
        if self.asset_subset_size < 1:
            raise ValueError(
                f"asset_subset_size must be >= 1, got {self.asset_subset_size}"
            )
        if self.window_size is not None and self.window_size < 1:
            raise ValueError(
                f"window_size must be >= 1 or None, got {self.window_size}"
            )

    @classmethod
    def for_robustness_check(
        cls,
        n_subsamples: int = 20,
        asset_subset_size: int = 10,
    ) -> MultipleRandomizedCVConfig:
        """Standard robustness check with 20 trials."""
        return cls(
            n_subsamples=n_subsamples,
            asset_subset_size=asset_subset_size,
            random_state=42,
        )
