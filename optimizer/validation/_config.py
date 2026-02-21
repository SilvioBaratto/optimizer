"""Configuration for model selection and cross-validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WalkForwardConfig:
    """Immutable configuration for :class:`skfolio.model_selection.WalkForward`.

    Walk-forward backtesting partitions time series into successive
    train/test windows that respect the causal arrow of time.

    Parameters
    ----------
    test_size : int
        Number of observations in each test window.
    train_size : int
        Number of observations in each training window.  When
        ``expend_train`` is ``True``, this is the *initial* training
        window size.
    purged_size : int
        Number of observations purged between train and test windows
        to prevent look-ahead bias.
    expend_train : bool
        When ``True``, the training window expands as new data arrives
        (expanding window).  When ``False``, the training window rolls
        forward (rolling window).
    reduce_test : bool
        When ``True``, the last test window may be shorter than
        ``test_size`` to avoid discarding data.
    """

    test_size: int = 63
    train_size: int = 252
    purged_size: int = 0
    expend_train: bool = False
    reduce_test: bool = False

    @classmethod
    def for_monthly_rolling(cls) -> WalkForwardConfig:
        """Monthly test windows with one-year rolling training."""
        return cls(test_size=21, train_size=252)

    @classmethod
    def for_quarterly_rolling(cls) -> WalkForwardConfig:
        """Quarterly test windows with one-year rolling training."""
        return cls(test_size=63, train_size=252)

    @classmethod
    def for_quarterly_expanding(cls) -> WalkForwardConfig:
        """Quarterly test windows with expanding training."""
        return cls(test_size=63, train_size=252, expend_train=True)


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

    @classmethod
    def for_statistical_testing(cls) -> CPCVConfig:
        """High-path-count configuration for significance testing."""
        return cls(n_folds=10, n_test_folds=8)

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
