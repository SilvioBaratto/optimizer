"""Tests for validation factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)

from optimizer.validation import (
    CPCVConfig,
    MultipleRandomizedCVConfig,
    WalkForwardConfig,
    build_cpcv,
    build_multiple_randomized_cv,
    build_walk_forward,
    compute_optimal_folds,
    run_cross_val,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 20 assets and 400 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 400, 20
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2022-01-01", periods=n_obs, freq="B"),
    )


class TestBuildWalkForward:
    def test_default_config(self) -> None:
        cv = build_walk_forward()
        assert isinstance(cv, WalkForward)

    def test_custom_config(self) -> None:
        cfg = WalkForwardConfig(test_size=42, train_size=126, purged_size=3)
        cv = build_walk_forward(cfg)
        assert isinstance(cv, WalkForward)

    def test_expanding_window(self) -> None:
        cfg = WalkForwardConfig.for_quarterly_expanding()
        cv = build_walk_forward(cfg)
        assert isinstance(cv, WalkForward)

    def test_splits_returns(self, returns_df: pd.DataFrame) -> None:
        cfg = WalkForwardConfig(test_size=21, train_size=100)
        cv = build_walk_forward(cfg)
        splits = list(cv.split(returns_df))
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(test_idx) <= 21
            assert len(train_idx) >= 100


class TestBuildCPCV:
    def test_default_config(self) -> None:
        cv = build_cpcv()
        assert isinstance(cv, CombinatorialPurgedCV)

    def test_custom_config(self) -> None:
        cfg = CPCVConfig(n_folds=6, n_test_folds=2, purged_size=5, embargo_size=3)
        cv = build_cpcv(cfg)
        assert isinstance(cv, CombinatorialPurgedCV)

    def test_splits_returns(self, returns_df: pd.DataFrame) -> None:
        cfg = CPCVConfig.for_small_sample()
        cv = build_cpcv(cfg)
        splits = list(cv.split(returns_df))
        assert len(splits) > 0


class TestBuildMultipleRandomizedCV:
    def test_default_config(self) -> None:
        cv = build_multiple_randomized_cv()
        assert isinstance(cv, MultipleRandomizedCV)

    def test_custom_config(self) -> None:
        cfg = MultipleRandomizedCVConfig(
            n_subsamples=5,
            asset_subset_size=8,
            random_state=123,
        )
        cv = build_multiple_randomized_cv(cfg)
        assert isinstance(cv, MultipleRandomizedCV)


class TestComputeOptimalFolds:
    def test_returns_tuple(self) -> None:
        result = compute_optimal_folds(
            n_observations=500,
            target_train_size=252,
            target_n_test_paths=10,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        n_folds, n_test_folds = result
        assert n_folds > 0
        assert n_test_folds > 0
        assert n_test_folds < n_folds


class TestRunCrossVal:
    def test_walk_forward_backtest(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = WalkForwardConfig(test_size=21, train_size=100)
        cv = build_walk_forward(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        assert hasattr(pred, "sharpe_ratio")

    def test_default_cv(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        pred = run_cross_val(EqualWeighted(), returns_df)
        assert hasattr(pred, "sharpe_ratio")
