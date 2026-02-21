"""Tests for validation configs."""

from __future__ import annotations

import pytest

from optimizer.validation import (
    CPCVConfig,
    MultipleRandomizedCVConfig,
    WalkForwardConfig,
)


class TestWalkForwardConfig:
    def test_defaults(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.test_size == 63
        assert cfg.train_size == 252
        assert cfg.purged_size == 0
        assert cfg.expend_train is False
        assert cfg.reduce_test is False

    def test_frozen(self) -> None:
        cfg = WalkForwardConfig()
        with pytest.raises(AttributeError):
            cfg.test_size = 42  # type: ignore[misc]

    def test_for_monthly_rolling(self) -> None:
        cfg = WalkForwardConfig.for_monthly_rolling()
        assert cfg.test_size == 21
        assert cfg.train_size == 252
        assert cfg.expend_train is False

    def test_for_quarterly_rolling(self) -> None:
        cfg = WalkForwardConfig.for_quarterly_rolling()
        assert cfg.test_size == 63
        assert cfg.train_size == 252

    def test_for_quarterly_expanding(self) -> None:
        cfg = WalkForwardConfig.for_quarterly_expanding()
        assert cfg.test_size == 63
        assert cfg.expend_train is True

    def test_custom_params(self) -> None:
        cfg = WalkForwardConfig(
            test_size=42,
            train_size=504,
            purged_size=5,
            expend_train=True,
            reduce_test=True,
        )
        assert cfg.test_size == 42
        assert cfg.train_size == 504
        assert cfg.purged_size == 5


class TestCPCVConfig:
    def test_defaults(self) -> None:
        cfg = CPCVConfig()
        assert cfg.n_folds == 10
        assert cfg.n_test_folds == 8
        assert cfg.purged_size == 0
        assert cfg.embargo_size == 0

    def test_frozen(self) -> None:
        cfg = CPCVConfig()
        with pytest.raises(AttributeError):
            cfg.n_folds = 5  # type: ignore[misc]

    def test_for_statistical_testing(self) -> None:
        cfg = CPCVConfig.for_statistical_testing()
        assert cfg.n_folds == 12
        assert cfg.n_test_folds == 2

    def test_for_small_sample(self) -> None:
        cfg = CPCVConfig.for_small_sample()
        assert cfg.n_folds == 6
        assert cfg.n_test_folds == 2

    def test_custom_params(self) -> None:
        cfg = CPCVConfig(
            n_folds=8,
            n_test_folds=3,
            purged_size=10,
            embargo_size=5,
        )
        assert cfg.n_folds == 8
        assert cfg.purged_size == 10
        assert cfg.embargo_size == 5


class TestMultipleRandomizedCVConfig:
    def test_defaults(self) -> None:
        cfg = MultipleRandomizedCVConfig()
        assert cfg.n_subsamples == 10
        assert cfg.asset_subset_size == 10
        assert cfg.window_size is None
        assert cfg.random_state is None
        assert isinstance(cfg.walk_forward_config, WalkForwardConfig)

    def test_frozen(self) -> None:
        cfg = MultipleRandomizedCVConfig()
        with pytest.raises(AttributeError):
            cfg.n_subsamples = 5  # type: ignore[misc]

    def test_for_robustness_check(self) -> None:
        cfg = MultipleRandomizedCVConfig.for_robustness_check(
            n_subsamples=30,
            asset_subset_size=15,
        )
        assert cfg.n_subsamples == 30
        assert cfg.asset_subset_size == 15
        assert cfg.random_state == 42

    def test_embedded_walk_forward(self) -> None:
        wf = WalkForwardConfig.for_monthly_rolling()
        cfg = MultipleRandomizedCVConfig(walk_forward_config=wf)
        assert cfg.walk_forward_config.test_size == 21
