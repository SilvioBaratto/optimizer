"""Tests for BenchmarkTracker factory and configuration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.optimization import BenchmarkTracker

from optimizer.optimization import (
    BenchmarkTrackerConfig,
    build_benchmark_tracker,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


@pytest.fixture(scope="module")
def benchmark(returns):
    """Equal-weight benchmark (mean across the universe)."""
    return returns.mean(axis=1).rename("benchmark")


class TestBenchmarkTrackerConfig:
    def test_when_default_then_no_prior_config(self) -> None:
        assert BenchmarkTrackerConfig().prior_config is None

    def test_when_default_then_min_weights_zero(self) -> None:
        assert BenchmarkTrackerConfig().min_weights == 0.0

    def test_when_constructed_then_frozen(self) -> None:
        cfg = BenchmarkTrackerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.min_weights = 0.05  # type: ignore[misc]


class TestPresets:
    def test_when_for_te_target_then_l2_coef_set(self) -> None:
        cfg = BenchmarkTrackerConfig.for_te_target(target=0.01)
        assert cfg.l2_coef > 0.0

    def test_when_for_information_ratio_then_returns_config(self) -> None:
        cfg = BenchmarkTrackerConfig.for_information_ratio()
        assert isinstance(cfg, BenchmarkTrackerConfig)


class TestBuildBenchmarkTracker:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_benchmark_tracker(BenchmarkTrackerConfig())
        assert isinstance(est, BenchmarkTracker)

    def test_when_min_weights_set_then_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(min_weights=0.01)
        est = build_benchmark_tracker(cfg)
        assert est.min_weights == 0.01

    def test_when_max_weights_set_then_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(max_weights=0.10)
        est = build_benchmark_tracker(cfg)
        assert est.max_weights == 0.10

    def test_when_transaction_costs_set_then_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(transaction_costs=0.001)
        est = build_benchmark_tracker(cfg)
        assert est.transaction_costs == 0.001

    def test_when_explicit_prior_estimator_then_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        prior = EmpiricalPrior()
        est = build_benchmark_tracker(BenchmarkTrackerConfig(), prior_estimator=prior)
        assert est.prior_estimator is prior

    def test_when_prior_config_then_inner_prior_built(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        cfg = BenchmarkTrackerConfig(prior_config=MomentEstimationConfig())
        est = build_benchmark_tracker(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_benchmark_tracker(BenchmarkTrackerConfig(), raise_on_failure=False)
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_benchmark_tracker(None)
        assert isinstance(est, BenchmarkTracker)


class TestIntegration:
    def test_when_fit_with_y_benchmark_then_weights_returned(
        self, returns, benchmark
    ) -> None:
        est = build_benchmark_tracker(BenchmarkTrackerConfig())
        est.fit(returns, benchmark)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_fit_without_y_then_raises(self, returns) -> None:
        est = build_benchmark_tracker(BenchmarkTrackerConfig())
        with pytest.raises(ValueError, match="benchmark"):
            est.fit(returns)

    def test_when_for_te_target_then_fits(self, returns, benchmark) -> None:
        est = build_benchmark_tracker(BenchmarkTrackerConfig.for_te_target(target=0.01))
        est.fit(returns, benchmark)
        assert est.weights_.shape == (returns.shape[1],)

    def test_when_for_information_ratio_then_fits(self, returns, benchmark) -> None:
        est = build_benchmark_tracker(BenchmarkTrackerConfig.for_information_ratio())
        est.fit(returns, benchmark)
        assert est.weights_.shape == (returns.shape[1],)
