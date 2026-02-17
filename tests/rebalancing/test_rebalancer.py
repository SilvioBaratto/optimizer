"""Tests for rebalancing logic."""

from __future__ import annotations

import numpy as np
import pytest

from optimizer.rebalancing import (
    ThresholdRebalancingConfig,
    ThresholdType,
    compute_drifted_weights,
    compute_rebalancing_cost,
    compute_turnover,
    should_rebalance,
)


class TestComputeDriftedWeights:
    def test_no_returns(self) -> None:
        weights = np.array([0.5, 0.3, 0.2])
        returns = np.array([0.0, 0.0, 0.0])
        drifted = compute_drifted_weights(weights, returns)
        np.testing.assert_allclose(drifted, weights)

    def test_positive_returns(self) -> None:
        weights = np.array([0.5, 0.5])
        returns = np.array([0.10, 0.0])  # first asset up 10%
        drifted = compute_drifted_weights(weights, returns)
        # 0.5*1.1 = 0.55, 0.5*1.0 = 0.5, total = 1.05
        expected = np.array([0.55 / 1.05, 0.5 / 1.05])
        np.testing.assert_allclose(drifted, expected)

    def test_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(5))
        returns = rng.normal(0.01, 0.02, size=5)
        drifted = compute_drifted_weights(weights, returns)
        assert pytest.approx(drifted.sum(), abs=1e-10) == 1.0


class TestComputeTurnover:
    def test_no_change(self) -> None:
        w = np.array([0.5, 0.3, 0.2])
        assert compute_turnover(w, w) == pytest.approx(0.0)

    def test_full_rebalance(self) -> None:
        current = np.array([1.0, 0.0])
        target = np.array([0.0, 1.0])
        assert compute_turnover(current, target) == pytest.approx(1.0)

    def test_partial_rebalance(self) -> None:
        current = np.array([0.6, 0.4])
        target = np.array([0.5, 0.5])
        # |0.1| + |0.1| = 0.2 / 2 = 0.1
        assert compute_turnover(current, target) == pytest.approx(0.1)


class TestComputeRebalancingCost:
    def test_zero_cost(self) -> None:
        w = np.array([0.5, 0.5])
        cost = compute_rebalancing_cost(w, w, 0.001)
        assert cost == pytest.approx(0.0)

    def test_uniform_cost(self) -> None:
        current = np.array([0.6, 0.4])
        target = np.array([0.5, 0.5])
        cost = compute_rebalancing_cost(current, target, 0.01)
        # trades: |0.1| + |0.1| = 0.2, cost = 0.01 * 0.2 = 0.002
        assert cost == pytest.approx(0.002)

    def test_asset_specific_costs(self) -> None:
        current = np.array([0.6, 0.4])
        target = np.array([0.5, 0.5])
        costs = np.array([0.01, 0.02])
        cost = compute_rebalancing_cost(current, target, costs)
        # 0.01*0.1 + 0.02*0.1 = 0.003
        assert cost == pytest.approx(0.003)


class TestShouldRebalance:
    def test_no_drift(self) -> None:
        w = np.array([0.5, 0.3, 0.2])
        assert should_rebalance(w, w) is False

    def test_absolute_breach(self) -> None:
        current = np.array([0.56, 0.24, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)
        assert should_rebalance(current, target, cfg) is True

    def test_absolute_no_breach(self) -> None:
        current = np.array([0.54, 0.26, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)
        assert should_rebalance(current, target, cfg) is False

    def test_relative_breach(self) -> None:
        current = np.array([0.50, 0.38, 0.12])
        target = np.array([0.50, 0.30, 0.20])
        # asset 2: |0.38-0.30|/0.30 = 0.267 > 0.25
        cfg = ThresholdRebalancingConfig.for_relative(threshold=0.25)
        assert should_rebalance(current, target, cfg) is True

    def test_relative_no_breach(self) -> None:
        current = np.array([0.52, 0.28, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        cfg = ThresholdRebalancingConfig.for_relative(threshold=0.25)
        assert should_rebalance(current, target, cfg) is False

    def test_relative_zero_target_handled(self) -> None:
        current = np.array([0.5, 0.3, 0.2])
        target = np.array([0.5, 0.5, 0.0])
        cfg = ThresholdRebalancingConfig.for_relative(threshold=0.25)
        # zero-weight target should not cause division error
        result = should_rebalance(current, target, cfg)
        assert isinstance(result, bool)

    def test_default_config(self) -> None:
        current = np.array([0.56, 0.24, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        # default is absolute 5pp
        assert should_rebalance(current, target) is True
