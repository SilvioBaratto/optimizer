"""Tests for rebalancing logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.rebalancing import (
    CalendarRebalancingConfig,
    HybridRebalancingConfig,
    ThresholdRebalancingConfig,
    compute_drifted_weights,
    compute_rebalancing_cost,
    compute_turnover,
    should_rebalance,
    should_rebalance_hybrid,
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


# ---------------------------------------------------------------------------
# Shared helpers for hybrid tests
# ---------------------------------------------------------------------------

_BREACH = np.array([0.56, 0.24, 0.20])   # drifted by 6pp → breaches 5pp threshold
_TARGET = np.array([0.50, 0.30, 0.20])
_NO_BREACH = np.array([0.53, 0.27, 0.20])  # drifted by 3pp → below threshold
_MONTHLY_CFG = HybridRebalancingConfig.for_monthly_with_5pct_threshold()  # 21 bdays


def _review_date(base: pd.Timestamp, bdays: int) -> pd.Timestamp:
    """Return a date exactly ``bdays`` business days after ``base``."""
    return cast_ts(pd.bdate_range(base, periods=bdays + 1)[-1])


def cast_ts(x: object) -> pd.Timestamp:
    return pd.Timestamp(x)  # type: ignore[arg-type]


_LAST_REVIEW = pd.Timestamp("2024-01-02")


class TestShouldRebalanceHybrid:
    # -- Acceptance criterion 1: calendar date + breach → True ----------------

    def test_review_date_with_breach_returns_true(self) -> None:
        """At a calendar review date with breach, rebalancing is triggered."""
        current = _review_date(_LAST_REVIEW, 21)  # exactly 21 bdays later
        assert should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        ) is True

    # -- Acceptance criterion 2: calendar date + no breach → False ------------

    def test_review_date_no_breach_returns_false(self) -> None:
        """At a calendar review date with no drift breach, no rebalancing."""
        current = _review_date(_LAST_REVIEW, 21)
        assert should_rebalance_hybrid(
            _NO_BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        ) is False

    # -- Acceptance criterion 3: mid-calendar + breach → False ----------------

    def test_mid_calendar_breach_returns_false(self) -> None:
        """Between review dates, always returns False regardless of drift."""
        current = _review_date(_LAST_REVIEW, 10)  # only 10 bdays elapsed
        assert should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        ) is False

    # -- Edge cases -----------------------------------------------------------

    def test_exactly_at_threshold_bdays(self) -> None:
        """Exactly trading_days elapsed is treated as a review date."""
        current = _review_date(_LAST_REVIEW, 21)
        result = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert isinstance(result, bool)
        assert result is True

    def test_one_day_before_review(self) -> None:
        """One business day before the review interval → always False."""
        current = _review_date(_LAST_REVIEW, 20)
        assert should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        ) is False

    def test_past_due_review_with_breach(self) -> None:
        """Overdue review (more than trading_days elapsed) with breach → True."""
        current = _review_date(_LAST_REVIEW, 42)  # 2 months elapsed
        assert should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        ) is True

    def test_relative_threshold_variant(self) -> None:
        """Hybrid with relative threshold behaves consistently."""
        cfg = HybridRebalancingConfig(
            calendar=CalendarRebalancingConfig.for_quarterly(),
            threshold=ThresholdRebalancingConfig.for_relative(threshold=0.25),
        )
        # asset 1 relative drift: |0.38-0.30|/0.30 = 0.267 > 0.25 → breach
        breaching = np.array([0.50, 0.38, 0.12])
        current = _review_date(_LAST_REVIEW, 63)  # quarterly = 63 bdays
        assert should_rebalance_hybrid(
            breaching, _TARGET, cfg, current, _LAST_REVIEW
        ) is True
