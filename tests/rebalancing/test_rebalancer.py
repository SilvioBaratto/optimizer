"""Tests for rebalancing logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.rebalancing import (
    CalendarRebalancingConfig,
    HybridRebalancingConfig,
    ThresholdRebalancingConfig,
    apply_no_trade_band,
    build_rebalancing_walk_forward,
    compute_drifted_weights,
    compute_rebalancing_cost,
    compute_turnover,
    drift_breach_mask,
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
        # zero-target with non-zero current weight must always trigger exit
        assert should_rebalance(current, target, cfg) is True

    def test_relative_zero_target_exit_always_triggers(self) -> None:
        """Issue #303: zero-target position with non-zero current must trigger exit."""
        current = np.array([0.33, 0.33, 0.34])
        target = np.array([0.50, 0.50, 0.0])  # third position being exited
        cfg = ThresholdRebalancingConfig.for_relative(threshold=0.05)
        assert should_rebalance(current, target, cfg) is True

    def test_relative_zero_target_zero_current_no_trigger(self) -> None:
        """Zero-target AND zero-current weight should not trigger spuriously."""
        current = np.array([0.50, 0.50, 0.0])
        target = np.array([0.50, 0.50, 0.0])
        cfg = ThresholdRebalancingConfig.for_relative(threshold=0.05)
        assert should_rebalance(current, target, cfg) is False

    def test_default_config(self) -> None:
        current = np.array([0.56, 0.24, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        # default is absolute 5pp
        assert should_rebalance(current, target) is True


class TestInputRobustness:
    def test_drifted_weights_accepts_lists(self) -> None:
        drifted = compute_drifted_weights([0.5, 0.5], [0.0, 0.0])
        np.testing.assert_allclose(drifted, [0.5, 0.5])

    def test_drifted_weights_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_drifted_weights([0.5, 0.5], [0.0, 0.0, 0.0])

    def test_drifted_weights_zero_total_returns_grown(self) -> None:
        # returns of -1 wipe the portfolio value to zero.
        out = compute_drifted_weights([0.5, 0.5], [-1.0, -1.0])
        np.testing.assert_allclose(out, [0.0, 0.0])

    def test_drifted_weights_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            compute_drifted_weights([[0.5, 0.5]], [[0.0, 0.0]])

    def test_turnover_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_turnover([0.5, 0.5], [1.0])

    def test_cost_scalar_ok(self) -> None:
        assert compute_rebalancing_cost([0.6, 0.4], [0.5, 0.5], 0.01) == pytest.approx(
            0.002
        )

    def test_cost_array_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_rebalancing_cost([0.6, 0.4], [0.5, 0.5], np.array([0.01]))

    def test_cost_2d_costs_raises(self) -> None:
        with pytest.raises(ValueError, match="scalar or 1-D"):
            compute_rebalancing_cost([0.6, 0.4], [0.5, 0.5], np.array([[0.01, 0.02]]))


class TestDriftBreachMask:
    def test_absolute_per_asset(self) -> None:
        current = np.array([0.56, 0.24, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)
        mask = drift_breach_mask(current, target, cfg)
        np.testing.assert_array_equal(mask, [True, True, False])

    def test_relative_exit_flagged(self) -> None:
        current = np.array([0.50, 0.50, 0.0])
        target = np.array([0.50, 0.0, 0.0])
        cfg = ThresholdRebalancingConfig.for_relative(threshold=0.25)
        mask = drift_breach_mask(current, target, cfg)
        # asset 1 held with zero target -> exit; asset 0/2 unchanged.
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_default_config_absolute_5pp(self) -> None:
        mask = drift_breach_mask([0.56, 0.24, 0.20], [0.50, 0.30, 0.20])
        assert mask.any()

    def test_mask_matches_should_rebalance(self) -> None:
        current = np.array([0.54, 0.26, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)
        assert bool(drift_breach_mask(current, target, cfg).any()) == should_rebalance(
            current, target, cfg
        )


class TestApplyNoTradeBand:
    def test_no_breach_returns_current(self) -> None:
        current = np.array([0.52, 0.28, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)
        out = apply_no_trade_band(current, target, cfg)
        np.testing.assert_allclose(out, current)

    def test_breaching_asset_snaps_to_target(self) -> None:
        current = np.array([0.60, 0.20, 0.20])
        target = np.array([0.50, 0.30, 0.20])
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)
        out = apply_no_trade_band(current, target, cfg)
        # both breaching assets snap to target (0.50, 0.30); asset 2 stays 0.20;
        # blend sums to 1 already so renormalisation is a no-op.
        np.testing.assert_allclose(out, [0.50, 0.30, 0.20])
        assert out.sum() == pytest.approx(1.0)

    def test_result_sums_to_one(self) -> None:
        current = np.array([0.70, 0.20, 0.10])
        target = np.array([0.40, 0.40, 0.20])
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.05)
        out = apply_no_trade_band(current, target, cfg)
        assert out.sum() == pytest.approx(1.0)

    def test_reduces_turnover_vs_full_rebalance(self) -> None:
        current = np.array([0.55, 0.26, 0.19])
        target = np.array([0.50, 0.30, 0.20])
        # only asset 0 breaches 5pp threshold.
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.049)
        banded = apply_no_trade_band(current, target, cfg)
        assert compute_turnover(current, banded) <= compute_turnover(current, target)

    def test_zero_total_returns_blend(self) -> None:
        out = apply_no_trade_band([0.0, 0.0], [0.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 0.0])


class TestBuildRebalancingWalkForward:
    def _returns(self) -> pd.DataFrame:
        idx = pd.bdate_range("2020-01-01", "2023-12-31")
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            rng.normal(0.0, 0.01, size=(len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )

    def test_default_is_quarterly(self) -> None:
        cv = build_rebalancing_walk_forward()
        assert cv.freq == "QS"
        assert cv.test_size == 1
        assert cv.expand_train is False
        assert cv.purged_size == 0

    def test_monthly_freq_and_splits(self) -> None:
        cv = build_rebalancing_walk_forward(
            CalendarRebalancingConfig.for_monthly(), train_size=12
        )
        assert cv.freq == "MS"
        assert cv.get_n_splits(self._returns()) > 0

    def test_expand_and_purge_forwarded(self) -> None:
        cv = build_rebalancing_walk_forward(
            CalendarRebalancingConfig.for_quarterly(),
            train_size=4,
            expand_train=True,
            purged_size=2,
            reduce_test=True,
            previous=True,
        )
        assert cv.expand_train is True
        assert cv.purged_size == 2
        assert cv.reduce_test is True
        assert cv.previous is True

    def test_freq_offset_string_parsed(self) -> None:
        cv = build_rebalancing_walk_forward(
            CalendarRebalancingConfig.for_monthly(), freq_offset="2D"
        )
        assert cv.freq_offset == pd.tseries.frequencies.to_offset("2D")

    def test_splits_are_time_ordered(self) -> None:
        cv = build_rebalancing_walk_forward(
            CalendarRebalancingConfig.for_monthly(), train_size=6
        )
        X = self._returns()
        prev_test_start = -1
        for train_idx, test_idx in cv.split(X):
            # no leakage: every test index sits after all train indices.
            assert train_idx.max() < test_idx.min()
            assert test_idx.min() > prev_test_start
            prev_test_start = test_idx.min()


# ---------------------------------------------------------------------------
# Shared helpers for hybrid tests
# ---------------------------------------------------------------------------

_BREACH = np.array([0.56, 0.24, 0.20])  # drifted by 6pp → breaches 5pp threshold
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
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is True

    # -- Acceptance criterion 2: calendar date + no breach → False ------------

    def test_review_date_no_breach_returns_false(self) -> None:
        """At a calendar review date with no drift breach, no rebalancing."""
        current = _review_date(_LAST_REVIEW, 21)
        decision, _reason = should_rebalance_hybrid(
            _NO_BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is False

    # -- Acceptance criterion 3: mid-calendar + breach → False ----------------

    def test_mid_calendar_breach_returns_false(self) -> None:
        """Between review dates, always returns False regardless of drift."""
        current = _review_date(_LAST_REVIEW, 10)  # only 10 bdays elapsed
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is False

    # -- Edge cases -----------------------------------------------------------

    def test_exactly_at_threshold_bdays(self) -> None:
        """Exactly trading_days elapsed is treated as a review date."""
        current = _review_date(_LAST_REVIEW, 21)
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert isinstance(decision, bool)
        assert decision is True

    def test_one_day_before_review(self) -> None:
        """One business day before the review interval → always False."""
        current = _review_date(_LAST_REVIEW, 20)
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is False

    def test_past_due_review_with_breach(self) -> None:
        """Overdue review (more than trading_days elapsed) with breach → True."""
        current = _review_date(_LAST_REVIEW, 42)  # 2 months elapsed
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is True

    def test_relative_threshold_variant(self) -> None:
        """Hybrid with relative threshold behaves consistently."""
        cfg = HybridRebalancingConfig(
            calendar=CalendarRebalancingConfig.for_quarterly(),
            threshold=ThresholdRebalancingConfig.for_relative(threshold=0.25),
        )
        # asset 1 relative drift: |0.38-0.30|/0.30 = 0.267 > 0.25 → breach
        breaching = np.array([0.50, 0.38, 0.12])
        current = _review_date(_LAST_REVIEW, 63)  # quarterly = 63 bdays
        decision, _reason = should_rebalance_hybrid(
            breaching, _TARGET, cfg, current, _LAST_REVIEW
        )
        assert decision is True

    def test_same_day_as_review_returns_false(self) -> None:
        """current_date == last_review_date (0 elapsed) → False."""
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, _LAST_REVIEW, _LAST_REVIEW
        )
        assert decision is False

    def test_weekend_before_review_boundary_returns_false(self) -> None:
        """Saturday before the 21-bday mark → False.

        BDay(19) from 2024-01-02 = 2024-01-26 (Fri); +1 day = Sat 2024-01-27,
        which is before next_review = BDay(21) = 2024-01-30.
        """
        current = _LAST_REVIEW + pd.offsets.BDay(19) + pd.offsets.Day(1)
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is False

    def test_weekend_at_review_boundary_fires_on_monday(self) -> None:
        """BDay(21) from last_review is a business day → True with breach."""
        current = _LAST_REVIEW + pd.offsets.BDay(21)
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is True

    def test_63_bday_quarterly_boundary(self) -> None:
        """Exactly 63 bdays elapsed for quarterly config → True with breach."""
        cfg = HybridRebalancingConfig(
            calendar=CalendarRebalancingConfig.for_quarterly(),
            threshold=ThresholdRebalancingConfig.for_absolute(threshold=0.05),
        )
        current = _LAST_REVIEW + pd.offsets.BDay(63)
        decision, _reason = should_rebalance_hybrid(
            _BREACH, _TARGET, cfg, current, _LAST_REVIEW
        )
        assert decision is True


class TestShouldRebalanceHybridReason:
    """Each branch surfaces a structured reason literal."""

    def test_when_between_review_dates_then_reason_between_review_dates(self) -> None:
        current = _review_date(_LAST_REVIEW, 10)
        decision, reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is False
        assert reason == "between_review_dates"

    def test_when_review_date_with_breach_then_reason_threshold_met(self) -> None:
        current = _review_date(_LAST_REVIEW, 21)
        decision, reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is True
        assert reason == "threshold_met"

    def test_when_review_date_without_breach_then_reason_threshold_not_met(
        self,
    ) -> None:
        current = _review_date(_LAST_REVIEW, 21)
        decision, reason = should_rebalance_hybrid(
            _NO_BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert decision is False
        assert reason == "threshold_not_met"

    def test_when_returned_then_reason_is_in_known_set(self) -> None:
        known = {"between_review_dates", "threshold_met", "threshold_not_met"}
        current = _review_date(_LAST_REVIEW, 21)
        _decision, reason = should_rebalance_hybrid(
            _BREACH, _TARGET, _MONTHLY_CFG, current, _LAST_REVIEW
        )
        assert reason in known
