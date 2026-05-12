"""Cycle-3 §7.3 Top-4 retighten loop contract tests (issue #553).

Exhausts the contract for ``_solve_with_retighten`` and ``_top4_weight``:
trace shape, retry cap, infeasibility ``RuntimeError`` message, and the
order-invariance of the Top-4 measure.  Extends the partial coverage in
``tests/research/test_optimization.py`` with a dedicated dedicated module.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from optimizer.optimization import MeanRiskConfig
from research.optimization._retighten import (
    _SHRINK_FACTOR,
    _TOP4_THRESHOLD,
    _TRADING_DAYS_PER_YEAR,
    _annualized_sharpe,
    _solve_with_retighten,
    _split_into_subperiods,
    _top4_weight,
)

# ---------------------------------------------------------------------------
# Fakes & helpers
# ---------------------------------------------------------------------------


class _FakeMeanRisk:
    """Stand-in for ``skfolio.MeanRisk`` driven by a fixed weight vector."""

    def __init__(self, max_weights: float, weights: np.ndarray) -> None:
        self.max_weights = max_weights
        self._weights = np.asarray(weights, dtype=float)

    def fit(self, _: pd.DataFrame) -> _FakeMeanRisk:
        self.weights_ = self._weights
        return self


def _config(max_weights: float = 0.10) -> MeanRiskConfig:
    return MeanRiskConfig(max_weights=max_weights)


def _max_w(config: MeanRiskConfig) -> float:
    return cast(float, config.max_weights)


def _builder_from_sequence(
    weight_vectors: list[np.ndarray],
) -> tuple[Any, list[float]]:
    """Builder that pops a fresh weight vector per call; tracks max_weights."""
    pending = list(weight_vectors)
    seen_max: list[float] = []

    def builder(config: MeanRiskConfig) -> _FakeMeanRisk:
        seen_max.append(_max_w(config))
        return _FakeMeanRisk(_max_w(config), pending.pop(0))

    return builder, seen_max


@pytest.fixture()
def returns_8() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.normal(size=(60, 8)))


@pytest.fixture()
def returns_20() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.normal(size=(60, 20)))


# ---------------------------------------------------------------------------
# _top4_weight — order invariance
# ---------------------------------------------------------------------------


class TestTop4Weight:
    def test_when_weights_sorted_descending_then_returns_sum_of_top_4(self) -> None:
        weights = np.array([0.30, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.10])
        assert _top4_weight(weights) == pytest.approx(0.75)

    def test_when_weights_sorted_ascending_then_same_result(self) -> None:
        ascending = np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.30])
        descending = np.sort(ascending)[::-1]
        assert _top4_weight(ascending) == _top4_weight(descending)

    def test_when_weights_shuffled_then_same_result(self) -> None:
        base = np.array([0.30, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.10])
        rng = np.random.default_rng(0)
        for _ in range(5):
            shuffled = base.copy()
            rng.shuffle(shuffled)
            assert _top4_weight(shuffled) == pytest.approx(_top4_weight(base))

    def test_when_fewer_than_four_weights_then_returns_sum_of_all(self) -> None:
        assert _top4_weight(np.array([0.50, 0.30, 0.20])) == pytest.approx(1.0)

    def test_when_all_zero_then_returns_zero(self) -> None:
        assert _top4_weight(np.zeros(10)) == 0.0


# ---------------------------------------------------------------------------
# Convergence — single attempt, multi-attempt
# ---------------------------------------------------------------------------


class TestRetightenConvergence:
    def test_when_first_attempt_below_threshold_then_one_entry_trace(
        self, returns_20: pd.DataFrame
    ) -> None:
        diversified = np.full(20, 1.0 / 20)  # top4 = 0.20 < 0.30
        opt: Any = _FakeMeanRisk(0.10, diversified)
        builder, seen_max = _builder_from_sequence([])

        result, trace = _solve_with_retighten(
            opt,
            returns_20,
            config=_config(0.10),
            builder=builder,
        )

        assert result is opt
        assert len(trace) == 1
        assert seen_max == []

    def test_when_three_attempts_to_converge_then_max_weights_decreases_by_shrink(
        self, returns_20: pd.DataFrame
    ) -> None:
        # First two attempts: top-4 = 0.40 (concentrated). Third: top-4 = 0.20.
        concentrated = np.zeros(20)
        concentrated[:4] = 0.10
        concentrated[4:] = 0.60 / 16
        diversified = np.full(20, 1.0 / 20)

        opt: Any = _FakeMeanRisk(0.10, concentrated)
        builder, _ = _builder_from_sequence([concentrated, diversified])

        _, trace = _solve_with_retighten(
            opt,
            returns_20,
            config=_config(0.10),
            builder=builder,
        )

        assert len(trace) == 3
        assert trace[0]["max_weights"] == pytest.approx(0.10)
        assert trace[1]["max_weights"] == pytest.approx(0.10 * _SHRINK_FACTOR)
        assert trace[2]["max_weights"] == pytest.approx(
            0.10 * _SHRINK_FACTOR * _SHRINK_FACTOR
        )

    def test_when_three_attempts_to_converge_then_attempt_indices_one_to_three(
        self, returns_20: pd.DataFrame
    ) -> None:
        concentrated = np.zeros(20)
        concentrated[:4] = 0.10
        concentrated[4:] = 0.60 / 16
        diversified = np.full(20, 1.0 / 20)
        opt: Any = _FakeMeanRisk(0.10, concentrated)
        builder, _ = _builder_from_sequence([concentrated, diversified])

        _, trace = _solve_with_retighten(
            opt,
            returns_20,
            config=_config(0.10),
            builder=builder,
        )

        assert [e["attempt"] for e in trace] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Divergence — RuntimeError after cap
# ---------------------------------------------------------------------------


class TestRetightenDivergence:
    def test_when_never_converges_then_runtime_error_raised(
        self, returns_8: pd.DataFrame
    ) -> None:
        concentrated = np.array(
            [0.10, 0.10, 0.10, 0.10, 0.20, 0.20, 0.10, 0.10],
        )
        opt: Any = _FakeMeanRisk(0.10, concentrated)
        builder, _ = _builder_from_sequence(
            [concentrated.copy() for _ in range(5)],
        )

        with pytest.raises(RuntimeError):
            _solve_with_retighten(
                opt,
                returns_8,
                config=_config(0.10),
                builder=builder,
            )

    def test_when_runtime_error_then_message_includes_top4_value(
        self, returns_8: pd.DataFrame
    ) -> None:
        concentrated = np.array(
            [0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15],
        )
        opt: Any = _FakeMeanRisk(0.10, concentrated)
        builder, _ = _builder_from_sequence(
            [concentrated.copy() for _ in range(5)],
        )
        with pytest.raises(RuntimeError, match=r"top4=\d+\.\d{4}"):
            _solve_with_retighten(
                opt,
                returns_8,
                config=_config(0.10),
                builder=builder,
            )

    def test_when_runtime_error_then_message_includes_max_weights(
        self, returns_8: pd.DataFrame
    ) -> None:
        concentrated = np.array(
            [0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15],
        )
        opt: Any = _FakeMeanRisk(0.10, concentrated)
        builder, _ = _builder_from_sequence(
            [concentrated.copy() for _ in range(5)],
        )
        with pytest.raises(RuntimeError, match="max_weights"):
            _solve_with_retighten(
                opt,
                returns_8,
                config=_config(0.10),
                builder=builder,
            )

    def test_when_default_max_retries_then_builder_called_five_times(
        self, returns_8: pd.DataFrame
    ) -> None:
        concentrated = np.array(
            [0.10, 0.10, 0.10, 0.10, 0.20, 0.20, 0.10, 0.10],
        )
        opt: Any = _FakeMeanRisk(0.10, concentrated)
        builder, seen_max = _builder_from_sequence(
            [concentrated.copy() for _ in range(5)],
        )
        with pytest.raises(RuntimeError):
            _solve_with_retighten(
                opt,
                returns_8,
                config=_config(0.10),
                builder=builder,
            )
        assert len(seen_max) == 5


# ---------------------------------------------------------------------------
# Trace shape — keys + types
# ---------------------------------------------------------------------------


class TestTraceShape:
    def test_when_called_then_each_entry_has_required_keys(
        self, returns_20: pd.DataFrame
    ) -> None:
        diversified = np.full(20, 1.0 / 20)
        opt: Any = _FakeMeanRisk(0.10, diversified)
        builder, _ = _builder_from_sequence([])
        _, trace = _solve_with_retighten(
            opt,
            returns_20,
            config=_config(0.10),
            builder=builder,
        )
        for entry in trace:
            assert set(entry.keys()) == {"attempt", "top4", "max_weights"}

    def test_when_called_then_entry_types_are_int_float_float(
        self, returns_20: pd.DataFrame
    ) -> None:
        diversified = np.full(20, 1.0 / 20)
        opt: Any = _FakeMeanRisk(0.10, diversified)
        builder, _ = _builder_from_sequence([])
        _, trace = _solve_with_retighten(
            opt,
            returns_20,
            config=_config(0.10),
            builder=builder,
        )
        entry = trace[0]
        assert isinstance(entry["attempt"], int)
        assert isinstance(entry["top4"], float)
        assert isinstance(entry["max_weights"], float)

    def test_when_top4_threshold_imported_then_equals_030(self) -> None:
        assert pytest.approx(0.30) == _TOP4_THRESHOLD

    def test_when_shrink_factor_imported_then_equals_095(self) -> None:
        assert pytest.approx(0.95) == _SHRINK_FACTOR


# ---------------------------------------------------------------------------
# _annualized_sharpe — edge cases
# ---------------------------------------------------------------------------


class TestAnnualizedSharpe:
    def test_when_empty_array_then_returns_zero(self) -> None:
        assert _annualized_sharpe(np.array([])) == 0.0

    def test_when_single_element_array_then_returns_zero(self) -> None:
        # std with ddof=1 is undefined for size-1; falls back to 0
        assert _annualized_sharpe(np.array([0.05])) == 0.0

    def test_when_zero_variance_returns_then_returns_zero(self) -> None:
        # All identical values → std == 0
        assert _annualized_sharpe(np.full(10, 0.01)) == 0.0

    def test_when_normal_returns_then_result_is_finite_float(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(loc=0.001, scale=0.01, size=252)
        result = _annualized_sharpe(returns)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_when_positive_mean_positive_std_then_positive_sharpe(self) -> None:
        # Positive drift → positive Sharpe
        rng = np.random.default_rng(1)
        returns = rng.normal(loc=0.005, scale=0.01, size=252)
        assert _annualized_sharpe(returns) > 0.0

    def test_when_annualized_then_scales_by_sqrt_trading_days(self) -> None:
        # Use a deterministic array; verify annualization factor
        returns = np.array([0.01, 0.02, -0.01, 0.03])
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        expected = mean / std * np.sqrt(_TRADING_DAYS_PER_YEAR)
        assert _annualized_sharpe(returns) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _split_into_subperiods — edge cases
# ---------------------------------------------------------------------------


class TestSplitIntoSubperiods:
    def _make_series(self, n: int) -> pd.Series:
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(np.arange(float(n)), index=idx)

    def test_when_n_equals_one_then_returns_single_slice_with_all_data(
        self,
    ) -> None:
        s = self._make_series(10)
        slices = _split_into_subperiods(s, 1)
        assert len(slices) == 1
        assert len(slices[0]) == 10

    def test_when_n_equals_length_then_each_slice_has_one_element(self) -> None:
        s = self._make_series(4)
        slices = _split_into_subperiods(s, 4)
        assert len(slices) == 4
        assert all(len(sl) == 1 for sl in slices)

    def test_when_n_larger_than_length_then_some_slices_are_empty(self) -> None:
        s = self._make_series(2)
        slices = _split_into_subperiods(s, 5)
        assert len(slices) == 5
        total = sum(len(sl) for sl in slices)
        assert total == 2

    def test_when_three_subperiods_then_slices_are_contiguous_and_cover_all(
        self,
    ) -> None:
        s = self._make_series(9)
        slices = _split_into_subperiods(s, 3)
        assert len(slices) == 3
        assert sum(len(sl) for sl in slices) == 9

    def test_when_split_then_values_match_original_order(self) -> None:
        s = self._make_series(6)
        slices = _split_into_subperiods(s, 3)
        reconstructed = pd.concat(slices)
        pd.testing.assert_series_equal(reconstructed, s)

    def test_when_trading_days_constant_then_equals_252(self) -> None:
        assert _TRADING_DAYS_PER_YEAR == 252
