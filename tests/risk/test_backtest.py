"""Tests for the Value-at-Risk backtest.

The golden numbers come from the Kupiec (1995) and Christoffersen (1998) likelihood
ratios written out with plain ``math`` from the paper formulas, and from the Basel
traffic-light table, never from the module under test.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.risk import (
    VarBacktestResult,
    VarConfig,
    VarMethod,
    backtest_var,
    rolling_var_forecasts,
)

_FORECAST = 0.02
_BREACH = -0.03
_QUIET = 0.0
_YEAR = 250

# Three 250-day exception patterns: ten exceptions in one block, ten exceptions
# 25 days apart, and an exception every other day (125 of them).
_CLUSTERED = [0] * 120 + [1] * 10 + [0] * 120
_SPACED = ([0] * 24 + [1]) * 10
_ALTERNATING = [0, 1] * 125


def _from_pattern(pattern: list[int]) -> tuple[pd.Series, pd.Series]:
    """Return returns and a constant VaR forecast that breach on the 1s."""
    index = pd.bdate_range("2021-01-04", periods=len(pattern))
    returns = pd.Series([_BREACH if hit else _QUIET for hit in pattern], index=index)
    return returns, pd.Series(_FORECAST, index=index)


def _run(pattern: list[int], confidence_level: float = 0.99) -> VarBacktestResult:
    """Backtest a constant forecast against an exception pattern."""
    returns, forecasts = _from_pattern(pattern)
    return backtest_var(returns, forecasts, confidence_level)


class TestExceptionCount:
    """What counts as an exception and how the result reports it."""

    def test_counts_the_days_the_loss_exceeds_the_forecast(self) -> None:
        """Ten breaches in 250 days at 99% against 2.5 expected."""
        result = _run(_SPACED)
        assert isinstance(result, VarBacktestResult)
        assert result.n_observations == _YEAR
        assert result.n_exceptions == 10
        assert result.expected_exceptions == pytest.approx(2.5)
        assert result.exception_rate == pytest.approx(0.04)

    def test_a_loss_equal_to_the_forecast_is_not_an_exception(self) -> None:
        """An exception needs the return strictly below the negated forecast."""
        index = pd.bdate_range("2021-01-04", periods=5)
        returns = pd.Series([-0.02, -0.02, 0.0, 0.01, -0.01], index=index)
        forecasts = pd.Series(0.02, index=index)
        assert backtest_var(returns, forecasts, 0.95).n_exceptions == 0

    def test_the_result_is_immutable(self) -> None:
        """A result cannot be changed behind a caller's back."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            _run(_SPACED).n_exceptions = 0  # type: ignore[misc]


class TestKupiec:
    """Kupiec's proportion-of-failures test."""

    def test_ten_exceptions_in_a_year_reject_the_99_percent_model(self) -> None:
        """T=250, x=10, p=1%: LR = 2*(10 ln 4 + 240 ln(240/247.5)) = 12.9555."""
        # Evaluated with math.log from the likelihood ratio in the paper; the
        # p-value is erfc(sqrt(LR/2)), the chi-square(1) survival function.
        result = _run(_SPACED)
        assert result.kupiec_statistic == pytest.approx(12.955491062, rel=1e-9)
        assert result.kupiec_p_value == pytest.approx(0.000318984508, rel=1e-6)

    def test_the_expected_number_of_exceptions_gives_a_zero_statistic(self) -> None:
        """T=500 at 99% expects 5 exceptions: the likelihood ratio is 0, p is 1."""
        result = _run([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 5 + [0] * 450)
        assert result.n_exceptions == 5
        assert result.kupiec_statistic == pytest.approx(0.0, abs=1e-9)
        # The p-value follows the square root of the statistic, so rounding noise of
        # 1e-14 in the statistic moves it by about 1e-7.
        assert result.kupiec_p_value == pytest.approx(1.0, abs=1e-6)

    def test_no_exceptions_use_the_zero_log_zero_convention(self) -> None:
        """x=0: LR = -2 T ln(1-p) = 5.02517 for T=250 at 99%."""
        # Closed form for x=0: -2 * 250 * ln(0.99); erfc gives the p-value.
        result = _run([0] * _YEAR)
        assert result.kupiec_statistic == pytest.approx(
            -2 * _YEAR * math.log(0.99), rel=1e-9
        )
        assert result.kupiec_p_value == pytest.approx(0.0249815031, rel=1e-6)

    def test_every_day_an_exception_uses_the_zero_log_zero_convention(self) -> None:
        """x=T: LR = -2 T ln(p) = 2302.585 for T=250 at 99%."""
        result = _run([1] * _YEAR)
        assert result.kupiec_statistic == pytest.approx(
            -2 * _YEAR * math.log(0.01), rel=1e-9
        )
        assert result.kupiec_p_value == pytest.approx(0.0, abs=1e-300)

    def test_a_95_percent_model_is_judged_against_five_percent(self) -> None:
        """T=200 at 95% expects 10 exceptions: ten of them is a perfect fit."""
        result = _run([1] * 10 + [0] * 190, confidence_level=0.95)
        assert result.expected_exceptions == pytest.approx(10.0)
        assert result.kupiec_statistic == pytest.approx(0.0, abs=1e-9)

    def test_more_exceptions_than_expected_give_a_larger_statistic(self) -> None:
        """The statistic grows as the exception count moves away from 2.5."""
        counts = [_run([1] * k + [0] * (_YEAR - k)).kupiec_statistic for k in (4, 5, 8)]
        assert counts == sorted(counts)

    def test_too_few_exceptions_are_also_a_misfit(self) -> None:
        """Zero exceptions in a year at 99% is mildly over-conservative."""
        assert _run([0] * _YEAR).kupiec_p_value < 0.05


class TestChristoffersen:
    """Christoffersen's independence and conditional coverage tests."""

    def test_clustered_exceptions_are_rejected(self) -> None:
        """One block of ten breaches: n=(238, 1, 1, 9) and LR_ind = 64.44."""
        # Transition counts and the likelihood ratio from the paper's formula,
        # evaluated with math.log; the p-value is erfc(sqrt(LR/2)).
        result = _run(_CLUSTERED)
        assert result.independence_statistic == pytest.approx(64.4398661328, rel=1e-9)
        assert result.independence_p_value == pytest.approx(9.9524e-16, rel=1e-3)

    def test_evenly_spaced_exceptions_look_independent(self) -> None:
        """Ten breaches 25 days apart: n=(230, 10, 9, 0) and LR_ind = 0.7518."""
        result = _run(_SPACED)
        assert result.independence_statistic == pytest.approx(0.7517635167, rel=1e-8)
        assert result.independence_p_value == pytest.approx(0.3859184647, rel=1e-6)

    def test_alternating_exceptions_are_rejected(self) -> None:
        """A breach every other day is dependent too: LR_ind = 345.18."""
        result = _run(_ALTERNATING)
        assert result.independence_statistic == pytest.approx(345.1832798438, rel=1e-9)
        assert result.independence_p_value < 1e-70

    def test_clustering_scores_worse_than_spacing_at_equal_counts(self) -> None:
        """Both patterns breach ten times, so Kupiec cannot tell them apart."""
        clustered, spaced = _run(_CLUSTERED), _run(_SPACED)
        assert clustered.kupiec_statistic == pytest.approx(spaced.kupiec_statistic)
        assert clustered.independence_statistic > 10 * spaced.independence_statistic

    @pytest.mark.parametrize("pattern", [[0] * _YEAR, [1] * _YEAR])
    def test_no_transitions_to_estimate_give_a_zero_statistic(
        self, pattern: list[int]
    ) -> None:
        """Without exceptions, or with only exceptions, there is nothing to test."""
        result = _run(pattern)
        assert result.independence_statistic == pytest.approx(0.0, abs=1e-12)
        assert result.independence_p_value == pytest.approx(1.0)

    def test_conditional_coverage_adds_the_two_statistics(self) -> None:
        """LR_cc = LR_pof + LR_ind, with a chi-square(2) p-value of exp(-LR/2)."""
        result = _run(_CLUSTERED)
        total = result.kupiec_statistic + result.independence_statistic
        assert result.conditional_coverage_statistic == pytest.approx(total)
        assert result.conditional_coverage_statistic == pytest.approx(
            77.3953571952, rel=1e-9
        )
        assert result.conditional_coverage_p_value == pytest.approx(
            math.exp(-total / 2.0), rel=1e-6
        )

    def test_two_observations_are_enough(self) -> None:
        """Two observations hold one transition and still give a result."""
        result = _run([0, 1])
        assert result.n_observations == 2
        assert result.independence_statistic >= 0.0


class TestTrafficLight:
    """The Basel traffic-light zones."""

    @pytest.mark.parametrize(
        ("exceptions", "zone"),
        [
            (0, "green"),
            (4, "green"),
            (5, "yellow"),
            (9, "yellow"),
            (10, "red"),
            (25, "red"),
        ],
    )
    def test_zones_at_99_percent_over_250_days(
        self, exceptions: int, zone: str
    ) -> None:
        """Green up to 4 exceptions, yellow 5 to 9, red from 10."""
        result = _run([1] * exceptions + [0] * (_YEAR - exceptions))
        assert result.traffic_light == zone

    @pytest.mark.parametrize(
        ("n_days", "confidence_level"),
        [(249, 0.99), (251, 0.99), (_YEAR, 0.95), (_YEAR, 0.975)],
    )
    def test_no_zone_outside_the_basel_setting(
        self, n_days: int, confidence_level: float
    ) -> None:
        """The table is defined for 250 days at 99% only."""
        result = _run([1] * 12 + [0] * (n_days - 12), confidence_level)
        assert result.traffic_light is None


class TestAlignment:
    """Returns and forecasts are matched on their common non-missing dates."""

    def test_only_the_common_dates_are_scored(self) -> None:
        """Forecasts that start late and have a gap shrink the sample."""
        index = pd.bdate_range("2021-01-04", periods=12)
        returns = pd.Series(0.0, index=index)
        forecasts = pd.Series(0.02, index=index[4:])
        forecasts.iloc[2] = np.nan
        result = backtest_var(returns, forecasts, 0.95)
        assert result.n_observations == 7

    def test_missing_returns_are_dropped(self) -> None:
        """A NaN return removes that date from the sample."""
        returns, forecasts = _from_pattern([0] * 10)
        returns.iloc[3] = np.nan
        assert backtest_var(returns, forecasts, 0.95).n_observations == 9

    def test_rolling_forecasts_can_be_scored_directly(self) -> None:
        """The leading NaNs of a rolling forecast are dropped by the alignment."""
        returns = pd.Series(
            np.random.default_rng(1).normal(0.0, 0.01, 600),
            index=pd.bdate_range("2019-01-01", periods=600),
        )
        forecasts = rolling_var_forecasts(returns, 250)
        result = backtest_var(returns, forecasts, 0.95)
        assert result.n_observations == 350

    def test_array_inputs_are_aligned_by_position(self) -> None:
        """Plain arrays of equal length are matched element by element."""
        returns = np.array([0.0, -0.05, 0.0, 0.0])
        forecasts = np.array([0.02, 0.02, 0.02, 0.02])
        assert backtest_var(returns, forecasts, 0.95).n_exceptions == 1

    def test_calibrated_forecasts_pass_the_backtest(self) -> None:
        """99% normal VaR on normal returns breaches about 1% of the days."""
        returns = pd.Series(
            np.random.default_rng(21).normal(0.0003, 0.01, 1500),
            index=pd.bdate_range("2019-01-01", periods=1500),
        )
        config = VarConfig(method=VarMethod.PARAMETRIC, confidence_level=0.99)
        forecasts = rolling_var_forecasts(returns, 250, config)
        result = backtest_var(returns, forecasts, 0.99)
        assert 0.002 < result.exception_rate < 0.025
        assert result.kupiec_p_value > 0.001


class TestBacktestValidation:
    """``backtest_var`` rejects inputs it cannot score."""

    @pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.2, 1.5, math.nan])
    def test_an_invalid_confidence_level_raises(self, confidence_level: float) -> None:
        """The confidence level has to lie strictly between 0 and 1."""
        returns, forecasts = _from_pattern([0] * 10)
        with pytest.raises(ConfigurationError, match="confidence_level"):
            backtest_var(returns, forecasts, confidence_level)

    def test_series_with_no_common_date_raise(self) -> None:
        """Disjoint indexes leave nothing to compare."""
        returns, forecasts = _from_pattern([0] * 10)
        shifted = forecasts.copy()
        shifted.index = shifted.index + pd.Timedelta(days=365)
        with pytest.raises(DataError, match="at least 2"):
            backtest_var(returns, shifted, 0.95)

    def test_a_single_common_observation_raises(self) -> None:
        """One day cannot be backtested."""
        returns, forecasts = _from_pattern([0] * 10)
        with pytest.raises(DataError, match="at least 2"):
            backtest_var(returns.iloc[:1], forecasts, 0.95)

    def test_duplicate_index_labels_raise(self) -> None:
        """Dates that repeat cannot be matched one to one."""
        returns, forecasts = _from_pattern([0] * 10)
        labels = forecasts.index.to_list()
        labels[-1] = labels[0]
        repeated = pd.Series(forecasts.to_numpy(), index=labels)
        with pytest.raises(DataError, match="unique"):
            backtest_var(returns, repeated, 0.95)

    def test_an_infinite_value_raises(self) -> None:
        """An infinite forecast is a data error, not an exception."""
        returns, forecasts = _from_pattern([0] * 10)
        forecasts.iloc[4] = np.inf
        with pytest.raises(DataError, match="finite"):
            backtest_var(returns, forecasts, 0.95)

    def test_a_two_dimensional_input_raises(self) -> None:
        """Each input has to be one series."""
        _, forecasts = _from_pattern([0] * 10)
        with pytest.raises(DataError, match="one-dimensional"):
            backtest_var(np.zeros((10, 2)), forecasts, 0.95)

    def test_non_numeric_input_raises(self) -> None:
        """Text cannot be a return."""
        _, forecasts = _from_pattern([0] * 3)
        with pytest.raises(DataError, match="one-dimensional"):
            backtest_var(["a", "b", "c"], forecasts, 0.95)
