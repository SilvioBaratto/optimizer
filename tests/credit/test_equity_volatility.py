"""Tests for equity-volatility estimation and the market-data entry point.

Hand-derived numbers use price paths built from known log returns, so the sample
standard deviation is computed on paper rather than by the module under test.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from optimizer.credit import (
    MertonConfig,
    MertonResult,
    estimate_equity_volatility,
    merton_from_market,
    solve_merton,
)
from optimizer.exceptions import ConfigurationError, DataError


def _prices_from_log_levels(log_levels: list[float]) -> list[float]:
    """Return the price path whose log level at each step is ``log_levels``."""
    return [math.exp(level) for level in log_levels]


# Log returns 0.01, 0.02, -0.01: mean 2/3 %, sample variance (7/3) * 1e-4.
_HAND_PRICES = _prices_from_log_levels([0.0, 0.01, 0.03, 0.02])
_HAND_DAILY_VOL = math.sqrt(7.0 / 3.0) / 100.0
# Log returns 0.05, -0.04, 0.02, -0.01.
_WINDOW_PRICES = _prices_from_log_levels([0.0, 0.05, 0.01, 0.03, 0.02])


class TestEstimateEquityVolatility:
    """``estimate_equity_volatility``: annualised sample volatility of log returns."""

    def test_matches_a_hand_computed_series(self) -> None:
        """The sample standard deviation is sqrt(7/3) % a day, sqrt(588) % a year."""
        assert estimate_equity_volatility(_HAND_PRICES) == pytest.approx(
            math.sqrt(588.0) / 100.0, abs=1e-12
        )

    def test_the_annualisation_factor_scales_with_the_square_root(self) -> None:
        """Volatility grows with the square root of the periods per year."""
        daily = estimate_equity_volatility(_HAND_PRICES, periods_per_year=1)
        weekly = estimate_equity_volatility(_HAND_PRICES, periods_per_year=52)
        assert daily == pytest.approx(_HAND_DAILY_VOL, abs=1e-12)
        assert weekly == pytest.approx(_HAND_DAILY_VOL * math.sqrt(52.0), abs=1e-12)

    def test_uses_the_sample_standard_deviation(self) -> None:
        """Four returns 0.05, -0.04, 0.02, -0.01 have a sample variance of 0.0015."""
        vol = estimate_equity_volatility(_WINDOW_PRICES, periods_per_year=1)
        assert vol == pytest.approx(math.sqrt(0.0015), abs=1e-12)

    def test_window_keeps_only_the_latest_returns(self) -> None:
        """The last three returns -0.04, 0.02, -0.01 have a sample deviation of 0.03."""
        vol = estimate_equity_volatility(_WINDOW_PRICES, window=3)
        assert vol == pytest.approx(0.03 * math.sqrt(252.0), abs=1e-12)

    def test_a_window_of_two_uses_the_last_two_returns(self) -> None:
        """Returns 0.02 and -0.01 are 0.03 apart: deviation 0.03 / sqrt(2)."""
        vol = estimate_equity_volatility(_WINDOW_PRICES, periods_per_year=1, window=2)
        assert vol == pytest.approx(0.03 / math.sqrt(2.0), abs=1e-12)

    def test_a_window_covering_every_return_changes_nothing(self) -> None:
        """Four returns and a window of four is the whole sample."""
        assert estimate_equity_volatility(
            _WINDOW_PRICES, window=4
        ) == estimate_equity_volatility(_WINDOW_PRICES)

    def test_prices_before_the_window_do_not_matter(self) -> None:
        """Rewriting the early history leaves a windowed estimate alone."""
        altered = [*_prices_from_log_levels([0.0, 0.5]), *_WINDOW_PRICES[2:]]
        assert estimate_equity_volatility(altered, window=2) == pytest.approx(
            estimate_equity_volatility(_WINDOW_PRICES, window=2), abs=1e-12
        )

    def test_the_price_level_does_not_matter(self) -> None:
        """Multiplying every price by a constant leaves log returns unchanged."""
        scaled = [250.0 * price for price in _HAND_PRICES]
        assert estimate_equity_volatility(scaled) == pytest.approx(
            estimate_equity_volatility(_HAND_PRICES), abs=1e-12
        )

    def test_accepts_a_pandas_series_and_a_numpy_array(self) -> None:
        """Any array-like of prices works."""
        expected = estimate_equity_volatility(_HAND_PRICES)
        dated = pd.Series(_HAND_PRICES, index=pd.bdate_range("2024-01-01", periods=4))
        assert estimate_equity_volatility(dated) == pytest.approx(expected, abs=1e-15)
        assert estimate_equity_volatility(np.array(_HAND_PRICES)) == pytest.approx(
            expected, abs=1e-15
        )

    def test_a_constant_series_has_zero_volatility(self) -> None:
        """A price that never moves has no volatility."""
        assert estimate_equity_volatility([50.0, 50.0, 50.0, 50.0]) == 0.0

    def test_recovers_the_volatility_of_a_long_simulated_path(self) -> None:
        """A seeded geometric Brownian motion is measured within 4 standard errors.

        The standard error of a sample standard deviation of n normal draws is
        sigma / sqrt(2 (n - 1)), here about 0.0013 for sigma = 0.30.
        """
        sigma, drift, periods = 0.30, 0.08, 252
        n_returns = 25_200
        rng = np.random.default_rng(20260)
        step = 1.0 / periods
        log_returns = (drift - sigma**2 / 2) * step + sigma * math.sqrt(step) * (
            rng.standard_normal(n_returns)
        )
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
        standard_error = sigma / math.sqrt(2.0 * (n_returns - 1))
        estimate = estimate_equity_volatility(prices, periods_per_year=periods)
        assert abs(estimate - sigma) < 4.0 * standard_error

    def test_the_simulated_estimate_is_reproducible(self) -> None:
        """The same seed gives the same estimate."""

        def estimate() -> float:
            """Return the volatility of a path drawn from a fixed seed."""
            rng = np.random.default_rng(7)
            levels = np.cumsum(rng.normal(0.0, 0.01, 500))
            return estimate_equity_volatility(100.0 * np.exp(levels))

        assert estimate() == estimate()


class TestEstimateEquityVolatilityValidation:
    """Every way the price series or the settings can be wrong."""

    @pytest.mark.parametrize("prices", [[], [100.0], [100.0, 101.0]])
    def test_fewer_than_three_prices_raise(self, prices: list[float]) -> None:
        """Two returns are the least a sample deviation can use."""
        with pytest.raises(DataError, match="at least 3"):
            estimate_equity_volatility(prices)

    @pytest.mark.parametrize("bad_price", [0.0, -5.0])
    def test_non_positive_prices_raise(self, bad_price: float) -> None:
        """A log return needs strictly positive prices."""
        with pytest.raises(DataError, match="positive"):
            estimate_equity_volatility([100.0, bad_price, 101.0, 102.0])

    @pytest.mark.parametrize("bad_price", [math.nan, math.inf, -math.inf])
    def test_non_finite_prices_raise(self, bad_price: float) -> None:
        """Missing or infinite prices are rejected rather than skipped."""
        with pytest.raises(DataError, match="finite"):
            estimate_equity_volatility([100.0, bad_price, 101.0, 102.0])

    def test_a_two_dimensional_input_raises(self) -> None:
        """One equity, one price per date."""
        with pytest.raises(DataError, match="one-dimensional"):
            estimate_equity_volatility(np.full((4, 2), 100.0))

    def test_non_numeric_prices_raise(self) -> None:
        """Strings are not prices."""
        with pytest.raises(DataError, match="numeric"):
            estimate_equity_volatility(["a", "b", "c", "d"])

    @pytest.mark.parametrize("periods_per_year", [0, -252])
    def test_a_non_positive_annualisation_factor_raises(
        self, periods_per_year: int
    ) -> None:
        """A year has at least one period."""
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            estimate_equity_volatility(_HAND_PRICES, periods_per_year=periods_per_year)

    @pytest.mark.parametrize("window", [1, 0, -3])
    def test_a_window_below_two_raises(self, window: int) -> None:
        """A sample deviation needs two returns."""
        with pytest.raises(ConfigurationError, match="window"):
            estimate_equity_volatility(_WINDOW_PRICES, window=window)

    def test_a_window_longer_than_the_history_raises(self) -> None:
        """Asking for 5 returns out of 4 is a data problem, not a silent shrink."""
        with pytest.raises(DataError, match="window"):
            estimate_equity_volatility(_WINDOW_PRICES, window=5)


def _assert_same_solution(actual: MertonResult, expected: MertonResult) -> None:
    """Assert every reported number agrees to rounding in the volatility input."""
    for field in dataclasses.fields(MertonResult):
        assert getattr(actual, field.name) == pytest.approx(
            getattr(expected, field.name), abs=1e-9
        ), field.name


class TestMertonFromMarket:
    """``merton_from_market`` composes the volatility estimate with the solver."""

    def test_matches_an_independent_solve_of_the_hand_series(self) -> None:
        """Scratch plain-math solve for E = 3, sigma_E = sqrt(588) %, D = 10, r = 5%."""
        result = merton_from_market(3.0, _HAND_PRICES, 10.0, 0.05)
        assert result.asset_value == pytest.approx(12.512294094569462, abs=1e-8)
        assert result.asset_volatility == pytest.approx(0.058139786016617796, abs=1e-8)
        assert result.d2 == pytest.approx(4.685887183976383, abs=1e-6)
        assert result.default_probability == pytest.approx(
            1.3937489324877816e-06, rel=1e-4
        )

    def test_equals_estimating_then_solving(self) -> None:
        """The entry point is the two public steps in sequence."""
        volatility = estimate_equity_volatility(_HAND_PRICES)
        assert merton_from_market(3.0, _HAND_PRICES, 10.0, 0.05) == solve_merton(
            3.0, volatility, 10.0, 0.05
        )

    def test_forwards_the_annualisation_factor(self) -> None:
        """Weekly prices annualise with 52 periods, not 252."""
        weekly = merton_from_market(3.0, _HAND_PRICES, 10.0, 0.05, periods_per_year=52)
        expected = solve_merton(3.0, _HAND_DAILY_VOL * math.sqrt(52.0), 10.0, 0.05)
        _assert_same_solution(weekly, expected)

    def test_forwards_the_configuration(self) -> None:
        """A drift and a maturity reach the solver."""
        config = MertonConfig(maturity=2.0, asset_drift=0.10)
        result = merton_from_market(3.0, _HAND_PRICES, 10.0, 0.05, config=config)
        expected = solve_merton(3.0, math.sqrt(588.0) / 100.0, 10.0, 0.05, config)
        _assert_same_solution(result, expected)

    def test_constant_prices_raise_a_data_error(self) -> None:
        """Zero volatility cannot be inverted, and it is the price series at fault."""
        with pytest.raises(DataError, match="constant"):
            merton_from_market(3.0, [50.0, 50.0, 50.0, 50.0], 10.0, 0.05)

    def test_a_bad_market_cap_is_a_configuration_error(self) -> None:
        """The solver's input validation applies to the market cap."""
        with pytest.raises(ConfigurationError, match="equity_value"):
            merton_from_market(0.0, _HAND_PRICES, 10.0, 0.05)

    def test_bad_prices_are_a_data_error(self) -> None:
        """The estimator's validation applies to the price series."""
        with pytest.raises(DataError, match="at least 3"):
            merton_from_market(3.0, [100.0, 101.0], 10.0, 0.05)

    def test_a_bad_total_debt_is_a_configuration_error(self) -> None:
        """The debt face value must be positive."""
        with pytest.raises(ConfigurationError, match="debt_face_value"):
            merton_from_market(3.0, _HAND_PRICES, 0.0, 0.05)
