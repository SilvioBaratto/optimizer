"""Tests for the Kalman-filter hedge ratio.

The golden numbers come from exact rational arithmetic worked out by hand for a
two-observation example, from a plain-Python (no numpy) recursion of the textbook
filter, from the closed-form ridge and ordinary least-squares solutions that the
filter must approach when the state does not move, and from the statistical
properties of a correctly specified filter.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.statarb import KalmanConfig, KalmanResult, kalman_hedge_ratio


def _index(size: int) -> pd.DatetimeIndex:
    """Return a business-day index of ``size`` dates."""
    return pd.date_range("2021-01-01", periods=size, freq="B")


def _price_like(seed: int, size: int) -> np.ndarray:
    """Return a random walk that starts at 20, like a price series."""
    return 20.0 + np.cumsum(np.random.default_rng(seed).normal(size=size))


def _series(values: np.ndarray) -> pd.Series:
    """Wrap ``values`` in a Series on a business-day index."""
    return pd.Series(values, index=_index(len(values)))


def _reference_filter(
    y: np.ndarray, x: np.ndarray, delta: float, variance: float, p0: float
) -> list[tuple[float, float, float, float]]:
    """Return ``(beta, alpha, error, innovation std)`` per step, in plain Python.

    State ``[beta, alpha]`` starts at zero with covariance ``p0 I``, follows a
    random walk with covariance ``delta / (1 - delta) I`` and is observed through
    ``y = beta x + alpha`` plus noise of variance ``variance``.
    """
    q = delta / (1.0 - delta)
    beta = alpha = 0.0
    p = [[p0, 0.0], [0.0, p0]]
    rows = []
    for observed, regressor in zip(y, x, strict=True):
        r = [[p[i][j] + (q if i == j else 0.0) for j in range(2)] for i in range(2)]
        rh = [r[0][0] * regressor + r[0][1], r[1][0] * regressor + r[1][1]]
        innovation_variance = regressor * rh[0] + rh[1] + variance
        error = observed - (beta * regressor + alpha)
        gain = [rh[0] / innovation_variance, rh[1] / innovation_variance]
        beta += gain[0] * error
        alpha += gain[1] * error
        p = [[r[i][j] - gain[i] * rh[j] for j in range(2)] for i in range(2)]
        rows.append((beta, alpha, error, math.sqrt(innovation_variance)))
    return rows


class TestKalmanRecursion:
    """The predict and update steps against independent calculations."""

    def test_two_steps_match_exact_fractions(self) -> None:
        """A hand calculation in exact fractions fixes every output."""
        config = KalmanConfig(
            delta=0.5, observation_variance=0.5, initial_state_variance=1.0
        )
        result = kalman_hedge_ratio(
            _series(np.array([3.0, 2.0])), _series(np.array([2.0, 1.0])), config
        )
        # Fractions.Fraction arithmetic: step 1 has prior covariance 2 I, so the
        # innovation variance is 21/2 and the estimate (8/7, 4/7); step 2 has
        # innovation variance 43/14 and the estimate (52/43, 32/43).
        np.testing.assert_allclose(result.beta, [8 / 7, 52 / 43], rtol=1e-12)
        np.testing.assert_allclose(result.alpha, [4 / 7, 32 / 43], rtol=1e-12)
        np.testing.assert_allclose(result.spread, [3.0, 2 / 7], rtol=1e-12)
        np.testing.assert_allclose(
            result.spread_std, [math.sqrt(21 / 2), math.sqrt(43 / 14)], rtol=1e-12
        )
        np.testing.assert_allclose(
            result.zscore,
            [3.0 / math.sqrt(21 / 2), (2 / 7) / math.sqrt(43 / 14)],
            rtol=1e-12,
        )

    def test_a_longer_sample_matches_a_plain_python_recursion(self) -> None:
        """Every output equals an independent scalar implementation of the filter."""
        x = _price_like(1, 80)
        y = 1.2 * x + 0.5 + np.random.default_rng(2).normal(scale=0.2, size=80)
        config = KalmanConfig(delta=1e-3, observation_variance=0.04)
        result = kalman_hedge_ratio(_series(y), _series(x), config)
        expected = np.array(_reference_filter(y, x, 1e-3, 0.04, 1.0))
        np.testing.assert_allclose(result.beta, expected[:, 0], rtol=1e-10)
        np.testing.assert_allclose(result.alpha, expected[:, 1], rtol=1e-10)
        np.testing.assert_allclose(result.spread, expected[:, 2], rtol=1e-10)
        np.testing.assert_allclose(result.spread_std, expected[:, 3], rtol=1e-10)
        np.testing.assert_allclose(
            result.zscore, expected[:, 2] / expected[:, 3], rtol=1e-10
        )

    def test_the_spread_is_the_one_step_ahead_prediction_error(self) -> None:
        """``spread[t]`` uses the estimate made at ``t - 1``, not the one at ``t``."""
        x = _price_like(3, 200)
        y = 0.8 * x + 2.0 + np.random.default_rng(4).normal(scale=0.1, size=200)
        result = kalman_hedge_ratio(_series(y), _series(x))
        previous_fit = result.beta.shift(1) * x + result.alpha.shift(1)
        np.testing.assert_allclose(
            result.spread.iloc[1:], (_series(y) - previous_fit).iloc[1:], atol=1e-10
        )
        # The state starts at zero, so the first prediction error is y itself.
        assert result.spread.iloc[0] == pytest.approx(y[0])

    def test_results_are_series_on_the_input_index(self) -> None:
        """All five outputs are Series on ``y``'s index."""
        y, x = _series(_price_like(5, 30) * 2), _series(_price_like(5, 30))
        result = kalman_hedge_ratio(y, x)
        assert isinstance(result, KalmanResult)
        for series in (
            result.beta,
            result.alpha,
            result.spread,
            result.spread_std,
            result.zscore,
        ):
            assert isinstance(series, pd.Series)
            assert series.index.equals(y.index)
        assert (result.spread_std > 0).all()

    def test_result_is_immutable(self) -> None:
        """The result is a frozen dataclass."""
        y, x = _series(_price_like(5, 30) * 2), _series(_price_like(5, 30))
        result = kalman_hedge_ratio(y, x)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.beta = result.alpha  # type: ignore[misc]


class TestKalmanTracking:
    """How well the filter follows the hedge ratio."""

    def test_a_noise_free_constant_beta_is_recovered(self) -> None:
        """With no noise and a fixed relation, the estimate lands on the truth."""
        x = _price_like(0, 1000)
        result = kalman_hedge_ratio(_series(1.5 * x + 0.3), _series(x))
        assert result.beta.iloc[-1] == pytest.approx(1.5, abs=1e-3)
        assert result.alpha.iloc[-1] == pytest.approx(0.3, abs=1e-3)
        assert abs(result.spread.iloc[-1]) < 1e-3

    def test_a_step_change_in_beta_is_tracked_within_thirty_steps(self) -> None:
        """After beta moves from 1 to 2 the estimate is near 2 thirty steps later."""
        size, change = 1000, 500
        rng = np.random.default_rng(1)
        x = _price_like(1, size)
        beta = np.where(np.arange(size) < change, 1.0, 2.0)
        y = beta * x + 0.5 + rng.normal(scale=0.03, size=size)
        estimate = kalman_hedge_ratio(_series(y), _series(x)).beta
        assert abs(estimate.iloc[change - 1] - 1.0) < 0.02
        assert abs(estimate.iloc[change + 29] - 2.0) < 0.05

    def test_the_estimate_at_t_does_not_use_later_data(self) -> None:
        """Changing every observation after ``t`` leaves outputs up to ``t`` alone."""
        x = _price_like(6, 300)
        y = 1.1 * x + 1.0 + np.random.default_rng(7).normal(scale=0.2, size=300)
        full = kalman_hedge_ratio(_series(y), _series(x))
        y_changed, x_changed = y.copy(), x.copy()
        y_changed[200:] = y_changed[200:] * 3.0 + 50.0
        x_changed[200:] = x_changed[200:] - 10.0
        altered = kalman_hedge_ratio(_series(y_changed), _series(x_changed))
        for name in ("beta", "alpha", "spread", "spread_std", "zscore"):
            np.testing.assert_array_equal(
                getattr(full, name).iloc[:200], getattr(altered, name).iloc[:200]
            )
        truncated = kalman_hedge_ratio(_series(y[:200]), _series(x[:200]))
        np.testing.assert_array_equal(truncated.beta, full.beta.iloc[:200])

    def test_a_vanishing_delta_gives_least_squares(self) -> None:
        """With a frozen state the filter is Bayesian linear regression."""
        x = _price_like(2, 3000)
        y = 2.0 * x + 1.0 + np.random.default_rng(8).normal(scale=0.03, size=3000)
        config = KalmanConfig(delta=1e-15, observation_variance=1e-3)
        result = kalman_hedge_ratio(_series(y), _series(x), config)
        design = np.column_stack([x, np.ones_like(x)])
        ols = np.linalg.lstsq(design, y, rcond=None)[0]
        # A zero-mean prior of variance p0 adds observation_variance / p0 to the
        # normal equations, the ridge solution.
        ridge = np.linalg.solve(design.T @ design + 1e-3 * np.eye(2), design.T @ y)
        final = np.array([result.beta.iloc[-1], result.alpha.iloc[-1]])
        np.testing.assert_allclose(final, ridge, atol=1e-6)
        np.testing.assert_allclose(final, ols, atol=1e-5)

    def test_a_larger_delta_adapts_faster(self) -> None:
        """A higher state variance follows a regime change more quickly."""
        size = 700
        x = _price_like(9, size)
        beta = np.where(np.arange(size) < 500, 1.0, 2.0)
        y = beta * x + np.random.default_rng(10).normal(scale=0.03, size=size)
        errors = []
        for delta in (1e-8, 1e-5, 1e-3):
            estimate = kalman_hedge_ratio(
                _series(y), _series(x), KalmanConfig(delta=delta)
            ).beta
            errors.append(abs(estimate.iloc[505] - 2.0))
        assert errors == sorted(errors, reverse=True)


class TestKalmanZscore:
    """The standardised innovations of a correctly specified filter."""

    def test_a_stationary_spread_has_unit_variance_zscores(self) -> None:
        """Noise of variance ``observation_variance`` around fixed parameters."""
        size = 4000
        rng = np.random.default_rng(3)
        x = rng.normal(size=size)
        y = 2.0 * x + 1.0 + rng.normal(scale=math.sqrt(1e-3), size=size)
        config = KalmanConfig(delta=1e-8, observation_variance=1e-3)
        zscore = kalman_hedge_ratio(_series(y), _series(x), config).zscore.iloc[200:]
        # The standard error of a standard deviation over 3800 draws is 0.012.
        assert zscore.std() == pytest.approx(1.0, abs=0.05)
        assert zscore.mean() == pytest.approx(0.0, abs=0.05)

    def test_a_random_walk_state_gives_white_unit_variance_zscores(self) -> None:
        """When the data follow the model, the innovations are standard and white."""
        size, delta, variance = 4000, 1e-4, 1e-3
        rng = np.random.default_rng(3)
        state_step = math.sqrt(delta / (1.0 - delta))
        state = np.cumsum(rng.normal(scale=state_step, size=(size, 2)), axis=0)
        state += np.array([2.0, 1.0])
        x = _price_like(3, size)
        y = (
            state[:, 0] * x
            + state[:, 1]
            + rng.normal(scale=math.sqrt(variance), size=size)
        )
        config = KalmanConfig(delta=delta, observation_variance=variance)
        zscore = kalman_hedge_ratio(_series(y), _series(x), config).zscore.iloc[200:]
        assert zscore.std() == pytest.approx(1.0, abs=0.05)
        assert zscore.autocorr(lag=1) == pytest.approx(0.0, abs=0.06)


class TestKalmanValidation:
    """Every rejected setting or input of ``kalman_hedge_ratio``."""

    @pytest.mark.parametrize(
        "delta", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf, True, "0.5", None]
    )
    def test_delta_must_lie_strictly_between_zero_and_one(self, delta: object) -> None:
        """The state variance ``delta / (1 - delta)`` needs ``0 < delta < 1``."""
        with pytest.raises(ConfigurationError, match="delta"):
            KalmanConfig(delta=delta)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value", [0.0, -1.0, math.nan, math.inf, True, "1.0", None]
    )
    def test_observation_variance_must_be_positive_and_finite(
        self, value: object
    ) -> None:
        """A zero measurement variance would make the gain degenerate."""
        with pytest.raises(ConfigurationError, match="observation_variance"):
            KalmanConfig(observation_variance=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value", [0.0, -1.0, math.nan, math.inf, True, "1.0", None]
    )
    def test_initial_state_variance_must_be_positive_and_finite(
        self, value: object
    ) -> None:
        """The prior covariance must be a proper covariance."""
        with pytest.raises(ConfigurationError, match="initial_state_variance"):
            KalmanConfig(initial_state_variance=value)  # type: ignore[arg-type]

    def test_integer_valued_variances_are_accepted(self) -> None:
        """A whole number is a perfectly good variance."""
        config = KalmanConfig(
            observation_variance=1, initial_state_variance=np.int64(2)
        )
        assert config.observation_variance == 1

    def test_config_is_immutable(self) -> None:
        """The configuration is a frozen dataclass."""
        config = KalmanConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.delta = 0.5  # type: ignore[misc]

    def test_misaligned_series_raise(self) -> None:
        """The two series must share the same dates."""
        y, x = _series(_price_like(1, 20) * 2), _series(_price_like(1, 20))
        with pytest.raises(DataError, match="same index"):
            kalman_hedge_ratio(y.iloc[1:], x)

    def test_non_series_input_raises(self) -> None:
        """Only pandas Series carry the index the outputs are returned on."""
        with pytest.raises(DataError, match="Series"):
            kalman_hedge_ratio(np.ones(10), _series(np.ones(10)))  # type: ignore[arg-type]

    def test_non_finite_values_raise(self) -> None:
        """A missing price would silently poison every later estimate."""
        y, x = _price_like(1, 20) * 2, _price_like(1, 20)
        y[4] = np.nan
        with pytest.raises(DataError, match="finite"):
            kalman_hedge_ratio(_series(y), _series(x))

    def test_non_numeric_values_raise(self) -> None:
        """Text among the prices is a data error, not a raw ValueError."""
        y, x = _price_like(1, 20) * 2, _price_like(1, 20)
        text = _series(y).astype(object)
        text.iloc[4] = "n/a"
        with pytest.raises(DataError, match="y must contain only numbers"):
            kalman_hedge_ratio(text, _series(x))

    def test_an_empty_series_raises(self) -> None:
        """There is nothing to filter."""
        empty = pd.Series(dtype=float)
        with pytest.raises(DataError, match="at least 1"):
            kalman_hedge_ratio(empty, empty)
