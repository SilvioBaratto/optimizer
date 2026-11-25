"""Tests for the Engle-Granger cointegration test and the Ornstein-Uhlenbeck half-life.

The golden numbers come from computations written in this file with plain numpy
loops and ``math`` (the closed-form OLS, the Augmented Dickey-Fuller regression
built row by row, the Akaike criterion over a common sample) and from textbook
values (the asymptotic MacKinnon critical values, the AR(1) half-life formula),
never from the module under test.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.statarb import (
    EngleGrangerConfig,
    EngleGrangerResult,
    engle_granger,
    ou_half_life,
)

# Asymptotic MacKinnon (2010) values for the two-variable Engle-Granger test with a
# constant: the numbers the task states and statsmodels' ``coint`` tabulates.
_CRITICAL_VALUES = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


def _index(size: int) -> pd.DatetimeIndex:
    """Return a business-day index of ``size`` dates."""
    return pd.date_range("2020-01-01", periods=size, freq="B")


def _cointegrated_pair(
    seed: int, size: int = 500, phi: float = 0.5
) -> tuple[pd.Series, pd.Series]:
    """Return ``y = 1 + 2 x + u`` with ``x`` a random walk and ``u`` AR(1)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=size))
    shocks = rng.normal(size=size)
    noise = np.zeros(size)
    for t in range(1, size):
        noise[t] = phi * noise[t - 1] + shocks[t]
    index = _index(size)
    return pd.Series(1.0 + 2.0 * x + noise, index), pd.Series(x, index)


def _independent_walks(seed: int, size: int = 500) -> tuple[pd.Series, pd.Series]:
    """Return two unrelated random walks."""
    rng = np.random.default_rng(seed)
    index = _index(size)
    y = pd.Series(np.cumsum(rng.normal(size=size)), index)
    x = pd.Series(np.cumsum(rng.normal(size=size)), index)
    return y, x


def _reference_ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Return the intercept and slope of ``y = a + b x`` from the closed form."""
    x_mean, y_mean = x.mean(), y.mean()
    slope = float(((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum())
    return float(y_mean - slope * x_mean), slope


def _reference_adf(
    resid: np.ndarray, lags: int, first: int
) -> tuple[float, float, int, int]:
    """Return t-statistic, RSS, rows and columns of the no-constant ADF regression.

    Row ``j`` regresses ``d[j]`` on ``e[j]`` and ``d[j-1] ... d[j-lags]`` where
    ``d`` holds the first differences; ``first`` is the first row used.
    """
    diffs = np.diff(resid)
    rows = [
        [resid[j]] + [diffs[j - i] for i in range(1, lags + 1)]
        for j in range(first, len(diffs))
    ]
    design = np.array(rows)
    target = diffs[first:]
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ target
    errors = target - design @ coefficients
    rss = float(errors @ errors)
    n_rows, n_columns = design.shape
    variance = rss / (n_rows - n_columns)
    t_stat = float(coefficients[0] / math.sqrt(variance * inverse[0, 0]))
    return t_stat, rss, n_rows, n_columns


def _reference_residuals(y: pd.Series, x: pd.Series) -> np.ndarray:
    """Return the residuals of the closed-form OLS of ``y`` on ``x``."""
    y_values, x_values = y.to_numpy(), x.to_numpy()
    intercept, slope = _reference_ols(y_values, x_values)
    return y_values - intercept - slope * x_values


def _reference_aic_lag(resid: np.ndarray, max_lags: int) -> int:
    """Return the lag in ``0..max_lags`` with the lowest AIC on a common sample."""
    scores = []
    for lags in range(max_lags + 1):
        _, rss, rows, columns = _reference_adf(resid, lags, first=max_lags)
        scores.append(rows * math.log(rss / rows) + 2 * columns)
    return int(np.argmin(scores))


def _schwert(size: int) -> int:
    """Return the Schwert (1989) rule ``floor(12 (T / 100) ** 0.25)``."""
    return math.floor(12 * (size / 100) ** 0.25)


class TestEngleGrangerRegression:
    """Step 1 of ``engle_granger``: the OLS hedge ratio and the spread."""

    def test_hedge_ratio_and_intercept_equal_the_closed_form(self) -> None:
        """Slope and intercept match the covariance formula."""
        y, x = _cointegrated_pair(seed=0)
        result = engle_granger(y, x)
        intercept, slope = _reference_ols(y.to_numpy(), x.to_numpy())
        assert result.hedge_ratio == pytest.approx(slope, abs=1e-10)
        assert result.intercept == pytest.approx(intercept, abs=1e-10)

    def test_hedge_ratio_is_close_to_the_true_value(self) -> None:
        """The cointegrating regression is super-consistent around 2."""
        y, x = _cointegrated_pair(seed=0)
        result = engle_granger(y, x)
        resid = _reference_residuals(y, x)
        # Ten times the naive standard errors is a wide band: the slope error of a
        # cointegrating regression shrinks like 1 / T, faster than the naive formula.
        n = len(x)
        sxx = float(((x - x.mean()) ** 2).sum())
        slope_tolerance = 10 * resid.std(ddof=2) / math.sqrt(sxx)
        intercept_tolerance = (
            10 * resid.std(ddof=2) * math.sqrt(1 / n + float(x.mean()) ** 2 / sxx)
        )
        assert abs(result.hedge_ratio - 2.0) < slope_tolerance
        assert abs(result.intercept - 1.0) < intercept_tolerance

    def test_spread_is_the_ols_residual_on_the_input_index(self) -> None:
        """``spread = y - intercept - hedge_ratio * x`` and keeps ``y``'s index."""
        y, x = _cointegrated_pair(seed=1)
        result = engle_granger(y, x)
        np.testing.assert_allclose(
            result.spread.to_numpy(), _reference_residuals(y, x), atol=1e-9
        )
        assert result.spread.index.equals(y.index)
        assert result.spread.mean() == pytest.approx(0.0, abs=1e-9)

    def test_result_is_immutable(self) -> None:
        """The result is a frozen dataclass."""
        y, x = _cointegrated_pair(seed=0)
        result = engle_granger(y, x)
        assert isinstance(result, EngleGrangerResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.hedge_ratio = 0.0  # type: ignore[misc]


class TestEngleGrangerAdf:
    """Step 2 of ``engle_granger``: the ADF statistic on the residuals."""

    @pytest.mark.parametrize("lags", [0, 1, 3])
    def test_statistic_equals_an_independent_regression(self, lags: int) -> None:
        """The t-statistic of ``e[t-1]`` matches normal equations written here."""
        y, x = _cointegrated_pair(seed=2)
        result = engle_granger(y, x, EngleGrangerConfig(adf_lags=lags))
        expected, *_ = _reference_adf(_reference_residuals(y, x), lags, first=lags)
        assert result.adf_statistic == pytest.approx(expected, abs=1e-8)
        assert result.adf_lags == lags

    def test_statistic_equals_an_independent_regression_for_a_random_walk_pair(
        self,
    ) -> None:
        """The same identity holds when the residuals are not stationary."""
        y, x = _independent_walks(seed=3)
        result = engle_granger(y, x, EngleGrangerConfig(adf_lags=2))
        expected, *_ = _reference_adf(_reference_residuals(y, x), 2, first=2)
        assert result.adf_statistic == pytest.approx(expected, abs=1e-8)

    def test_lag_selection_follows_the_akaike_criterion(self) -> None:
        """``adf_lags=None`` picks the AIC minimiser over ``0..max_lags``."""
        y, x = _cointegrated_pair(seed=4, phi=0.7)
        result = engle_granger(y, x, EngleGrangerConfig(max_lags=6))
        resid = _reference_residuals(y, x)
        chosen = _reference_aic_lag(resid, 6)
        expected, *_ = _reference_adf(resid, chosen, first=chosen)
        assert result.adf_lags == chosen
        assert result.adf_statistic == pytest.approx(expected, abs=1e-8)

    def test_default_maximum_lag_is_the_schwert_rule(self) -> None:
        """Without ``max_lags`` the search runs up to ``floor(12 (T/100)^0.25)``."""
        y, x = _cointegrated_pair(seed=5, size=500)
        result = engle_granger(y, x)
        # Hand calculation: 5 ** 0.25 = 1.4953, 12 * 1.4953 = 17.94, floor is 17.
        assert _schwert(500) == 17
        assert result.adf_lags == _reference_aic_lag(_reference_residuals(y, x), 17)

    def test_explicit_lags_ignore_the_search(self) -> None:
        """A fixed ``adf_lags`` is used even when ``max_lags`` is also given."""
        y, x = _cointegrated_pair(seed=5)
        config = EngleGrangerConfig(adf_lags=2, max_lags=9)
        assert engle_granger(y, x, config).adf_lags == 2

    def test_a_search_limit_larger_than_the_sample_allows_is_capped(self) -> None:
        """``max_lags`` never exceeds what half the sample can support."""
        y, x = _cointegrated_pair(seed=6, size=60)
        result = engle_granger(y, x, EngleGrangerConfig(max_lags=500))
        assert 0 <= result.adf_lags <= (60 - 4) // 2

    def test_lags_that_leave_no_degrees_of_freedom_raise(self) -> None:
        """More lags than the regression has rows cannot be estimated."""
        y, x = _cointegrated_pair(seed=6, size=50)
        with pytest.raises(DataError, match="degrees of freedom"):
            engle_granger(y, x, EngleGrangerConfig(adf_lags=48))

    def test_identical_series_have_no_statistic(self) -> None:
        """A spread that is identically zero has no ADF statistic."""
        x = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=80)), _index(80))
        with pytest.raises(DataError, match="degenerate"):
            engle_granger(x.copy(), x, EngleGrangerConfig(adf_lags=1))


class TestEngleGrangerDecision:
    """The cointegration decision against the MacKinnon critical values."""

    def test_critical_values_are_the_asymptotic_mackinnon_numbers(self) -> None:
        """The result carries the 1%, 5% and 10% values stated in the docstring."""
        y, x = _cointegrated_pair(seed=0)
        assert engle_granger(y, x).critical_values == _CRITICAL_VALUES

    def test_critical_values_are_a_copy(self) -> None:
        """Editing one result's dictionary does not change the next result."""
        y, x = _cointegrated_pair(seed=0)
        first = engle_granger(y, x)
        first.critical_values["5%"] = 0.0
        assert engle_granger(y, x).critical_values["5%"] == -3.34

    def test_cointegrated_pair_is_detected(self) -> None:
        """An AR(1) spread with coefficient 0.5 gives a strongly negative statistic."""
        y, x = _cointegrated_pair(seed=0)
        result = engle_granger(y, x)
        assert result.adf_statistic < -3.34
        assert result.is_cointegrated is True

    def test_half_life_is_the_half_life_of_the_spread(self) -> None:
        """``half_life`` equals ``ou_half_life(spread)`` and is near one period."""
        y, x = _cointegrated_pair(seed=0)
        result = engle_granger(y, x)
        assert result.half_life == ou_half_life(result.spread)
        # An AR(1) coefficient of 0.5 halves a shock every -ln 2 / ln 0.5 = 1 period.
        assert result.half_life == pytest.approx(1.0, abs=0.4)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_independent_random_walks_are_not_cointegrated(self, seed: int) -> None:
        """Seeds 0, 1 and 2 give statistics near -1.7, -2.1 and -1.7, above -3.0."""
        y, x = _independent_walks(seed)
        result = engle_granger(y, x)
        assert result.adf_statistic > -3.0
        assert result.is_cointegrated is False

    def test_the_test_has_the_nominal_size_on_independent_random_walks(self) -> None:
        """Over 400 unrelated pairs the 5% test rejects about 5% of the time."""
        rejections = sum(
            engle_granger(*_independent_walks(seed, 250)).is_cointegrated
            for seed in range(1000, 1400)
        )
        # The binomial standard error at 5% is 1.1 points; the band is 3 of them.
        assert 0.02 <= rejections / 400 <= 0.09

    def test_the_test_has_power_against_a_cointegrated_pair(self) -> None:
        """Every one of 100 cointegrated pairs of length 250 is detected."""
        detections = sum(
            engle_granger(*_cointegrated_pair(seed, 250)).is_cointegrated
            for seed in range(2000, 2100)
        )
        assert detections == 100

    def test_the_configured_significance_selects_the_critical_value(self) -> None:
        """A statistic of about -3.54 passes at 10% and 5% but not at 1%."""
        y, x = _cointegrated_pair(seed=0, phi=0.95)
        statistic = engle_granger(y, x).adf_statistic
        assert -3.90 < statistic < -3.34
        decisions = {
            level: engle_granger(
                y, x, EngleGrangerConfig(significance=level)
            ).is_cointegrated
            for level in _CRITICAL_VALUES
        }
        assert decisions == {"1%": False, "5%": True, "10%": True}


class TestEngleGrangerValidation:
    """Every rejected setting or input of ``engle_granger``."""

    def test_unequal_lengths_raise(self) -> None:
        """Series of different lengths cannot be paired."""
        y, x = _cointegrated_pair(seed=0)
        with pytest.raises(DataError, match="same index"):
            engle_granger(y.iloc[:-1], x)

    def test_different_indexes_raise(self) -> None:
        """Equal lengths with different dates are a misalignment."""
        y, x = _cointegrated_pair(seed=0)
        shifted = x.set_axis(x.index + pd.Timedelta(days=1))
        with pytest.raises(DataError, match="same index"):
            engle_granger(y, shifted)

    def test_non_series_input_raises(self) -> None:
        """Only pandas Series carry the index the spread is returned on."""
        y, x = _cointegrated_pair(seed=0)
        with pytest.raises(DataError, match="Series"):
            engle_granger(y.to_numpy(), x)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_values_raise(self, bad: float) -> None:
        """Missing or infinite prices are rejected rather than propagated."""
        y, x = _cointegrated_pair(seed=0)
        y.iloc[10] = bad
        with pytest.raises(DataError, match="finite"):
            engle_granger(y, x)

    def test_too_few_observations_raise(self) -> None:
        """Fewer than ``min_observations`` rows are rejected."""
        y, x = _cointegrated_pair(seed=0, size=49)
        with pytest.raises(DataError, match="at least 50"):
            engle_granger(y, x)

    def test_min_observations_is_configurable(self) -> None:
        """A lower floor lets a shorter sample through."""
        y, x = _cointegrated_pair(seed=0, size=40)
        result = engle_granger(y, x, EngleGrangerConfig(min_observations=30))
        assert math.isfinite(result.adf_statistic)

    def test_constant_regressor_raises(self) -> None:
        """A constant ``x`` makes the hedge ratio undefined."""
        y, _ = _cointegrated_pair(seed=0)
        flat = pd.Series(1.0, index=y.index)
        with pytest.raises(DataError, match="constant"):
            engle_granger(y, flat)

    def test_negative_adf_lags_raise(self) -> None:
        """A lag count below zero is meaningless."""
        with pytest.raises(ConfigurationError, match="adf_lags"):
            EngleGrangerConfig(adf_lags=-1)

    def test_negative_max_lags_raise(self) -> None:
        """A search limit below zero is meaningless."""
        with pytest.raises(ConfigurationError, match="max_lags"):
            EngleGrangerConfig(max_lags=-1)

    def test_unknown_significance_raises(self) -> None:
        """Only the tabulated levels have a critical value."""
        with pytest.raises(ConfigurationError, match="significance"):
            EngleGrangerConfig(significance="2%")

    def test_min_observations_below_the_floor_raises(self) -> None:
        """The ADF regression needs a handful of rows at the very least."""
        with pytest.raises(ConfigurationError, match="min_observations"):
            EngleGrangerConfig(min_observations=5)

    def test_config_is_immutable(self) -> None:
        """The configuration is a frozen dataclass."""
        config = EngleGrangerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.adf_lags = 1  # type: ignore[misc]


def _ar1_path(seed: int, size: int, phi: float) -> np.ndarray:
    """Return a zero-mean AR(1) path with unit shocks."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=size)
    path = np.zeros(size)
    for t in range(1, size):
        path[t] = phi * path[t - 1] + shocks[t]
    return path


def _reference_slope(series: np.ndarray) -> float:
    """Return the slope of ``diff(s)`` on ``s[:-1]`` (with a constant)."""
    lagged, change = series[:-1], np.diff(series)
    centred = lagged - lagged.mean()
    return float((centred * (change - change.mean())).sum() / (centred**2).sum())


class TestOuHalfLife:
    """``ou_half_life``: the mean-reversion speed of a series."""

    def test_geometric_decay_has_the_textbook_half_life(self) -> None:
        """A path with ``s[t] = 0.9 s[t-1]`` halves every -ln 2 / ln 0.9 periods."""
        path = 0.9 ** np.arange(60)
        # -ln 2 / ln 0.9 = 6.5788 from the textbook formula.
        assert ou_half_life(path) == pytest.approx(math.log(2) / -math.log(0.9))

    def test_noisy_ar1_matches_the_textbook_value_within_sampling_error(self) -> None:
        """An AR(1) with coefficient 0.9 has a half-life near 6.58 periods."""
        size, phi = 20_000, 0.9
        half_life = ou_half_life(pd.Series(_ar1_path(7, size, phi)))
        expected = math.log(2) / -math.log(phi)
        # Delta method: d(half-life)/d(phi) = ln 2 / (phi ln(phi)^2), and the
        # standard error of the AR(1) estimate is sqrt((1 - phi^2) / T).
        slope = math.log(2) / (phi * math.log(phi) ** 2)
        tolerance = 4 * slope * math.sqrt((1 - phi**2) / size)
        assert half_life == pytest.approx(expected, abs=tolerance)

    def test_a_faster_reversion_has_a_shorter_half_life(self) -> None:
        """The half-life increases with the AR(1) coefficient."""
        lives = [ou_half_life(_ar1_path(8, 5_000, phi)) for phi in (0.5, 0.8, 0.95)]
        assert lives == sorted(lives)

    @pytest.mark.parametrize(
        "path",
        [
            np.arange(100, dtype=float),
            1.01 ** np.arange(100),
            (-1.0) ** np.arange(100),
        ],
        ids=["linear trend", "geometric growth", "alternating"],
    )
    def test_series_that_do_not_revert_have_an_infinite_half_life(
        self, path: np.ndarray
    ) -> None:
        """A zero or positive slope, or one at or below -1, is not mean reverting."""
        assert ou_half_life(path) == math.inf

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 31])
    def test_a_random_walk_has_a_long_or_infinite_half_life(self, seed: int) -> None:
        """The sample slope of a unit root is almost always slightly negative.

        Seeds 0 to 4 give slopes between -0.0174 and -0.0017, hence half-lives of
        about 40 to 400 periods. Of the seeds 0 to 39 only seed 31 gives a
        non-negative slope, and only there is the half-life infinite.
        """
        walk = np.cumsum(np.random.default_rng(seed).normal(size=500))
        slope = _reference_slope(walk)
        expected = math.inf if slope >= 0.0 else -math.log(2) / math.log(1 + slope)
        assert (seed == 31) == (slope >= 0.0)
        assert ou_half_life(walk) == pytest.approx(expected)
        assert expected > 30

    def test_series_and_array_inputs_agree(self) -> None:
        """A Series and the same values as an array give the same half-life."""
        path = _ar1_path(9, 300, 0.8)
        assert ou_half_life(pd.Series(path)) == ou_half_life(path)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_values_raise(self, bad: float) -> None:
        """Missing or infinite values are rejected."""
        path = _ar1_path(9, 50, 0.8)
        path[5] = bad
        with pytest.raises(DataError, match="finite"):
            ou_half_life(path)

    def test_too_few_observations_raise(self) -> None:
        """Two points give a line that fits exactly, which is no estimate."""
        with pytest.raises(DataError, match="at least 3"):
            ou_half_life(np.array([1.0, 0.5]))

    def test_a_constant_series_raises(self) -> None:
        """A constant series has no variation to regress on."""
        with pytest.raises(DataError, match="constant"):
            ou_half_life(np.ones(20))

    def test_a_two_dimensional_input_raises(self) -> None:
        """Only one-dimensional series are accepted."""
        with pytest.raises(DataError, match="one-dimensional"):
            ou_half_life(np.ones((10, 2)))
