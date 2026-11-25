"""Tests for the Value-at-Risk calculator.

The golden numbers come from the normal and Cornish-Fisher formulas evaluated with
``statistics.NormalDist`` and plain ``math``, and from hand-built return vectors
whose quantiles are exact, never from the module under test.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.risk import (
    VarConfig,
    VarMethod,
    VarResult,
    compute_var,
    cornish_fisher_quantile,
    rolling_var_forecasts,
)

_NORMAL = NormalDist()
_MEAN = 0.0005
_SIGMA = 0.01
# 101 evenly spaced returns, -5.0% to +5.0%: every quantile at a multiple of 1% lies
# exactly on an order statistic, so the expected values can be read off by hand.
_LADDER = np.arange(-50, 51) / 1000.0


def _normal_pdf(x: float) -> float:
    """Return the standard normal density."""
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def _sample_with_moments(mean: float, std: float, size: int = 200) -> np.ndarray:
    """Return a sample whose mean and ddof=1 standard deviation are exact."""
    raw = np.random.default_rng(3).normal(size=size)
    return mean + std * (raw - raw.mean()) / raw.std(ddof=1)


def _skewed_sample(size: int = 500, seed: int = 11) -> np.ndarray:
    """Return daily returns with negative skew and fat tails (jump days)."""
    values = np.random.default_rng(seed).normal(0.0005, 0.01, size)
    values[::25] -= 0.03
    return values


def _python_cornish_fisher(
    values: np.ndarray, confidence: float, horizon: int
) -> tuple[float, float]:
    """Return Cornish-Fisher VaR and ES from plain-Python moments.

    The expected shortfall uses the closed form of the tail integral of the
    Cornish-Fisher quantile function against the normal density.
    """
    sample = [float(v) for v in values]
    n = len(sample)
    mean = math.fsum(sample) / n
    deviations = [v - mean for v in sample]
    m2 = math.fsum(d**2 for d in deviations) / n
    skewness = math.fsum(d**3 for d in deviations) / n / m2**1.5
    kurtosis = math.fsum(d**4 for d in deviations) / n / m2**2 - 3.0
    std = math.sqrt(math.fsum(d**2 for d in deviations) / (n - 1))
    t = _NORMAL.inv_cdf(1.0 - confidence)
    z_cf = (
        t
        + (t**2 - 1.0) * skewness / 6.0
        + (t**3 - 3.0 * t) * kurtosis / 24.0
        - (2.0 * t**3 - 5.0 * t) * skewness**2 / 36.0
    )
    tail_integral = -_normal_pdf(t) * (
        1.0
        + t * skewness / 6.0
        + (t**2 - 1.0) * kurtosis / 24.0
        - (2.0 * t**2 - 1.0) * skewness**2 / 36.0
    )
    mean_h, std_h = mean * horizon, std * math.sqrt(horizon)
    var = -(mean_h + std_h * z_cf)
    es = -(mean_h + std_h * tail_integral / (1.0 - confidence))
    return var, es


def _config(
    method: VarMethod, *, confidence_level: float = 0.95, horizon: int = 1
) -> VarConfig:
    """Return a config for ``method`` with a seeded, fast Monte Carlo."""
    return VarConfig(
        method=method,
        confidence_level=confidence_level,
        horizon=horizon,
        random_state=7,
        n_simulations=20_000,
    )


class TestCornishFisherQuantile:
    """``cornish_fisher_quantile``: the expansion of a normal quantile."""

    @pytest.mark.parametrize("z", [-2.3263479, -1.6448536, 0.0, 1.2815516])
    def test_a_normal_distribution_leaves_the_quantile_unchanged(
        self, z: float
    ) -> None:
        """Zero skewness and zero excess kurtosis give back the normal quantile."""
        assert cornish_fisher_quantile(z, 0.0, 0.0) == pytest.approx(z)

    def test_matches_the_formula_for_a_case_worked_by_hand(self) -> None:
        """z=-2, S=-0.6, K=3: -2 - 0.3 - 0.25 + 0.06 = -2.49."""
        # Worked by hand from the published formula: the skewness term is -0.3, the
        # kurtosis term is -0.25 and the squared-skewness term is +0.06.
        assert cornish_fisher_quantile(-2.0, -0.6, 3.0) == pytest.approx(-2.49)

    def test_fat_tails_push_the_loss_quantile_further_out(self) -> None:
        """Positive excess kurtosis lowers the left-tail quantile."""
        # (z^3-3z)K/24 = (-2)*3/24 = -0.25 with S = 0.
        assert cornish_fisher_quantile(-2.0, 0.0, 3.0) == pytest.approx(-2.25)


class TestParametricVar:
    """``compute_var`` with the normal (parametric) method."""

    @pytest.fixture
    def sample(self) -> np.ndarray:
        """Return returns with exact mean 0.05% and standard deviation 1%."""
        return _sample_with_moments(_MEAN, _SIGMA)

    def test_matches_the_normal_formula_at_99_percent(self, sample: np.ndarray) -> None:
        """VaR=-(mu+sigma*z) and ES=-mu+sigma*phi(z)/(1-c), z=-2.3263479."""
        # NormalDist().inv_cdf(0.01) = -2.3263478740 and phi(z)/0.01 = 2.6652 by math.
        config = VarConfig(method=VarMethod.PARAMETRIC, confidence_level=0.99)
        result = compute_var(sample, config)
        assert result.var == pytest.approx(0.02276347874, rel=1e-9)
        assert result.expected_shortfall == pytest.approx(0.02615214220, rel=1e-9)

    def test_scales_the_mean_by_h_and_the_deviation_by_root_h(
        self, sample: np.ndarray
    ) -> None:
        """Ten periods: VaR=-(10 mu + sqrt(10) sigma z)."""
        config = VarConfig(
            method=VarMethod.PARAMETRIC, confidence_level=0.99, horizon=10
        )
        result = compute_var(sample, config)
        assert result.var == pytest.approx(0.06856557912, rel=1e-9)
        assert result.expected_shortfall == pytest.approx(0.07928147389, rel=1e-9)

    def test_matches_the_normal_formula_at_95_percent(self, sample: np.ndarray) -> None:
        """The default confidence level uses z=-1.6448536."""
        result = compute_var(sample, VarConfig(method=VarMethod.PARAMETRIC))
        assert result.var == pytest.approx(0.01594853627, rel=1e-9)
        assert result.expected_shortfall == pytest.approx(0.02012712808, rel=1e-9)

    def test_uses_the_sample_standard_deviation_with_one_degree_of_freedom(
        self,
    ) -> None:
        """Thirty returns of +/-1% around zero have a deviation of sqrt(30/29)%."""
        sample = np.array([-1.0, 1.0] * 15) * 0.01
        result = compute_var(sample, VarConfig(method=VarMethod.PARAMETRIC))
        std = 0.01 * math.sqrt(30.0 / 29.0)
        assert result.var == pytest.approx(-std * _NORMAL.inv_cdf(0.05), rel=1e-12)


class TestHistoricalVar:
    """``compute_var`` with the historical method."""

    def test_reads_the_exact_order_statistic(self) -> None:
        """At 95% the quantile is the 6th smallest of 101 returns, -4.5%."""
        # np.quantile position 100*0.05 = 5 -> value -0.045; the tail is the six
        # returns -0.050 ... -0.045, whose mean is -0.0475.
        result = compute_var(_LADDER)
        assert result.var == pytest.approx(0.045, abs=1e-12)
        assert result.expected_shortfall == pytest.approx(0.0475, abs=1e-12)

    def test_reads_the_exact_order_statistic_at_99_percent(self) -> None:
        """At 99% the quantile is the 2nd smallest return, -4.9%."""
        # Position 100*0.01 = 1 -> -0.049; tail {-0.050, -0.049} has mean -0.0495.
        result = compute_var(_LADDER, VarConfig(confidence_level=0.99))
        assert result.var == pytest.approx(0.049, abs=1e-12)
        assert result.expected_shortfall == pytest.approx(0.0495, abs=1e-12)

    def test_interpolates_linearly_between_order_statistics(self) -> None:
        """At 94.3% the position is 5.7: 70% of the way from -4.5% to -4.4%."""
        # -0.045 + 0.7 * 0.001 = -0.0443; the tail is still -0.050 ... -0.045.
        result = compute_var(_LADDER, VarConfig(confidence_level=0.943))
        assert result.var == pytest.approx(0.0443, abs=1e-12)
        assert result.expected_shortfall == pytest.approx(0.0475, abs=1e-12)

    def test_the_tail_keeps_a_return_that_sits_on_the_quantile(self) -> None:
        """Rounding in the quantile index must not drop the boundary return."""
        # At 93% np.quantile lands one ulp below the 8th smallest return (-0.043),
        # which a bare ``<=`` would drop: the tail is -0.050 ... -0.043, mean -0.0465.
        result = compute_var(_LADDER, VarConfig(confidence_level=0.93))
        assert result.var == pytest.approx(0.043, abs=1e-12)
        assert result.expected_shortfall == pytest.approx(0.0465, abs=1e-12)

    def test_scales_to_the_horizon_by_the_square_root_of_time(self) -> None:
        """Four periods double both measures."""
        result = compute_var(_LADDER, VarConfig(horizon=4))
        assert result.var == pytest.approx(0.09, abs=1e-12)
        assert result.expected_shortfall == pytest.approx(0.095, abs=1e-12)

    def test_does_not_depend_on_the_order_of_the_returns(self) -> None:
        """The estimate is a function of the empirical distribution only."""
        shuffled = np.random.default_rng(0).permutation(_LADDER)
        assert compute_var(shuffled) == compute_var(_LADDER)

    def test_a_profitable_sample_reports_a_negative_loss(self) -> None:
        """When even the tail gains money the VaR is negative, not clipped to zero."""
        gains = np.linspace(0.01, 0.05, 50)
        result = compute_var(gains)
        assert result.var < 0.0
        assert result.expected_shortfall < 0.0


class TestCornishFisherVar:
    """``compute_var`` with the Cornish-Fisher method."""

    @pytest.mark.parametrize(
        ("confidence", "horizon"), [(0.95, 1), (0.99, 1), (0.99, 5)]
    )
    def test_matches_a_plain_python_evaluation(
        self, confidence: float, horizon: int
    ) -> None:
        """VaR and ES agree with the formulas and a closed-form tail integral."""
        sample = _skewed_sample()
        config = VarConfig(
            method=VarMethod.CORNISH_FISHER,
            confidence_level=confidence,
            horizon=horizon,
        )
        result = compute_var(sample, config)
        var, es = _python_cornish_fisher(sample, confidence, horizon)
        assert result.var == pytest.approx(var, rel=1e-9)
        assert result.expected_shortfall == pytest.approx(es, rel=1e-8)

    def test_a_fat_tailed_sample_has_a_larger_var_than_the_normal_estimate(
        self,
    ) -> None:
        """Negative skew and excess kurtosis raise the 99% loss over the normal one."""
        sample = _skewed_sample()
        normal = compute_var(
            sample, VarConfig(method=VarMethod.PARAMETRIC, confidence_level=0.99)
        )
        adjusted = compute_var(
            sample, VarConfig(method=VarMethod.CORNISH_FISHER, confidence_level=0.99)
        )
        assert adjusted.var > normal.var

    def test_a_constant_sample_has_no_tail_to_adjust(self) -> None:
        """A point mass has no skew or kurtosis: the loss is the negated constant."""
        result = compute_var(
            np.full(40, 0.002), VarConfig(method=VarMethod.CORNISH_FISHER)
        )
        assert result.var == pytest.approx(-0.002, abs=1e-15)
        assert result.expected_shortfall == pytest.approx(-0.002, abs=1e-15)

    def test_a_sample_outside_the_validity_range_raises(self) -> None:
        """Uniform returns (excess kurtosis -1.2) make the expansion non-monotone."""
        sample = np.random.default_rng(5).uniform(-0.02, 0.02, 2000)
        with pytest.raises(DataError, match="monotone"):
            compute_var(sample, VarConfig(method=VarMethod.CORNISH_FISHER))

    def test_a_skewed_thin_tailed_sample_raises(self) -> None:
        """Squares of a uniform grid (skewness 1.0, excess kurtosis -0.2) fail too."""
        sample = np.linspace(-0.01, 0.03, 200) ** 2
        with pytest.raises(DataError, match="monotone"):
            compute_var(sample, VarConfig(method=VarMethod.CORNISH_FISHER))


class TestMonteCarloVar:
    """``compute_var`` with the bootstrap Monte Carlo method."""

    def test_compounds_the_resampled_returns(self) -> None:
        """Every draw of five -1% returns compounds to 0.99^5 - 1 = -4.9005%."""
        # With a constant sample all simulated paths are identical, so VaR = ES
        # = 1 - 0.99^5 = 0.04900499 (by math, not by the module).
        config = _config(VarMethod.MONTE_CARLO, horizon=5)
        result = compute_var(np.full(40, -0.01), config)
        assert result.var == pytest.approx(1.0 - 0.99**5, abs=1e-12)
        assert result.expected_shortfall == pytest.approx(1.0 - 0.99**5, abs=1e-12)

    def test_the_same_seed_reproduces_the_result(self) -> None:
        """A seeded bootstrap is deterministic."""
        sample = _skewed_sample()
        config = _config(VarMethod.MONTE_CARLO, horizon=5)
        assert compute_var(sample, config) == compute_var(sample, config)

    def test_different_seeds_give_different_results(self) -> None:
        """The bootstrap really draws at random."""
        sample = _skewed_sample()
        first = compute_var(
            sample,
            VarConfig(method=VarMethod.MONTE_CARLO, random_state=1, horizon=5),
        )
        second = compute_var(
            sample,
            VarConfig(method=VarMethod.MONTE_CARLO, random_state=2, horizon=5),
        )
        assert first.var != second.var
        assert first.expected_shortfall != second.expected_shortfall

    def test_without_a_seed_it_still_runs(self) -> None:
        """``random_state=None`` draws fresh entropy and returns a finite result."""
        config = VarConfig(method=VarMethod.MONTE_CARLO, n_simulations=500)
        assert math.isfinite(compute_var(_skewed_sample(), config).var)

    def test_one_period_resampling_converges_to_the_historical_estimate(self) -> None:
        """Bootstrapping a single period reproduces the empirical distribution."""
        sample = _skewed_sample()
        config = VarConfig(
            method=VarMethod.MONTE_CARLO, n_simulations=400_000, random_state=3
        )
        bootstrap = compute_var(sample, config)
        historical = compute_var(sample)
        assert bootstrap.var == pytest.approx(historical.var, rel=0.05)
        assert bootstrap.expected_shortfall == pytest.approx(
            historical.expected_shortfall, rel=0.05
        )


class TestProperties:
    """Behaviour every method has to share."""

    @pytest.fixture(params=list(VarMethod), ids=lambda method: method.value)
    def method(self, request: pytest.FixtureRequest) -> VarMethod:
        """Return each estimator in turn."""
        return request.param

    @pytest.fixture
    def sample(self) -> np.ndarray:
        """Return a skewed, fat-tailed sample with a negative mean."""
        return _skewed_sample()

    def test_var_rises_with_the_confidence_level(
        self, method: VarMethod, sample: np.ndarray
    ) -> None:
        """A more demanding confidence level never lowers the loss estimate."""
        values = [
            compute_var(sample, _config(method, confidence_level=level)).var
            for level in (0.90, 0.95, 0.99)
        ]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    def test_var_rises_with_the_horizon(
        self, method: VarMethod, sample: np.ndarray
    ) -> None:
        """A longer holding period carries more risk."""
        values = [
            compute_var(sample, _config(method, horizon=horizon)).var
            for horizon in (1, 5, 20)
        ]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_expected_shortfall_is_at_least_var(
        self, method: VarMethod, sample: np.ndarray, confidence: float
    ) -> None:
        """The average loss beyond the quantile cannot be below the quantile."""
        result = compute_var(sample, _config(method, confidence_level=confidence))
        assert result.expected_shortfall >= result.var

    def test_the_result_reports_what_was_asked(
        self, method: VarMethod, sample: np.ndarray
    ) -> None:
        """The result echoes the settings and the sample size."""
        result = compute_var(sample, _config(method, confidence_level=0.97, horizon=3))
        assert isinstance(result, VarResult)
        assert result.method is method
        assert result.confidence_level == 0.97
        assert result.horizon == 3
        assert result.n_observations == sample.size

    def test_a_series_a_list_and_an_array_agree(
        self, method: VarMethod, sample: np.ndarray
    ) -> None:
        """Every array-like input produces the same result."""
        config = _config(method)
        from_array = compute_var(sample, config)
        assert compute_var(pd.Series(sample), config) == from_array
        assert compute_var(sample.tolist(), config) == from_array

    def test_the_default_config_is_historical_at_95_percent(
        self, sample: np.ndarray
    ) -> None:
        """Omitting the config is the same as passing ``VarConfig()``."""
        assert compute_var(sample) == compute_var(sample, VarConfig())


class TestComputeVarValidation:
    """``compute_var`` rejects data it cannot estimate from."""

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (np.zeros((60, 2)), "one-dimensional"),
            (np.full(29, 0.01), "at least 30"),
            (np.array([0.01, np.nan] * 20), "finite"),
            (np.array([0.01, np.inf] * 20), "finite"),
        ],
    )
    def test_unusable_returns_raise(self, values: np.ndarray, message: str) -> None:
        """Non-finite, short or 2-D input is rejected with a reason."""
        with pytest.raises(DataError, match=message):
            compute_var(values)

    def test_the_minimum_length_follows_the_config(self) -> None:
        """A smaller ``min_observations`` accepts a shorter sample."""
        values = np.linspace(-0.02, 0.02, 12)
        assert compute_var(values, VarConfig(min_observations=10)).n_observations == 12
        with pytest.raises(DataError, match="at least 20"):
            compute_var(values, VarConfig(min_observations=20))


class TestRollingVarForecasts:
    """``rolling_var_forecasts``: one-step-ahead forecasts without look-ahead."""

    @pytest.fixture
    def returns(self) -> pd.Series:
        """Return 400 skewed daily returns on business days."""
        index = pd.bdate_range("2020-01-01", periods=400)
        return pd.Series(_skewed_sample(400, seed=2), index=index, name="portfolio")

    def test_each_forecast_is_the_var_of_the_preceding_window(
        self, returns: pd.Series
    ) -> None:
        """The value at t equals ``compute_var`` on exactly the window before t."""
        window = 100
        config = VarConfig(confidence_level=0.99)
        forecasts = rolling_var_forecasts(returns, window, config)
        for position in (window, window + 1, 250, len(returns) - 1):
            expected = compute_var(
                returns.to_numpy()[position - window : position], config
            ).var
            assert forecasts.iloc[position] == pytest.approx(expected, rel=1e-12)

    def test_a_periodic_series_gives_the_hand_computed_forecast(self) -> None:
        """Any 101 consecutive values of a period-101 ladder have the 4.5% VaR."""
        # Each window holds every ladder value once, so the 95% VaR is 0.045 (the
        # exact order statistic) at every date.
        series = pd.Series(np.tile(_LADDER, 3))
        forecasts = rolling_var_forecasts(series, 101)
        assert forecasts.iloc[101:].to_numpy() == pytest.approx(0.045, abs=1e-12)

    def test_the_first_forecast_comes_after_the_first_window(
        self, returns: pd.Series
    ) -> None:
        """Dates before ``window`` have no forecast, and the index is kept."""
        forecasts = rolling_var_forecasts(returns, 60)
        assert forecasts.iloc[:60].isna().all()
        assert forecasts.iloc[60:].notna().all()
        assert forecasts.index.equals(returns.index)

    @pytest.mark.parametrize("method", list(VarMethod))
    def test_changing_returns_at_or_after_t_leaves_the_forecast_at_t(
        self, returns: pd.Series, method: VarMethod
    ) -> None:
        """No look-ahead: the forecast at t does not see the return at t."""
        t, window = 160, 100
        config = _config(method)
        baseline = rolling_var_forecasts(returns, window, config)
        altered_values = returns.to_numpy().copy()
        altered_values[t:] = _skewed_sample(len(returns) - t, seed=99)
        altered_values[t] = -0.06
        altered = rolling_var_forecasts(
            pd.Series(altered_values, index=returns.index), window, config
        )
        pd.testing.assert_series_equal(baseline.iloc[: t + 1], altered.iloc[: t + 1])
        assert baseline.iloc[t + 1] != altered.iloc[t + 1]

    def test_a_seeded_monte_carlo_forecast_is_reproducible(
        self, returns: pd.Series
    ) -> None:
        """The same seed gives the same forecast path."""
        config = VarConfig(
            method=VarMethod.MONTE_CARLO, n_simulations=200, random_state=4
        )
        first = rolling_var_forecasts(returns.iloc[:80], 40, config)
        second = rolling_var_forecasts(returns.iloc[:80], 40, config)
        pd.testing.assert_series_equal(first, second)

    def test_an_array_like_input_gets_a_range_index(self) -> None:
        """A plain array is accepted and indexed by position."""
        forecasts = rolling_var_forecasts(_skewed_sample(60), 40)
        assert list(forecasts.index) == list(range(60))
        assert forecasts.iloc[:40].isna().all()

    def test_a_multi_period_config_raises(self, returns: pd.Series) -> None:
        """One-step forecasts cannot be built from a multi-period VaR."""
        with pytest.raises(ConfigurationError, match="horizon"):
            rolling_var_forecasts(returns, 60, VarConfig(horizon=5))

    def test_a_window_shorter_than_the_minimum_raises(self, returns: pd.Series) -> None:
        """Each window must hold enough returns for ``compute_var``."""
        with pytest.raises(ConfigurationError, match="window"):
            rolling_var_forecasts(returns, 20, VarConfig(min_observations=30))

    @pytest.mark.parametrize("window", [60.0, 60.5, math.nan, True, "60", None])
    def test_a_window_that_is_not_an_integer_raises(
        self, returns: pd.Series, window: object
    ) -> None:
        """A float, NaN, bool, string or None window fails as a bad setting."""
        with pytest.raises(ConfigurationError, match="window"):
            rolling_var_forecasts(returns, window)  # type: ignore[arg-type]

    def test_a_numpy_integer_window_is_accepted(self, returns: pd.Series) -> None:
        """A window taken from a numpy array is a whole number all the same."""
        forecasts = rolling_var_forecasts(returns, np.int64(60))
        assert forecasts.iloc[60:].notna().all()

    @pytest.mark.parametrize("config", [{"confidence_level": 0.99}, "historical", 0.95])
    def test_a_config_of_the_wrong_type_raises(
        self, returns: pd.Series, config: object
    ) -> None:
        """Both public estimators reject a config that is not a ``VarConfig``."""
        with pytest.raises(ConfigurationError, match="config"):
            rolling_var_forecasts(returns, 60, config)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="config"):
            compute_var(returns, config)  # type: ignore[arg-type]

    def test_a_window_equal_to_the_minimum_is_accepted(
        self, returns: pd.Series
    ) -> None:
        """The boundary window is valid."""
        forecasts = rolling_var_forecasts(returns, 30, VarConfig(min_observations=30))
        assert forecasts.notna().sum() == len(returns) - 30

    def test_a_series_not_longer_than_the_window_raises(
        self, returns: pd.Series
    ) -> None:
        """With no date left after the first window there is nothing to forecast."""
        with pytest.raises(DataError, match="longer than the window"):
            rolling_var_forecasts(returns.iloc[:60], 60)

    def test_missing_values_raise(self, returns: pd.Series) -> None:
        """A NaN inside the series is rejected rather than skipped."""
        broken = returns.copy()
        broken.iloc[200] = np.nan
        with pytest.raises(DataError, match="finite"):
            rolling_var_forecasts(broken, 60)

    def test_a_two_dimensional_input_raises(self) -> None:
        """Only one series of returns is accepted."""
        with pytest.raises(DataError, match="one-dimensional"):
            rolling_var_forecasts(np.zeros((100, 2)), 40)
