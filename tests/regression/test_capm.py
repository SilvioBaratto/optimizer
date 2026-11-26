"""Tests for the CAPM report and the rolling beta.

The golden numbers come from exact fraction arithmetic and the ``statistics`` module
in a scratch script, from closed forms written out in this file, and from scipy's
Student t survival function.
"""

from __future__ import annotations

import dataclasses
import math
import statistics

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.regression import CapmResult, capm_regression, ols, rolling_beta

_YEAR = 252
_DATES = pd.date_range("2023-01-02", periods=6, freq="B")
_MARKET = pd.Series([0.010, 0.020, -0.020, 0.020, 0.000, -0.010], index=_DATES)
_ASSET = pd.Series([0.025, 0.015, -0.025, 0.035, 0.005, -0.015], index=_DATES)
_MONTHLY_RISK_FREE = 0.001
_MONTHS = 12


def _simple_fit(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float, float, float]:
    """Return intercept, slope and their standard errors by the closed form."""
    n_obs = x.size
    sxx = float(((x - x.mean()) ** 2).sum())
    slope = float(((x - x.mean()) * (y - y.mean())).sum()) / sxx
    intercept = float(y.mean() - slope * x.mean())
    residuals = y - intercept - slope * x
    sigma_squared = float(residuals @ residuals) / (n_obs - 2)
    se_slope = math.sqrt(sigma_squared / sxx)
    se_intercept = math.sqrt(sigma_squared * (1.0 / n_obs + x.mean() ** 2 / sxx))
    return intercept, slope, se_intercept, se_slope


def _simulate(
    n_obs: int = 1500, alpha: float = 0.0005, beta: float = 1.2
) -> tuple[pd.Series, pd.Series]:
    """Return a seeded market and an asset with a known alpha and beta."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", periods=n_obs, freq="B")
    market = rng.normal(0.0003, 0.01, n_obs)
    asset = alpha + beta * market + rng.normal(0.0, 0.004, n_obs)
    return pd.Series(asset, index=dates), pd.Series(market, index=dates)


class TestHandMadeCase:
    """Six monthly returns worked out by hand."""

    def test_every_statistic_matches_the_exact_calculation(self) -> None:
        """The report equals the figures computed with exact fractions."""
        result = capm_regression(
            _ASSET, _MARKET, risk_free_rate=_MONTHLY_RISK_FREE, periods_per_year=_MONTHS
        )
        # Scratch script with fractions.Fraction: excess returns, then the simple
        # regression (alpha 0.002575, beta 1.325), its standard errors from
        # s^2 = 8.5625e-05, and the annualised ratios with math.sqrt.
        assert result.alpha == pytest.approx(0.002575, rel=1e-9)
        assert result.annualized_alpha == pytest.approx(0.0309, rel=1e-9)
        assert result.alpha_t_statistic == pytest.approx(0.6734363971393097, rel=1e-9)
        assert result.beta == pytest.approx(1.325, rel=1e-9)
        assert result.beta_std_error == pytest.approx(0.2534141866589162, rel=1e-9)
        assert result.beta_t_statistic == pytest.approx(5.228594410870095, rel=1e-9)
        assert result.r_squared == pytest.approx(0.872360248447205, rel=1e-9)
        assert result.treynor_ratio == pytest.approx(0.05132075471698114, rel=1e-9)
        assert result.tracking_error == pytest.approx(0.0340587727318528, rel=1e-9)
        assert result.information_ratio == pytest.approx(1.174440439029407, rel=1e-9)
        assert result.n_observations == 6

    def test_p_values_come_from_the_student_t_distribution(self) -> None:
        """Both p-values use four residual degrees of freedom."""
        result = capm_regression(
            _ASSET, _MARKET, risk_free_rate=_MONTHLY_RISK_FREE, periods_per_year=_MONTHS
        )
        # scipy's t survival function evaluated in the test, not by the module.
        assert result.alpha_p_value == pytest.approx(
            2.0 * stats.t.sf(0.6734363971393097, 4), rel=1e-9
        )

    def test_the_treynor_ratio_is_the_annual_excess_return_per_unit_of_beta(
        self,
    ) -> None:
        """Treynor equals mean excess return times periods per year over beta."""
        excess = [a - _MONTHLY_RISK_FREE for a in _ASSET]
        market_excess = [m - _MONTHLY_RISK_FREE for m in _MARKET]
        beta = statistics.covariance(excess, market_excess) / statistics.variance(
            market_excess
        )
        result = capm_regression(
            _ASSET, _MARKET, risk_free_rate=_MONTHLY_RISK_FREE, periods_per_year=_MONTHS
        )
        assert result.treynor_ratio == pytest.approx(
            statistics.fmean(excess) * _MONTHS / beta, rel=1e-9
        )

    def test_tracking_error_and_information_ratio_describe_the_active_return(
        self,
    ) -> None:
        """Both use the annualised mean and deviation of asset minus market."""
        active = [a - m for a, m in zip(_ASSET, _MARKET, strict=True)]
        tracking_error = statistics.stdev(active) * math.sqrt(_MONTHS)
        result = capm_regression(
            _ASSET, _MARKET, risk_free_rate=_MONTHLY_RISK_FREE, periods_per_year=_MONTHS
        )
        assert result.tracking_error == pytest.approx(tracking_error, rel=1e-9)
        assert result.information_ratio == pytest.approx(
            statistics.fmean(active) * _MONTHS / tracking_error, rel=1e-9
        )

    def test_the_result_is_immutable(self) -> None:
        """A shared report cannot be changed behind a caller's back."""
        result = capm_regression(_ASSET, _MARKET)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.beta = 0.0  # type: ignore[misc]


class TestSimulatedData:
    """A seeded series with a known alpha and beta."""

    def test_the_known_alpha_and_beta_are_recovered(self) -> None:
        """Both estimates lie within four standard errors of the truth."""
        asset, market = _simulate()
        _, _, se_alpha, se_beta = _simple_fit(asset.to_numpy(), market.to_numpy())
        result = capm_regression(asset, market)
        assert abs(result.alpha - 0.0005) < 4.0 * se_alpha
        assert abs(result.beta - 1.2) < 4.0 * se_beta
        assert result.beta_std_error == pytest.approx(se_beta, rel=1e-9)

    def test_the_estimates_equal_the_closed_form(self) -> None:
        """Alpha, beta, t-statistics and R-squared follow the textbook formulas."""
        asset, market = _simulate()
        y, x = asset.to_numpy(), market.to_numpy()
        intercept, slope, se_alpha, se_beta = _simple_fit(y, x)
        result = capm_regression(asset, market)
        assert result.alpha == pytest.approx(intercept, rel=1e-9)
        assert result.beta == pytest.approx(slope, rel=1e-9)
        assert result.alpha_t_statistic == pytest.approx(intercept / se_alpha, rel=1e-9)
        assert result.beta_t_statistic == pytest.approx(slope / se_beta, rel=1e-9)
        assert result.r_squared == pytest.approx(
            float(np.corrcoef(x, y)[0, 1] ** 2), rel=1e-9
        )
        assert result.annualized_alpha == pytest.approx(intercept * _YEAR, rel=1e-9)
        assert result.n_observations == len(asset)

    def test_a_longer_sample_tightens_the_beta_estimate(self) -> None:
        """The standard error of beta shrinks as observations accumulate."""
        asset, market = _simulate(n_obs=1500)
        short = capm_regression(asset.iloc[:300], market.iloc[:300])
        full = capm_regression(asset, market)
        assert full.beta_std_error < short.beta_std_error

    def test_newey_west_lags_change_only_the_standard_errors(self) -> None:
        """hac_lags is passed through to the regression."""
        asset, market = _simulate()
        classical = capm_regression(asset, market)
        robust = capm_regression(asset, market, hac_lags=5)
        reference = ols(asset, market.rename("m"), hac_lags=5)
        assert robust.beta == pytest.approx(classical.beta, rel=1e-12)
        assert robust.beta_std_error == pytest.approx(
            float(reference.std_errors["m"]), rel=1e-9
        )
        assert robust.beta_std_error != pytest.approx(classical.beta_std_error)


class TestRiskFreeRate:
    """The regression uses excess returns, so Jensen's alpha is measured net."""

    def test_an_asset_that_earns_the_capm_return_has_no_alpha(self) -> None:
        """asset = rf + 1.5 (market - rf) gives beta 1.5 and alpha zero."""
        rf = 0.001
        asset = rf + 1.5 * (_MARKET - rf)
        net = capm_regression(asset, _MARKET, risk_free_rate=rf)
        raw = capm_regression(asset, _MARKET)
        assert net.alpha == pytest.approx(0.0, abs=1e-12)
        assert net.beta == pytest.approx(1.5, rel=1e-9)
        # Without the risk-free rate the same series shows alpha rf * (1 - beta).
        assert raw.alpha == pytest.approx(rf * (1.0 - 1.5), rel=1e-9)

    def test_a_constant_series_matches_the_scalar(self) -> None:
        """A flat risk-free series is the same as passing its level."""
        asset, market = _simulate(n_obs=200)
        flat = pd.Series(0.0002, index=asset.index)
        scalar = capm_regression(asset, market, risk_free_rate=0.0002)
        series = capm_regression(asset, market, risk_free_rate=flat)
        assert dataclasses.asdict(series) == pytest.approx(dataclasses.asdict(scalar))

    def test_a_time_varying_series_is_subtracted_period_by_period(self) -> None:
        """Each period's excess return uses that period's rate."""
        asset, market = _simulate(n_obs=200)
        rng = np.random.default_rng(3)
        rate = pd.Series(rng.uniform(0.0, 0.0004, len(asset)), index=asset.index)
        intercept, slope, _, _ = _simple_fit(
            (asset - rate).to_numpy(), (market - rate).to_numpy()
        )
        result = capm_regression(asset, market, risk_free_rate=rate)
        assert result.alpha == pytest.approx(intercept, rel=1e-9)
        assert result.beta == pytest.approx(slope, rel=1e-9)

    def test_the_risk_free_rate_does_not_change_the_tracking_error(self) -> None:
        """Asset minus market is the same with or without excess returns."""
        asset, market = _simulate(n_obs=200)
        with_rate = capm_regression(asset, market, risk_free_rate=0.001)
        without = capm_regression(asset, market)
        assert with_rate.tracking_error == pytest.approx(without.tracking_error)
        assert with_rate.information_ratio == pytest.approx(without.information_ratio)


class TestDegenerateRatios:
    """Ratios with a zero denominator are reported as NaN rather than raised."""

    def test_a_tracking_error_of_zero_leaves_the_information_ratio_undefined(
        self,
    ) -> None:
        """An asset identical to the market has no active risk and no active return."""
        result = capm_regression(_MARKET, _MARKET)
        assert result.tracking_error == 0.0
        assert math.isnan(result.information_ratio)
        assert result.beta == pytest.approx(1.0, rel=1e-9)
        assert result.alpha == pytest.approx(0.0, abs=1e-12)
        assert result.r_squared == pytest.approx(1.0, abs=1e-12)


class TestCapmValidation:
    """Every bad input raises the documented error."""

    def test_misaligned_indexes_raise(self) -> None:
        """Series are never aligned behind the caller's back."""
        with pytest.raises(DataError, match="index"):
            capm_regression(_ASSET, _MARKET.shift(freq="D"))

    def test_nan_in_the_asset_raises(self) -> None:
        """Missing returns must be handled by the caller."""
        asset = _ASSET.copy()
        asset.iloc[2] = np.nan
        with pytest.raises(DataError, match="finite"):
            capm_regression(asset, _MARKET)

    def test_inf_in_the_market_raises(self) -> None:
        """Infinite returns are rejected like missing ones."""
        market = _MARKET.copy()
        market.iloc[2] = np.inf
        with pytest.raises(DataError, match="finite"):
            capm_regression(_ASSET, market)

    def test_a_risk_free_series_on_another_index_raises(self) -> None:
        """The rate series must share the dates of the returns."""
        rate = pd.Series(0.001, index=_DATES + pd.Timedelta(days=1))
        with pytest.raises(DataError, match="index"):
            capm_regression(_ASSET, _MARKET, risk_free_rate=rate)

    def test_a_risk_free_series_with_nan_raises(self) -> None:
        """A rate with gaps is not silently forward-filled."""
        rate = pd.Series(0.001, index=_DATES)
        rate.iloc[0] = np.nan
        with pytest.raises(DataError, match="finite"):
            capm_regression(_ASSET, _MARKET, risk_free_rate=rate)

    @pytest.mark.parametrize("rate", [float("nan"), float("inf"), "high"])
    def test_a_bad_scalar_risk_free_rate_raises(self, rate: object) -> None:
        """The scalar rate has to be a finite number."""
        with pytest.raises(ConfigurationError, match="risk_free_rate"):
            capm_regression(_ASSET, _MARKET, risk_free_rate=rate)  # type: ignore[arg-type]

    @pytest.mark.parametrize("periods", [0, -252, 12.5])
    def test_a_bad_periods_per_year_raises(self, periods: float) -> None:
        """Annualisation needs a positive whole number of periods."""
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            capm_regression(_ASSET, _MARKET, periods_per_year=periods)  # type: ignore[arg-type]

    def test_inputs_must_be_series(self) -> None:
        """Plain lists carry no index to align on."""
        with pytest.raises(DataError, match="pandas Series"):
            capm_regression([0.1, 0.2, 0.3, 0.1], _MARKET)  # type: ignore[arg-type]

    def test_non_numeric_series_raise(self) -> None:
        """A series of strings cannot be regressed."""
        words = pd.Series(["a"] * len(_ASSET), index=_DATES)
        with pytest.raises(DataError, match="numeric"):
            capm_regression(words, _MARKET)

    def test_fewer_than_three_observations_raise(self) -> None:
        """Two points and two parameters leave no degree of freedom."""
        with pytest.raises(DataError, match="observations"):
            capm_regression(_ASSET.iloc[:2], _MARKET.iloc[:2])

    def test_a_constant_asset_raises(self) -> None:
        """R-squared is undefined for a series that never moves."""
        with pytest.raises(DataError, match="constant"):
            capm_regression(pd.Series(0.01, index=_DATES), _MARKET)

    def test_the_result_type_is_exported(self) -> None:
        """The public result class is the one the function returns."""
        assert isinstance(capm_regression(_ASSET, _MARKET), CapmResult)


class TestRollingBeta:
    """Trailing-window covariance over variance."""

    def test_matches_the_covariance_over_variance_of_every_window(self) -> None:
        """Each full window equals the hand-computed ratio."""
        asset, market = _simulate(n_obs=40)
        window = 8
        result = rolling_beta(asset, market, window)
        # statistics.covariance and statistics.variance are both sample (n - 1).
        expected = [
            statistics.covariance(
                asset.iloc[end - window : end].tolist(),
                market.iloc[end - window : end].tolist(),
            )
            / statistics.variance(market.iloc[end - window : end].tolist())
            for end in range(window, len(asset) + 1)
        ]
        assert result.iloc[window - 1 :].to_numpy() == pytest.approx(expected, rel=1e-9)

    def test_it_is_nan_until_the_window_is_full(self) -> None:
        """No value is reported before window observations exist."""
        asset, market = _simulate(n_obs=20)
        result = rolling_beta(asset, market, 5)
        assert result.iloc[:4].isna().all()
        assert result.iloc[4:].notna().all()
        assert result.index.equals(asset.index)

    def test_a_scaled_copy_of_the_market_has_that_beta_everywhere(self) -> None:
        """beta of 2 * market on market is 2 in every window."""
        _, market = _simulate(n_obs=30)
        result = rolling_beta(2.0 * market, market, 10)
        assert result.iloc[9:].to_numpy() == pytest.approx(2.0, rel=1e-9)

    def test_a_window_as_long_as_the_sample_gives_the_full_sample_beta(self) -> None:
        """The last value equals the regression slope when the window is everything."""
        asset, market = _simulate(n_obs=50)
        full = capm_regression(asset, market)
        assert rolling_beta(asset, market, 50).iloc[-1] == pytest.approx(
            full.beta, rel=1e-9
        )

    def test_changing_future_data_does_not_change_earlier_values(self) -> None:
        """A value at time t uses only observations up to t."""
        asset, market = _simulate(n_obs=40)
        before = rolling_beta(asset, market, 8)
        altered_asset, altered_market = asset.copy(), market.copy()
        altered_asset.iloc[25:] += 0.05
        altered_market.iloc[25:] *= -3.0
        after = rolling_beta(altered_asset, altered_market, 8)
        pd.testing.assert_series_equal(before.iloc[:25], after.iloc[:25])
        assert not before.iloc[25:].equals(after.iloc[25:])

    def test_a_window_with_no_market_variance_is_nan(self) -> None:
        """A flat market has no beta to report."""
        dates = pd.date_range("2023-01-02", periods=8, freq="B")
        market = pd.Series([0.01] * 4 + [0.02, -0.01, 0.03, 0.0], index=dates)
        asset = pd.Series(np.arange(8.0) / 100.0, index=dates)
        result = rolling_beta(asset, market, 4)
        assert math.isnan(result.iloc[3])
        assert result.iloc[4:].notna().all()

    @pytest.mark.parametrize("window", [0, 1, -3])
    def test_a_window_below_two_raises(self, window: int) -> None:
        """A covariance needs at least two observations."""
        with pytest.raises(ConfigurationError, match="window"):
            rolling_beta(_ASSET, _MARKET, window)

    def test_a_non_integer_window_raises(self) -> None:
        """The window counts observations."""
        with pytest.raises(ConfigurationError, match="window"):
            rolling_beta(_ASSET, _MARKET, 2.5)  # type: ignore[arg-type]

    def test_a_window_longer_than_the_sample_raises(self) -> None:
        """An all-NaN result would hide the mistake."""
        with pytest.raises(DataError, match="window"):
            rolling_beta(_ASSET, _MARKET, 7)

    def test_misaligned_indexes_raise(self) -> None:
        """Series are never aligned behind the caller's back."""
        with pytest.raises(DataError, match="index"):
            rolling_beta(_ASSET, _MARKET.shift(freq="D"), 3)

    def test_missing_values_raise(self) -> None:
        """NaN is rejected rather than skipped."""
        asset = _ASSET.copy()
        asset.iloc[1] = np.nan
        with pytest.raises(DataError, match="finite"):
            rolling_beta(asset, _MARKET, 3)
