"""Tests for the multifactor and Fama-French-style regressions.

The golden numbers come from the normal equations solved by hand with numpy in this
file, and from series built as a known combination of the factors.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.regression import (
    FactorRegressionResult,
    FamaFrenchFactor,
    fama_french_regression,
    multifactor_regression,
    ols,
)

_YEAR = 252
_ALPHA = 0.0002
_LOADINGS = {"Mkt-RF": 0.9, "SMB": 0.4, "HML": -0.3}


def _factors(n_obs: int = 1500) -> pd.DataFrame:
    """Return seeded daily factor returns named as in the Fama-French library."""
    rng = np.random.default_rng(17)
    dates = pd.date_range("2018-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.010, n_obs),
            "SMB": rng.normal(0.0001, 0.005, n_obs),
            "HML": rng.normal(0.0000, 0.005, n_obs),
        },
        index=dates,
    )


def _asset(factors: pd.DataFrame, noise: float = 0.002) -> pd.Series:
    """Return 0.0002 + 0.9 MKT + 0.4 SMB - 0.3 HML plus seeded noise."""
    rng = np.random.default_rng(23)
    mix = _ALPHA + sum(_LOADINGS[name] * factors[name] for name in _LOADINGS)
    return mix + pd.Series(noise * rng.normal(size=len(factors)), index=factors.index)


def _normal_equations(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return coefficients, their standard errors and the residuals, by hand."""
    design = np.column_stack([np.ones(len(y)), x])
    gram = design.T @ design
    coefficients = np.linalg.solve(gram, design.T @ y)
    residuals = y - design @ coefficients
    sigma_squared = float(residuals @ residuals) / (len(y) - design.shape[1])
    std_errors = np.sqrt(np.diag(sigma_squared * np.linalg.inv(gram)))
    return coefficients, std_errors, residuals


class TestMultifactorRegression:
    """The excess return of an asset regressed on a set of factor returns."""

    def test_the_known_alpha_and_loadings_are_recovered(self) -> None:
        """Every estimate lies within four standard errors of its true value."""
        factors = _factors()
        asset = _asset(factors)
        _, std_errors, _ = _normal_equations(asset.to_numpy(), factors.to_numpy())
        result = multifactor_regression(asset, factors)
        assert abs(result.alpha - _ALPHA) < 4.0 * std_errors[0]
        for position, name in enumerate(factors.columns, start=1):
            assert abs(result.loadings[name] - _LOADINGS[name]) < (
                4.0 * std_errors[position]
            )

    def test_every_statistic_matches_the_normal_equations(self) -> None:
        """Coefficients, t-statistics, R-squared and residual risk agree."""
        factors = _factors()
        asset = _asset(factors)
        y, x = asset.to_numpy(), factors.to_numpy()
        coefficients, std_errors, residuals = _normal_equations(y, x)
        n_obs, n_params = len(y), x.shape[1] + 1
        r_squared = 1.0 - float(residuals @ residuals) / float(
            ((y - y.mean()) ** 2).sum()
        )
        sigma = math.sqrt(float(residuals @ residuals) / (n_obs - n_params))

        result = multifactor_regression(asset, factors)

        assert result.alpha == pytest.approx(coefficients[0], rel=1e-9)
        assert result.annualized_alpha == pytest.approx(coefficients[0] * _YEAR)
        assert result.alpha_t_statistic == pytest.approx(
            coefficients[0] / std_errors[0], rel=1e-9
        )
        assert list(result.loadings.index) == list(factors.columns)
        assert result.loadings.to_numpy() == pytest.approx(coefficients[1:], rel=1e-9)
        assert list(result.loading_t_statistics.index) == list(factors.columns)
        assert result.loading_t_statistics.to_numpy() == pytest.approx(
            coefficients[1:] / std_errors[1:], rel=1e-9
        )
        assert result.r_squared == pytest.approx(r_squared, rel=1e-9)
        assert result.adjusted_r_squared == pytest.approx(
            1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params), rel=1e-9
        )
        assert result.residual_volatility == pytest.approx(
            sigma * math.sqrt(_YEAR), rel=1e-9
        )

    def test_factor_names_are_free_for_a_generic_model(self) -> None:
        """The generic call accepts any factor labels."""
        factors = _factors().rename(columns={"Mkt-RF": "beta", "SMB": "size"})
        result = multifactor_regression(_asset(_factors()), factors)
        assert list(result.loadings.index) == ["beta", "size", "HML"]

    def test_the_asset_excess_return_is_measured_over_a_scalar_rate(self) -> None:
        """The rate is subtracted from the asset and not from the factors."""
        factors = _factors()
        asset = _asset(factors)
        rate = 0.0001
        coefficients, _, _ = _normal_equations(
            (asset - rate).to_numpy(), factors.to_numpy()
        )
        result = multifactor_regression(asset, factors, risk_free_rate=rate)
        assert result.alpha == pytest.approx(coefficients[0], rel=1e-9)
        assert result.loadings.to_numpy() == pytest.approx(coefficients[1:], rel=1e-9)

    def test_a_series_rate_is_subtracted_period_by_period(self) -> None:
        """An asset that earns the rate plus its factor premia has no alpha."""
        factors = _factors()
        rng = np.random.default_rng(4)
        rate = pd.Series(rng.uniform(0.0, 0.0004, len(factors)), index=factors.index)
        asset = rate + 0.9 * factors["Mkt-RF"] + 0.4 * factors["SMB"]
        result = multifactor_regression(asset, factors, risk_free_rate=rate)
        assert result.alpha == pytest.approx(0.0, abs=1e-12)
        assert result.loadings["HML"] == pytest.approx(0.0, abs=1e-9)
        assert result.loadings["Mkt-RF"] == pytest.approx(0.9, rel=1e-9)

    def test_newey_west_lags_change_the_t_statistics_only(self) -> None:
        """hac_lags is passed through to the regression."""
        factors = _factors()
        asset = _asset(factors)
        classical = multifactor_regression(asset, factors)
        robust = multifactor_regression(asset, factors, hac_lags=5)
        reference = ols(asset, factors, hac_lags=5)
        assert robust.loadings.to_numpy() == pytest.approx(
            classical.loadings.to_numpy(), rel=1e-12
        )
        assert robust.alpha_t_statistic == pytest.approx(
            float(reference.t_statistics["const"]), rel=1e-9
        )
        assert robust.loading_t_statistics.to_numpy() == pytest.approx(
            reference.t_statistics.iloc[1:].to_numpy(), rel=1e-9
        )
        assert robust.alpha_t_statistic != pytest.approx(classical.alpha_t_statistic)

    def test_a_noise_free_combination_has_full_explanatory_power(self) -> None:
        """Without noise the factors explain everything and no risk is left."""
        factors = _factors(200)
        result = multifactor_regression(_asset(factors, noise=0.0), factors)
        assert result.r_squared == pytest.approx(1.0, abs=1e-12)
        assert result.residual_volatility == pytest.approx(0.0, abs=1e-9)
        assert result.alpha == pytest.approx(_ALPHA, rel=1e-9)

    def test_the_result_is_immutable(self) -> None:
        """A shared result cannot be changed behind a caller's back."""
        factors = _factors(100)
        result = multifactor_regression(_asset(factors), factors)
        assert isinstance(result, FactorRegressionResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.alpha = 0.0  # type: ignore[misc]


class TestFamaFrenchRegression:
    """The same call with the standard factor names enforced."""

    def test_it_equals_the_generic_regression_for_valid_names(self) -> None:
        """The name check is the only difference."""
        factors = _factors()
        asset = _asset(factors)
        generic = multifactor_regression(asset, factors, risk_free_rate=0.0001)
        named = fama_french_regression(asset, factors, risk_free_rate=0.0001)
        pd.testing.assert_series_equal(named.loadings, generic.loadings)
        pd.testing.assert_series_equal(
            named.loading_t_statistics, generic.loading_t_statistics
        )
        assert named.alpha == generic.alpha
        assert named.r_squared == generic.r_squared
        assert named.residual_volatility == generic.residual_volatility

    def test_the_series_built_from_three_factors_recovers_them(self) -> None:
        """The loadings of 0.9, 0.4 and -0.3 are found."""
        factors = _factors()
        result = fama_french_regression(_asset(factors), factors)
        assert result.loadings.to_numpy() == pytest.approx([0.9, 0.4, -0.3], abs=0.05)
        assert result.alpha == pytest.approx(_ALPHA, abs=3e-4)

    def test_the_market_factor_alone_is_enough(self) -> None:
        """A one-factor Fama-French regression is the market model."""
        factors = _factors()[["Mkt-RF"]]
        result = fama_french_regression(_asset(_factors()), factors)
        assert list(result.loadings.index) == ["Mkt-RF"]

    def test_all_six_standard_names_are_accepted_in_any_order(self) -> None:
        """Five-factor plus momentum, with columns shuffled."""
        rng = np.random.default_rng(31)
        names = ["MOM", "CMA", "Mkt-RF", "RMW", "HML", "SMB"]
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        frame = pd.DataFrame(
            rng.normal(0.0, 0.01, (300, 6)), columns=names, index=dates
        )
        asset = pd.Series(
            0.5 * frame["Mkt-RF"] + rng.normal(0.0, 0.002, 300), index=dates
        )
        result = fama_french_regression(asset, frame)
        assert list(result.loadings.index) == names

    def test_the_enum_lists_the_standard_names(self) -> None:
        """The enum values are the library's column names."""
        assert [factor.value for factor in FamaFrenchFactor] == [
            "Mkt-RF",
            "SMB",
            "HML",
            "RMW",
            "CMA",
            "MOM",
        ]

    @pytest.mark.parametrize(
        ("keep", "rename"),
        [
            (["SMB", "HML"], {}),
            (["Mkt-RF", "SMB", "HML"], {"Mkt-RF": "Market"}),
            (["Mkt-RF", "SMB", "HML"], {"HML": "WML"}),
        ],
    )
    def test_bad_factor_names_raise(
        self, keep: list[str], rename: dict[str, str]
    ) -> None:
        """Names must be a subset of the standard six that includes Mkt-RF."""
        factors = pd.DataFrame(_factors(), columns=keep).rename(columns=rename)
        with pytest.raises(ConfigurationError, match="Mkt-RF"):
            fama_french_regression(_asset(_factors()), factors)

    def test_a_risk_free_column_points_the_caller_to_the_argument(self) -> None:
        """The RF column of the Fama-French files belongs in risk_free_rate."""
        factors = _factors()
        factors["RF"] = 0.0001
        with pytest.raises(ConfigurationError, match="risk_free_rate"):
            fama_french_regression(_asset(_factors()), factors)

    def test_names_are_case_sensitive(self) -> None:
        """Only the library's exact spelling is recognised."""
        factors = _factors().rename(columns={"Mkt-RF": "mkt-rf"})
        with pytest.raises(ConfigurationError, match="Mkt-RF"):
            fama_french_regression(_asset(_factors()), factors)


class TestFactorValidation:
    """Every bad input raises the documented error."""

    def test_factors_must_be_a_data_frame(self) -> None:
        """A single series is not a set of factors."""
        factors = _factors(50)
        with pytest.raises(DataError, match="DataFrame"):
            multifactor_regression(_asset(factors), factors["Mkt-RF"])  # type: ignore[arg-type]

    def test_the_asset_must_be_a_series(self) -> None:
        """An array carries no index to align on."""
        factors = _factors(50)
        with pytest.raises(DataError, match="pandas Series"):
            multifactor_regression(np.zeros(50), factors)  # type: ignore[arg-type]

    def test_misaligned_indexes_raise(self) -> None:
        """Asset and factors are never aligned behind the caller's back."""
        factors = _factors(50)
        with pytest.raises(DataError, match="index"):
            multifactor_regression(_asset(factors).shift(freq="D"), factors)

    def test_missing_factor_returns_raise(self) -> None:
        """NaN is rejected rather than skipped."""
        factors = _factors(50)
        factors.iloc[3, 1] = np.nan
        with pytest.raises(DataError, match="finite"):
            multifactor_regression(_asset(factors), factors)

    def test_missing_asset_returns_raise(self) -> None:
        """NaN is rejected rather than skipped."""
        factors = _factors(50)
        asset = _asset(factors)
        asset.iloc[3] = np.inf
        with pytest.raises(DataError, match="finite"):
            multifactor_regression(asset, factors)

    def test_duplicate_factor_names_raise(self) -> None:
        """Loadings are labelled by factor name."""
        factors = _factors(50)
        asset = _asset(factors)
        factors.columns = pd.Index(["Mkt-RF", "SMB", "SMB"])
        with pytest.raises(DataError, match="unique"):
            multifactor_regression(asset, factors)

    def test_a_frame_without_columns_raises(self) -> None:
        """At least one factor is needed."""
        factors = _factors(50)
        with pytest.raises(DataError, match="at least one"):
            multifactor_regression(_asset(factors), factors.iloc[:, :0])

    def test_collinear_factors_raise(self) -> None:
        """A factor that is a multiple of another cannot be told apart."""
        factors = _factors(50)
        factors["twin"] = 2.0 * factors["SMB"]
        with pytest.raises(DataError, match="collinear"):
            multifactor_regression(_asset(factors), factors)

    def test_too_few_observations_raise(self) -> None:
        """Three factors and a constant need at least five observations."""
        factors = _factors(4)
        with pytest.raises(DataError, match="observations"):
            multifactor_regression(_asset(factors), factors)

    def test_a_factor_called_const_clashes_with_the_intercept(self) -> None:
        """The intercept owns the name const."""
        original = _factors(50)
        factors = original.rename(columns={"SMB": "const"})
        with pytest.raises(DataError, match="const"):
            multifactor_regression(_asset(original), factors)

    @pytest.mark.parametrize("periods", [0, 1.5])
    def test_a_bad_periods_per_year_raises(self, periods: float) -> None:
        """Annualisation needs a positive whole number of periods."""
        factors = _factors(50)
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            multifactor_regression(
                _asset(factors),
                factors,
                periods_per_year=periods,  # type: ignore[arg-type]
            )

    def test_a_bad_risk_free_rate_raises(self) -> None:
        """The scalar rate has to be a finite number."""
        factors = _factors(50)
        with pytest.raises(ConfigurationError, match="risk_free_rate"):
            multifactor_regression(
                _asset(factors), factors, risk_free_rate=float("inf")
            )

    def test_a_bad_lag_count_raises(self) -> None:
        """hac_lags is validated like everywhere else."""
        factors = _factors(50)
        with pytest.raises(ConfigurationError, match="hac_lags"):
            multifactor_regression(_asset(factors), factors, hac_lags=-2)

    def test_the_fama_french_call_needs_a_data_frame_of_factors(self) -> None:
        """A bare series is reported as a type problem, not a naming one."""
        factors = _factors(50)
        with pytest.raises(DataError, match="DataFrame"):
            fama_french_regression(_asset(factors), factors["Mkt-RF"])  # type: ignore[arg-type]

    def test_the_fama_french_call_validates_the_data_too(self) -> None:
        """A valid name list does not excuse bad data."""
        factors = _factors(50)
        factors.iloc[0, 0] = np.nan
        with pytest.raises(DataError, match="finite"):
            fama_french_regression(_asset(factors), factors)
