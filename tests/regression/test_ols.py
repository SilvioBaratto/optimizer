"""Tests for the ordinary least squares core.

The golden numbers come from closed forms and normal equations written out in this
file with plain numpy, from the published R output for the first of Anscombe's
quartet, and from scipy's Student t survival function.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy import stats

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.regression import OlsResult, ols

# Anscombe (1973), "Graphs in Statistical Analysis", first data set.
_ANSCOMBE_X = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
_ANSCOMBE_Y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]


def _normal_equations(
    y: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return coefficients, residuals and the classical covariance, solved by hand."""
    gram = design.T @ design
    coefficients = np.linalg.solve(gram, design.T @ y)
    residuals = y - design @ coefficients
    n_obs, n_params = design.shape
    sigma_squared = float(residuals @ residuals) / (n_obs - n_params)
    return coefficients, residuals, sigma_squared * np.linalg.inv(gram)


def _newey_west_reference(
    design: npt.NDArray[np.float64], residuals: npt.NDArray[np.float64], lags: int
) -> npt.NDArray[np.float64]:
    """Return the Newey-West covariance with explicit loops over time and lags."""
    n_obs, n_params = design.shape
    meat = np.zeros((n_params, n_params))
    for t in range(n_obs):
        meat += residuals[t] ** 2 * np.outer(design[t], design[t])
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        for t in range(lag, n_obs):
            cross = (
                residuals[t] * residuals[t - lag] * np.outer(design[t], design[t - lag])
            )
            meat += weight * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    return bread @ meat @ bread


def _simple_data(n_obs: int = 60) -> tuple[npt.NDArray[np.float64], pd.Series]:
    """Return a seeded regressor and a noisy response."""
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, n_obs)
    y = 1.5 + 0.8 * x + rng.normal(0.0, 0.5, n_obs)
    return x, pd.Series(y)


def _ar1(rng: np.random.Generator, n_obs: int, rho: float) -> npt.NDArray[np.float64]:
    """Return an AR(1) path with unit-variance shocks."""
    shocks = rng.normal(size=n_obs)
    path = np.empty(n_obs)
    path[0] = shocks[0]
    for t in range(1, n_obs):
        path[t] = rho * path[t - 1] + shocks[t]
    return path


def _autocorrelated_data(
    n_obs: int = 300,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return two persistent regressors and a response with persistent errors."""
    rng = np.random.default_rng(5)
    x = np.column_stack([_ar1(rng, n_obs, 0.7), _ar1(rng, n_obs, 0.7)])
    y = 0.3 + x @ np.array([1.0, -0.5]) + _ar1(rng, n_obs, 0.7)
    return x, y


class TestSimpleRegression:
    """One regressor and a constant against the textbook closed form."""

    def test_matches_the_closed_form(self) -> None:
        """Slope, intercept, errors, t-statistics, p-values and R-squared agree."""
        x, y_series = _simple_data()
        y = y_series.to_numpy()
        n_obs = x.size
        # Textbook simple regression: beta = cov(x, y) / var(x) and
        # alpha = ybar - beta * xbar.
        beta = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
        alpha = float(y.mean() - beta * x.mean())
        residuals = y - alpha - beta * x
        sxx = float(((x - x.mean()) ** 2).sum())
        sigma_squared = float(residuals @ residuals) / (n_obs - 2)
        se_beta = math.sqrt(sigma_squared / sxx)
        se_alpha = math.sqrt(sigma_squared * (1.0 / n_obs + x.mean() ** 2 / sxx))
        r_squared = 1.0 - float(residuals @ residuals) / float(
            ((y - y.mean()) ** 2).sum()
        )

        result = ols(y_series, x)

        assert result.coefficients.to_numpy() == pytest.approx([alpha, beta], rel=1e-9)
        assert result.std_errors.to_numpy() == pytest.approx(
            [se_alpha, se_beta], rel=1e-9
        )
        assert result.t_statistics.to_numpy() == pytest.approx(
            [alpha / se_alpha, beta / se_beta], rel=1e-9
        )
        assert result.r_squared == pytest.approx(r_squared, rel=1e-9)
        assert result.adjusted_r_squared == pytest.approx(
            1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - 2), rel=1e-9
        )
        assert result.residuals.to_numpy() == pytest.approx(residuals, abs=1e-10)
        assert result.n_observations == n_obs
        assert result.degrees_of_freedom == n_obs - 2

    def test_p_values_come_from_the_student_t_distribution(self) -> None:
        """Two-sided p-values use the residual degrees of freedom."""
        x, y = _simple_data()
        result = ols(y, x)
        # scipy's t survival function evaluated in the test, not by the module.
        expected = 2.0 * stats.t.sf(np.abs(result.t_statistics.to_numpy()), x.size - 2)
        assert result.p_values.to_numpy() == pytest.approx(expected, rel=1e-9)

    def test_matches_published_output_for_anscombe_one(self) -> None:
        """The estimates equal the R summary of lm(y1 ~ x1) printed in textbooks."""
        result = ols(pd.Series(_ANSCOMBE_Y), np.array(_ANSCOMBE_X))
        # Published R output: estimates 3.0001 and 0.5001, standard errors 1.1247
        # and 0.1179, t values 2.667 and 4.241, p-values 0.02573 and 0.00217,
        # multiple R-squared 0.6665 and adjusted R-squared 0.6295 on 9 degrees
        # of freedom.
        assert result.coefficients.to_numpy() == pytest.approx(
            [3.0001, 0.5001], abs=5e-5
        )
        assert result.std_errors.to_numpy() == pytest.approx([1.1247, 0.1179], abs=5e-5)
        assert result.t_statistics.to_numpy() == pytest.approx([2.667, 4.241], abs=5e-4)
        assert result.p_values.to_numpy() == pytest.approx([0.02573, 0.00217], abs=5e-6)
        assert result.r_squared == pytest.approx(0.6665, abs=5e-5)
        assert result.adjusted_r_squared == pytest.approx(0.6295, abs=5e-5)
        assert result.degrees_of_freedom == 9

    def test_a_noise_free_line_is_recovered_exactly(self) -> None:
        """Without noise the fit is perfect and the R-squared is one."""
        x = np.arange(10.0)
        result = ols(2.0 + 3.0 * x, x)
        assert result.coefficients.to_numpy() == pytest.approx([2.0, 3.0], abs=1e-10)
        assert result.r_squared == pytest.approx(1.0, abs=1e-12)
        assert np.abs(result.residuals.to_numpy()).max() < 1e-10
        assert result.std_errors.to_numpy().max() < 1e-10

    def test_a_perfect_fit_gives_huge_t_statistics_without_warnings(self) -> None:
        """A zero standard error must not leak a numpy warning to the caller."""
        x = np.arange(10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = ols(2.0 + 3.0 * x, x)
        assert result.t_statistics.min() > 1e6
        assert result.p_values.max() < 1e-10

    def test_the_result_is_immutable(self) -> None:
        """A shared result cannot be changed behind a caller's back."""
        x, y = _simple_data()
        result = ols(y, x)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.r_squared = 0.0  # type: ignore[misc]


class TestMultipleRegression:
    """Several regressors against the normal equations solved by hand."""

    def test_matches_the_normal_equations(self) -> None:
        """Coefficients, errors, t-statistics, p-values and R-squared agree."""
        rng = np.random.default_rng(21)
        n_obs = 250
        base = rng.normal(size=(n_obs, 3))
        x = base + 0.5 * base[:, [0]]
        y = 0.5 + x @ np.array([1.0, -2.0, 0.5]) + rng.normal(0.0, 1.0, n_obs)
        design = np.column_stack([np.ones(n_obs), x])
        coefficients, residuals, covariance = _normal_equations(y, design)
        std_errors = np.sqrt(np.diag(covariance))
        r_squared = 1.0 - float(residuals @ residuals) / float(
            ((y - y.mean()) ** 2).sum()
        )

        result = ols(pd.Series(y), pd.DataFrame(x, columns=["a", "b", "c"]))

        assert list(result.coefficients.index) == ["const", "a", "b", "c"]
        assert result.coefficients.to_numpy() == pytest.approx(coefficients, rel=1e-9)
        assert result.std_errors.to_numpy() == pytest.approx(std_errors, rel=1e-9)
        assert result.t_statistics.to_numpy() == pytest.approx(
            coefficients / std_errors, rel=1e-9
        )
        expected_p = 2.0 * stats.t.sf(np.abs(coefficients / std_errors), n_obs - 4)
        assert result.p_values.to_numpy() == pytest.approx(expected_p, rel=1e-9)
        assert result.r_squared == pytest.approx(r_squared, rel=1e-9)
        assert result.adjusted_r_squared == pytest.approx(
            1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - 4), rel=1e-9
        )
        assert result.degrees_of_freedom == n_obs - 4

    def test_a_regression_through_the_origin_uses_the_uncentred_r_squared(
        self,
    ) -> None:
        """Without an intercept R-squared compares with the raw sum of squares."""
        x, y_series = _simple_data()
        y = y_series.to_numpy()
        beta = float(x @ y / (x @ x))
        residuals = y - beta * x
        result = ols(y_series, x, add_constant=False)
        assert list(result.coefficients.index) == ["x"]
        assert result.coefficients.iloc[0] == pytest.approx(beta, rel=1e-9)
        r_squared = 1.0 - float(residuals @ residuals) / float(y @ y)
        assert result.r_squared == pytest.approx(r_squared, rel=1e-9)
        # Uncentred total sum of squares has n degrees of freedom, residuals n - 1.
        assert result.adjusted_r_squared == pytest.approx(
            1.0 - (1.0 - r_squared) * x.size / (x.size - 1), rel=1e-9
        )
        assert result.degrees_of_freedom == x.size - 1

    def test_a_supplied_constant_column_is_treated_as_the_intercept(self) -> None:
        """R-squared is centred when the regressors already hold a constant."""
        x, y = _simple_data()
        supplied = pd.DataFrame({"one": np.ones(x.size), "x": x})
        automatic = ols(y, x)
        manual = ols(y, supplied, add_constant=False)
        assert manual.r_squared == pytest.approx(automatic.r_squared, rel=1e-9)
        assert manual.adjusted_r_squared == pytest.approx(
            automatic.adjusted_r_squared, rel=1e-9
        )


class TestLabels:
    """Names and indexes of the returned series."""

    def test_a_named_series_keeps_its_name(self) -> None:
        """The constant is called const and the regressor keeps its own name."""
        x, y = _simple_data()
        result = ols(y, pd.Series(x, name="mkt"))
        assert list(result.coefficients.index) == ["const", "mkt"]
        assert list(result.std_errors.index) == ["const", "mkt"]
        assert list(result.t_statistics.index) == ["const", "mkt"]
        assert list(result.p_values.index) == ["const", "mkt"]

    def test_an_unnamed_series_or_one_dimensional_array_is_called_x(self) -> None:
        """A regressor without a name falls back to x."""
        x, y = _simple_data()
        assert list(ols(y, pd.Series(x)).coefficients.index) == ["const", "x"]
        assert list(ols(y, x).coefficients.index) == ["const", "x"]

    def test_two_dimensional_arrays_number_their_columns(self) -> None:
        """Columns of a bare array are called x1, x2, and so on."""
        x, y = _autocorrelated_data()
        assert list(ols(y, x).coefficients.index) == ["const", "x1", "x2"]

    def test_residuals_carry_the_index_of_the_response(self) -> None:
        """Residuals line up with the dates of the response."""
        x, y = _simple_data()
        dates = pd.date_range("2024-01-01", periods=x.size, freq="B")
        result = ols(
            pd.Series(y.to_numpy(), index=dates), pd.Series(x, index=dates, name="m")
        )
        assert result.residuals.index.equals(dates)

    def test_the_constant_can_be_left_out(self) -> None:
        """With add_constant=False no constant column is added."""
        x, y = _autocorrelated_data()
        result = ols(y, pd.DataFrame(x, columns=["a", "b"]), add_constant=False)
        assert list(result.coefficients.index) == ["a", "b"]


class TestNeweyWest:
    """Heteroskedasticity and autocorrelation consistent standard errors."""

    @pytest.mark.parametrize("lags", [0, 1, 4, 10])
    def test_matches_an_independent_loop_implementation(self, lags: int) -> None:
        """The sandwich equals the Bartlett-weighted long-run variance in loops."""
        x, y = _autocorrelated_data()
        design = np.column_stack([np.ones(len(y)), x])
        coefficients, residuals, _ = _normal_equations(y, design)
        covariance = _newey_west_reference(design, residuals, lags)
        std_errors = np.sqrt(np.diag(covariance))

        result = ols(y, x, hac_lags=lags)

        assert result.std_errors.to_numpy() == pytest.approx(std_errors, rel=1e-9)
        assert result.t_statistics.to_numpy() == pytest.approx(
            coefficients / std_errors, rel=1e-9
        )
        expected_p = 2.0 * stats.t.sf(
            np.abs(coefficients / std_errors), len(y) - design.shape[1]
        )
        assert result.p_values.to_numpy() == pytest.approx(expected_p, rel=1e-9)

    def test_the_point_estimates_do_not_depend_on_the_error_model(self) -> None:
        """Only the standard errors change when lags are requested."""
        x, y = _autocorrelated_data()
        classical = ols(y, x)
        robust = ols(y, x, hac_lags=5)
        assert robust.coefficients.to_numpy() == pytest.approx(
            classical.coefficients.to_numpy(), rel=1e-12
        )
        assert robust.r_squared == pytest.approx(classical.r_squared, rel=1e-12)
        assert robust.degrees_of_freedom == classical.degrees_of_freedom

    def test_no_lags_means_the_classical_errors(self) -> None:
        """hac_lags=None is the textbook s-squared (X'X)^-1."""
        x, y = _autocorrelated_data()
        design = np.column_stack([np.ones(len(y)), x])
        _, _, covariance = _normal_equations(y, design)
        result = ols(y, x, hac_lags=None)
        assert result.std_errors.to_numpy() == pytest.approx(
            np.sqrt(np.diag(covariance)), rel=1e-9
        )

    def test_persistent_errors_widen_the_standard_errors(self) -> None:
        """Positive autocorrelation makes classical errors too small."""
        x, y = _autocorrelated_data()
        classical = ols(y, x)
        robust = ols(y, x, hac_lags=10)
        assert (robust.std_errors.iloc[1:] > classical.std_errors.iloc[1:]).all()

    def test_negative_lags_are_rejected(self) -> None:
        """A negative lag count has no meaning."""
        x, y = _autocorrelated_data()
        with pytest.raises(ConfigurationError, match="hac_lags"):
            ols(y, x, hac_lags=-1)

    @pytest.mark.parametrize("lags", [1.5, True])
    def test_non_integer_lags_are_rejected(self, lags: object) -> None:
        """The lag count must be a whole number."""
        x, y = _autocorrelated_data()
        with pytest.raises(ConfigurationError, match="hac_lags"):
            ols(y, x, hac_lags=lags)  # type: ignore[arg-type]

    def test_lags_that_exhaust_the_sample_are_rejected(self) -> None:
        """Lags must be shorter than the sample."""
        x, y = _autocorrelated_data(n_obs=20)
        with pytest.raises(DataError, match="hac_lags"):
            ols(y, x, hac_lags=20)


class TestValidation:
    """Bad data raises DataError instead of being dropped or patched."""

    def test_nan_in_the_response_raises(self) -> None:
        """A missing response is never dropped silently."""
        x, y = _simple_data()
        y.iloc[3] = np.nan
        with pytest.raises(DataError, match="finite"):
            ols(y, x)

    def test_nan_in_the_regressors_raises(self) -> None:
        """A missing regressor is never dropped silently."""
        x, y = _simple_data()
        x[7] = np.nan
        with pytest.raises(DataError, match="finite"):
            ols(y, x)

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_infinite_values_raise(self, bad: float) -> None:
        """Infinite values are rejected like missing ones."""
        x, y = _simple_data()
        y.iloc[0] = bad
        with pytest.raises(DataError, match="finite"):
            ols(y, x)

    def test_a_duplicated_regressor_is_rank_deficient(self) -> None:
        """Perfectly collinear regressors cannot be separated."""
        x, y = _simple_data()
        frame = pd.DataFrame({"a": x, "b": 2.0 * x})
        with pytest.raises(DataError, match="collinear"):
            ols(y, frame)

    def test_a_constant_regressor_next_to_the_intercept_is_rank_deficient(
        self,
    ) -> None:
        """A constant column is collinear with the added intercept."""
        _, y = _simple_data()
        with pytest.raises(DataError, match="collinear"):
            ols(y, np.full(len(y), 4.0))

    def test_a_collinear_design_without_a_constant_raises(self) -> None:
        """Rank is checked whether or not a constant is added."""
        x, y = _simple_data()
        frame = pd.DataFrame({"a": x, "b": -x})
        with pytest.raises(DataError, match="collinear"):
            ols(y, frame, add_constant=False)

    def test_fewer_observations_than_parameters_plus_one_raise(self) -> None:
        """No residual degree of freedom leaves nothing to estimate errors from."""
        with pytest.raises(DataError, match="observations"):
            ols(np.array([1.0, 2.0]), np.array([0.5, 1.5]))

    def test_one_residual_degree_of_freedom_is_enough(self) -> None:
        """Three points and two parameters is the smallest regression allowed."""
        result = ols(np.array([1.0, 2.5, 2.0]), np.array([0.0, 1.0, 2.0]))
        assert result.degrees_of_freedom == 1

    def test_mismatched_lengths_raise(self) -> None:
        """The response and the regressors need the same number of rows."""
        x, y = _simple_data()
        with pytest.raises(DataError, match="rows"):
            ols(y, x[:-1])

    def test_mismatched_indexes_raise(self) -> None:
        """Two pandas inputs are never aligned behind the caller's back."""
        x, y = _simple_data()
        shifted = pd.Series(x, index=y.index + 1, name="m")
        with pytest.raises(DataError, match="index"):
            ols(y, shifted)

    def test_a_two_dimensional_response_raises(self) -> None:
        """Only one response at a time."""
        x, _ = _simple_data()
        with pytest.raises(DataError, match="one-dimensional"):
            ols(np.column_stack([x, x]), x)

    def test_a_three_dimensional_regressor_raises(self) -> None:
        """Regressors are a vector or a matrix."""
        x, y = _simple_data()
        with pytest.raises(DataError, match="dimensional"):
            ols(y, x.reshape(-1, 1, 1))

    def test_a_constant_response_raises(self) -> None:
        """R-squared is undefined when the response never moves."""
        x, _ = _simple_data()
        with pytest.raises(DataError, match="constant"):
            ols(np.full(x.size, 0.01), x)

    def test_non_numeric_data_raises(self) -> None:
        """Strings cannot be regressed."""
        x, _ = _simple_data()
        with pytest.raises(DataError, match="numeric"):
            ols(["a"] * x.size, x)

    def test_duplicate_regressor_names_raise(self) -> None:
        """Coefficients are labelled by name, so names must be unique."""
        x, y = _autocorrelated_data()
        frame = pd.DataFrame(x, columns=["a", "a"])
        with pytest.raises(DataError, match="unique"):
            ols(y, frame)

    def test_a_regressor_called_const_clashes_with_the_intercept(self) -> None:
        """The added constant owns the name const."""
        x, y = _simple_data()
        with pytest.raises(DataError, match="const"):
            ols(y, pd.Series(x, name="const"))

    def test_a_regressor_matrix_without_columns_raises(self) -> None:
        """At least one regressor is needed, even with an intercept."""
        _, y = _simple_data()
        with pytest.raises(DataError, match="at least one column"):
            ols(y, np.empty((len(y), 0)))

    def test_empty_inputs_raise(self) -> None:
        """No observations is reported as too few observations."""
        with pytest.raises(DataError, match="observations"):
            ols(np.array([]), np.array([]))

    def test_scalar_inputs_raise(self) -> None:
        """A single number is neither a response nor a regressor vector."""
        with pytest.raises(DataError, match="dimensional"):
            ols(np.array([1.0, 2.0, 3.0]), 2.0)

    def test_the_result_type_is_exported(self) -> None:
        """The public result class is the one the function returns."""
        x, y = _simple_data()
        assert isinstance(ols(y, x), OlsResult)
