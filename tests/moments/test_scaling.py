"""Tests for log-normal moment scaling.

Covers apply_lognormal_correction and scale_moments_to_horizon for both
the "exact" (full log-normal) and "linear" (delta-method) covariance methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.moments import apply_lognormal_correction, scale_moments_to_horizon

TICKERS = ["AAPL", "MSFT", "GOOG"]
N = len(TICKERS)


@pytest.fixture()
def daily_mu() -> pd.Series:
    """Typical daily log-return means (~10% p.a. = 0.10/252)."""
    return pd.Series([0.10 / 252, 0.08 / 252, 0.12 / 252], index=TICKERS)


@pytest.fixture()
def daily_cov() -> pd.DataFrame:
    """Simple diagonal covariance (annualised vol ~ 20-30%)."""
    diag = np.array([0.20**2 / 252, 0.25**2 / 252, 0.30**2 / 252])
    return pd.DataFrame(np.diag(diag), index=TICKERS, columns=TICKERS)


# ---------------------------------------------------------------------------
# apply_lognormal_correction — shared / method-independent behaviour
# ---------------------------------------------------------------------------


class TestApplyLognormalCorrection:
    def test_returns_tuple_of_series_and_dataframe(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        mu_t, cov_t = apply_lognormal_correction(daily_mu, daily_cov, horizon=21)
        assert isinstance(mu_t, pd.Series)
        assert isinstance(cov_t, pd.DataFrame)

    def test_index_preserved(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        mu_t, cov_t = apply_lognormal_correction(daily_mu, daily_cov, horizon=21)
        assert list(mu_t.index) == TICKERS
        assert list(cov_t.index) == TICKERS
        assert list(cov_t.columns) == TICKERS

    def test_mu_corrected_is_greater_than_input(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """Log-normal correction is always positive: E[R_T] >= mu * T."""
        mu_t, _ = apply_lognormal_correction(daily_mu, daily_cov, horizon=252)
        assert (mu_t > daily_mu * 252).all()

    def test_annual_correction_is_material(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """At 252 days the Jensen correction is economically meaningful (> 1% extra)."""
        mu_t, _ = apply_lognormal_correction(daily_mu, daily_cov, horizon=252)
        naive_scale = daily_mu * 252
        delta = mu_t - naive_scale
        assert (delta > 0.001).all()

    def test_zero_mu_correction_driven_by_variance(
        self, daily_cov: pd.DataFrame
    ) -> None:
        """With mu=0, corrected mean = exp(0.5 * sigma^2 * T) - 1 > 0."""
        mu_zero = pd.Series(np.zeros(N), index=TICKERS)
        mu_t, _ = apply_lognormal_correction(mu_zero, daily_cov, horizon=252)
        assert (mu_t > 0).all()

    def test_horizon_1_mu_close_to_daily(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """At horizon=1, exp(mu + 0.5*sigma^2) - 1 matches exact formula."""
        mu_t, _ = apply_lognormal_correction(daily_mu, daily_cov, horizon=1)
        sigma2 = np.diag(daily_cov.to_numpy())
        expected = np.exp(daily_mu.to_numpy() + 0.5 * sigma2) - 1.0
        np.testing.assert_allclose(mu_t.to_numpy(), expected, rtol=1e-9)

    def test_invalid_horizon_zero_raises(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        with pytest.raises(DataError, match="positive integer"):
            apply_lognormal_correction(daily_mu, daily_cov, horizon=0)

    def test_invalid_horizon_negative_raises(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        with pytest.raises(DataError, match="positive integer"):
            apply_lognormal_correction(daily_mu, daily_cov, horizon=-5)

    def test_mismatched_index_raises(self, daily_cov: pd.DataFrame) -> None:
        mu_bad = pd.Series([0.001, 0.002, 0.003], index=["X", "Y", "Z"])
        with pytest.raises(DataError, match="same ticker index"):
            apply_lognormal_correction(mu_bad, daily_cov, horizon=21)

    def test_correction_exceeds_naive_compound_when_sigma_positive(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """E[R_T] = exp(mu*T + 0.5*sigma^2*T) - 1 > exp(mu*T) - 1 for sigma > 0."""
        horizon = 252
        mu_t, _ = apply_lognormal_correction(daily_mu, daily_cov, horizon=horizon)
        naive_compound = np.exp(daily_mu.to_numpy() * horizon) - 1.0
        assert np.all(mu_t.to_numpy() > naive_compound)

    def test_zero_variance_matches_naive_compound(
        self, daily_mu: pd.Series
    ) -> None:
        """With zero covariance, correction = exp(mu*T) - 1 exactly."""
        horizon = 63
        zero_cov = pd.DataFrame(
            np.zeros((len(daily_mu), len(daily_mu))),
            index=daily_mu.index,
            columns=daily_mu.index,
        )
        mu_t, _ = apply_lognormal_correction(daily_mu, zero_cov, horizon=horizon)
        expected = np.exp(daily_mu.to_numpy() * horizon) - 1.0
        np.testing.assert_allclose(mu_t.to_numpy(), expected, rtol=1e-12)

    def test_higher_sigma_produces_larger_correction(self) -> None:
        """Higher vol should produce a larger corrected expected return."""
        mu_val = 0.10 / 252
        horizon = 252

        # Low vol
        mu_low = pd.Series([mu_val], index=["X"])
        cov_low = pd.DataFrame(
            [[0.15**2 / 252]], index=["X"], columns=["X"]
        )
        mu_t_low, _ = apply_lognormal_correction(mu_low, cov_low, horizon=horizon)

        # High vol
        mu_high = pd.Series([mu_val], index=["X"])
        cov_high = pd.DataFrame(
            [[0.30**2 / 252]], index=["X"], columns=["X"]
        )
        mu_t_high, _ = apply_lognormal_correction(mu_high, cov_high, horizon=horizon)

        assert float(mu_t_high.iloc[0]) > float(mu_t_low.iloc[0])

    def test_invalid_method_raises(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="method must be one of"):
            apply_lognormal_correction(daily_mu, daily_cov, horizon=21, method="bad")


# ---------------------------------------------------------------------------
# "linear" method — delta-method approximation (Sigma * T)
# ---------------------------------------------------------------------------


class TestLinearMethod:
    def test_cov_diagonal_scales_linearly(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        horizon = 63
        _, cov_t = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=horizon, method="linear"
        )
        expected_diag = np.diag(daily_cov.to_numpy()) * horizon
        np.testing.assert_allclose(np.diag(cov_t.to_numpy()), expected_diag, rtol=1e-9)

    def test_off_diagonal_cov_scales_linearly(self) -> None:
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((N, N))
        cov_mat = raw @ raw.T / N
        mu = pd.Series(np.zeros(N), index=TICKERS)
        cov = pd.DataFrame(cov_mat, index=TICKERS, columns=TICKERS)
        horizon = 21
        _, cov_t = apply_lognormal_correction(mu, cov, horizon=horizon, method="linear")
        np.testing.assert_allclose(cov_t.to_numpy(), cov_mat * horizon, rtol=1e-9)

    def test_horizon_1_cov_unchanged(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        _, cov_t = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=1, method="linear"
        )
        np.testing.assert_allclose(cov_t.to_numpy(), daily_cov.to_numpy(), rtol=1e-9)

    def test_monthly_horizon_21(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        mu_t, cov_t = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=21, method="linear"
        )
        sigma2 = np.diag(daily_cov.to_numpy())
        expected_mu = np.exp(daily_mu.to_numpy() * 21 + 0.5 * sigma2 * 21) - 1.0
        np.testing.assert_allclose(mu_t.to_numpy(), expected_mu, rtol=1e-9)
        np.testing.assert_allclose(np.diag(cov_t.to_numpy()), sigma2 * 21, rtol=1e-9)


# ---------------------------------------------------------------------------
# "exact" method — full log-normal formula
# ---------------------------------------------------------------------------


class TestExactMethod:
    def test_default_method_is_exact(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """Omitting method= uses "exact" by default."""
        _, cov_default = apply_lognormal_correction(daily_mu, daily_cov, horizon=63)
        _, cov_exact = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=63, method="exact"
        )
        np.testing.assert_array_equal(cov_default.to_numpy(), cov_exact.to_numpy())

    def test_diagonal_exact_formula(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """Verify Var[R_T] = exp(2mu*T + sigma^2*T) * (exp(sigma^2*T) - 1)."""
        horizon = 63
        mu_arr = daily_mu.to_numpy()
        sigma2 = np.diag(daily_cov.to_numpy())
        expected_diag = np.exp(2 * mu_arr * horizon + sigma2 * horizon) * (
            np.exp(sigma2 * horizon) - 1.0
        )
        _, cov_t = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=horizon, method="exact"
        )
        np.testing.assert_allclose(np.diag(cov_t.to_numpy()), expected_diag, rtol=1e-9)

    def test_off_diagonal_exact_formula(self) -> None:
        """Verify Cov[R_T^i, R_T^j] formula for off-diagonal elements."""
        rng = np.random.default_rng(3)
        raw = rng.standard_normal((N, N))
        cov_arr = raw @ raw.T / N
        mu_arr = np.array([0.0004, 0.0003, 0.0005])
        mu = pd.Series(mu_arr, index=TICKERS)
        cov = pd.DataFrame(cov_arr, index=TICKERS, columns=TICKERS)
        horizon = 21
        sigma2 = np.diag(cov_arr)

        _, cov_t = apply_lognormal_correction(mu, cov, horizon=horizon, method="exact")

        i, j = 0, 1
        expected_01 = np.exp(
            (mu_arr[i] + mu_arr[j]) * horizon + 0.5 * (sigma2[i] + sigma2[j]) * horizon
        ) * (np.exp(cov_arr[i, j] * horizon) - 1.0)
        assert cov_t.iloc[i, j] == pytest.approx(expected_01, rel=1e-9)

    def test_index_preserved(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        _, cov_t = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=21, method="exact"
        )
        assert list(cov_t.index) == TICKERS
        assert list(cov_t.columns) == TICKERS

    # --- Acceptance criterion 1 ---

    def test_exact_variance_geq_linear(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """Acceptance criterion: exact diagonal >= linear diagonal at all horizons."""
        for horizon in (1, 21, 63, 252):
            _, cov_lin = apply_lognormal_correction(
                daily_mu, daily_cov, horizon=horizon, method="linear"
            )
            _, cov_ex = apply_lognormal_correction(
                daily_mu, daily_cov, horizon=horizon, method="exact"
            )
            diag_lin = np.diag(cov_lin.to_numpy())
            diag_ex = np.diag(cov_ex.to_numpy())
            assert np.all(diag_ex >= diag_lin - 1e-12), (
                f"horizon={horizon}: exact diagonal not >= linear"
            )

    # --- Acceptance criterion 2 ---

    def test_horizon_1_exact_agrees_with_linear_to_1e6(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """Acceptance criterion: at horizon=1 both methods agree to within 1e-6."""
        _, cov_lin = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=1, method="linear"
        )
        _, cov_ex = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=1, method="exact"
        )
        np.testing.assert_allclose(cov_ex.to_numpy(), cov_lin.to_numpy(), atol=1e-6)

    # --- Acceptance criterion 3 ---

    def test_horizon_252_exact_differs_materially_from_linear(self) -> None:
        """Acceptance criterion: at 252 days exact != linear for sigma=0.20."""
        sigma2_daily = 0.20**2 / 252
        mu_daily = 0.05 / 252

        mu = pd.Series([mu_daily], index=["X"])
        cov = pd.DataFrame([[sigma2_daily]], index=["X"], columns=["X"])

        _, cov_lin = apply_lognormal_correction(mu, cov, horizon=252, method="linear")
        _, cov_ex = apply_lognormal_correction(mu, cov, horizon=252, method="exact")

        var_lin = float(cov_lin.iloc[0, 0])
        var_ex = float(cov_ex.iloc[0, 0])

        assert var_ex > var_lin
        rel_diff = (var_ex - var_lin) / var_lin
        assert rel_diff > 0.001, (
            f"Expected measurable divergence but got rel_diff={rel_diff:.6f}"
        )


# ---------------------------------------------------------------------------
# scale_moments_to_horizon
# ---------------------------------------------------------------------------


class TestScaleMomentsToHorizon:
    def test_delegates_to_apply_linear(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        mu_a, cov_a = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=63, method="linear"
        )
        mu_s, cov_s = scale_moments_to_horizon(
            daily_mu, daily_cov, daily_horizon=63, method="linear"
        )
        pd.testing.assert_series_equal(mu_a, mu_s)
        pd.testing.assert_frame_equal(cov_a, cov_s)

    def test_delegates_to_apply_exact(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        mu_a, cov_a = apply_lognormal_correction(
            daily_mu, daily_cov, horizon=63, method="exact"
        )
        mu_s, cov_s = scale_moments_to_horizon(
            daily_mu, daily_cov, daily_horizon=63, method="exact"
        )
        pd.testing.assert_series_equal(mu_a, mu_s)
        pd.testing.assert_frame_equal(cov_a, cov_s)

    def test_default_method_is_exact(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        _, cov_default = scale_moments_to_horizon(daily_mu, daily_cov, daily_horizon=63)
        _, cov_exact = scale_moments_to_horizon(
            daily_mu, daily_cov, daily_horizon=63, method="exact"
        )
        pd.testing.assert_frame_equal(cov_default, cov_exact)

    def test_non_square_cov_raises(self, daily_mu: pd.Series) -> None:
        bad_cov = pd.DataFrame(np.ones((2, 3)))
        with pytest.raises(DataError, match="square matrix"):
            scale_moments_to_horizon(daily_mu, bad_cov, daily_horizon=21)

    def test_mismatched_mu_cov_length_raises(self) -> None:
        mu = pd.Series([0.001, 0.002], index=["A", "B"])
        cov = pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"])
        with pytest.raises(DataError):
            scale_moments_to_horizon(mu, cov, daily_horizon=21)

    def test_negative_variance_raises(self, daily_mu: pd.Series) -> None:
        bad_cov = pd.DataFrame(-np.eye(N) * 0.001, index=TICKERS, columns=TICKERS)
        with pytest.raises(DataError, match="negative values"):
            scale_moments_to_horizon(daily_mu, bad_cov, daily_horizon=21)

    def test_horizon_must_be_positive(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        with pytest.raises(DataError, match="positive integer"):
            scale_moments_to_horizon(daily_mu, daily_cov, daily_horizon=0)

    def test_output_mu_geq_naive_scaling(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """Corrected mu is always >= naive mu * T (Jensen's inequality)."""
        horizon = 252
        mu_t, _ = scale_moments_to_horizon(daily_mu, daily_cov, daily_horizon=horizon)
        assert (mu_t >= daily_mu * horizon - 1e-12).all()

    def test_output_cov_diagonal_linear(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        horizon = 42
        _, cov_t = scale_moments_to_horizon(
            daily_mu, daily_cov, daily_horizon=horizon, method="linear"
        )
        np.testing.assert_allclose(
            np.diag(cov_t.to_numpy()),
            np.diag(daily_cov.to_numpy()) * horizon,
            rtol=1e-9,
        )
