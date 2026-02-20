"""Tests for log-normal moment scaling (apply_lognormal_correction / scale_moments_to_horizon)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    # Daily variances: (0.20/sqrt(252))^2, etc.
    diag = np.array([0.20**2 / 252, 0.25**2 / 252, 0.30**2 / 252])
    return pd.DataFrame(np.diag(diag), index=TICKERS, columns=TICKERS)


# ---------------------------------------------------------------------------
# apply_lognormal_correction
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

    def test_cov_diagonal_scales_linearly(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        horizon = 63
        _, cov_t = apply_lognormal_correction(daily_mu, daily_cov, horizon=horizon)
        expected_diag = np.diag(daily_cov.to_numpy()) * horizon
        np.testing.assert_allclose(np.diag(cov_t.to_numpy()), expected_diag, rtol=1e-9)

    def test_horizon_1_mu_close_to_daily(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """At horizon=1 the correction exp(mu + 0.5*sigma^2) - 1 ≈ mu for small values."""
        mu_t, _ = apply_lognormal_correction(daily_mu, daily_cov, horizon=1)
        sigma2 = np.diag(daily_cov.to_numpy())
        # First-order Taylor: exp(mu + 0.5*s2) - 1 ≈ mu + 0.5*s2
        expected = np.exp(daily_mu.to_numpy() + 0.5 * sigma2) - 1.0
        np.testing.assert_allclose(mu_t.to_numpy(), expected, rtol=1e-9)

    def test_horizon_1_cov_unchanged(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        _, cov_t = apply_lognormal_correction(daily_mu, daily_cov, horizon=1)
        np.testing.assert_allclose(cov_t.to_numpy(), daily_cov.to_numpy(), rtol=1e-9)

    def test_annual_correction_is_material(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """At 252 days the Jensen correction is economically meaningful (> 1% extra)."""
        mu_t, _ = apply_lognormal_correction(daily_mu, daily_cov, horizon=252)
        naive_scale = daily_mu * 252
        delta = mu_t - naive_scale
        assert (delta > 0.001).all()

    def test_off_diagonal_cov_scales_linearly(self) -> None:
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((N, N))
        cov_mat = raw @ raw.T / N  # positive definite
        mu = pd.Series(np.zeros(N), index=TICKERS)
        cov = pd.DataFrame(cov_mat, index=TICKERS, columns=TICKERS)
        horizon = 21
        _, cov_t = apply_lognormal_correction(mu, cov, horizon=horizon)
        np.testing.assert_allclose(cov_t.to_numpy(), cov_mat * horizon, rtol=1e-9)

    def test_zero_mu_correction_driven_by_variance(
        self, daily_cov: pd.DataFrame
    ) -> None:
        """With mu=0, corrected mean = exp(0.5 * sigma^2 * T) - 1 > 0."""
        mu_zero = pd.Series(np.zeros(N), index=TICKERS)
        mu_t, _ = apply_lognormal_correction(mu_zero, daily_cov, horizon=252)
        assert (mu_t > 0).all()

    def test_invalid_horizon_zero_raises(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            apply_lognormal_correction(daily_mu, daily_cov, horizon=0)

    def test_invalid_horizon_negative_raises(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            apply_lognormal_correction(daily_mu, daily_cov, horizon=-5)

    def test_mismatched_index_raises(self, daily_cov: pd.DataFrame) -> None:
        mu_bad = pd.Series([0.001, 0.002, 0.003], index=["X", "Y", "Z"])
        with pytest.raises(ValueError, match="same ticker index"):
            apply_lognormal_correction(mu_bad, daily_cov, horizon=21)

    def test_monthly_horizon_21(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        mu_t, cov_t = apply_lognormal_correction(daily_mu, daily_cov, horizon=21)
        sigma2 = np.diag(daily_cov.to_numpy())
        expected_mu = np.exp(daily_mu.to_numpy() * 21 + 0.5 * sigma2 * 21) - 1.0
        np.testing.assert_allclose(mu_t.to_numpy(), expected_mu, rtol=1e-9)
        np.testing.assert_allclose(
            np.diag(cov_t.to_numpy()),
            sigma2 * 21,
            rtol=1e-9,
        )


# ---------------------------------------------------------------------------
# scale_moments_to_horizon
# ---------------------------------------------------------------------------


class TestScaleMomentsToHorizon:
    def test_returns_same_as_apply_lognormal_correction(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        mu_a, cov_a = apply_lognormal_correction(daily_mu, daily_cov, horizon=63)
        mu_s, cov_s = scale_moments_to_horizon(daily_mu, daily_cov, daily_horizon=63)
        pd.testing.assert_series_equal(mu_a, mu_s)
        pd.testing.assert_frame_equal(cov_a, cov_s)

    def test_non_square_cov_raises(self, daily_mu: pd.Series) -> None:
        bad_cov = pd.DataFrame(np.ones((2, 3)))
        with pytest.raises(ValueError, match="square matrix"):
            scale_moments_to_horizon(daily_mu, bad_cov, daily_horizon=21)

    def test_mismatched_mu_cov_length_raises(self) -> None:
        mu = pd.Series([0.001, 0.002], index=["A", "B"])
        cov = pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"])
        with pytest.raises(ValueError):
            scale_moments_to_horizon(mu, cov, daily_horizon=21)

    def test_negative_variance_raises(self, daily_mu: pd.Series) -> None:
        bad_cov = pd.DataFrame(
            -np.eye(N) * 0.001, index=TICKERS, columns=TICKERS
        )
        with pytest.raises(ValueError, match="negative values"):
            scale_moments_to_horizon(daily_mu, bad_cov, daily_horizon=21)

    def test_horizon_must_be_positive(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            scale_moments_to_horizon(daily_mu, daily_cov, daily_horizon=0)

    def test_output_mu_geq_naive_scaling(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        """Corrected mu is always >= naive mu * T (Jensen's inequality)."""
        horizon = 252
        mu_t, _ = scale_moments_to_horizon(daily_mu, daily_cov, daily_horizon=horizon)
        assert (mu_t >= daily_mu * horizon - 1e-12).all()

    def test_output_cov_diagonal_exact(
        self, daily_mu: pd.Series, daily_cov: pd.DataFrame
    ) -> None:
        horizon = 42
        _, cov_t = scale_moments_to_horizon(daily_mu, daily_cov, daily_horizon=horizon)
        np.testing.assert_allclose(
            np.diag(cov_t.to_numpy()),
            np.diag(daily_cov.to_numpy()) * horizon,
            rtol=1e-9,
        )
