"""Tests for empirical omega calibration from forecast error track record."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.views import calibrate_omega_from_track_record

N_VIEWS = 3
VIEW_NAMES = ["view_A", "view_B", "view_C"]


def _make_histories(
    n_obs: int = 20,
    bias: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic (view_history, return_history) pair with controllable error."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    realized = pd.DataFrame(
        rng.normal(0.001, 0.01, (n_obs, N_VIEWS)),
        index=dates,
        columns=VIEW_NAMES,
    )
    forecast = realized + bias + rng.normal(0.0, 0.005, (n_obs, N_VIEWS))
    return forecast, realized


class TestCalibrateOmegaFromTrackRecord:
    def test_returns_square_matrix(self) -> None:
        view_h, ret_h = _make_histories()
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        assert omega.shape == (N_VIEWS, N_VIEWS)

    def test_output_is_numpy_array(self) -> None:
        view_h, ret_h = _make_histories()
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        assert isinstance(omega, np.ndarray)

    def test_output_dtype_float64(self) -> None:
        view_h, ret_h = _make_histories()
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        assert omega.dtype == np.float64

    def test_diagonal_is_non_negative(self) -> None:
        view_h, ret_h = _make_histories()
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        assert np.all(np.diag(omega) >= 0.0)

    def test_off_diagonal_is_zero(self) -> None:
        """omega must be a diagonal matrix."""
        view_h, ret_h = _make_histories()
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        off_diag = omega - np.diag(np.diag(omega))
        np.testing.assert_array_equal(off_diag, 0.0)

    def test_matrix_is_psd(self) -> None:
        """Diagonal variance matrix is always positive semi-definite."""
        view_h, ret_h = _make_histories()
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        eigvals = np.linalg.eigvalsh(omega)
        assert np.all(eigvals >= -1e-14)

    def test_zero_forecast_error_gives_zero_diagonal(self) -> None:
        """Perfect forecasts (zero error) → Ω diagonal ≈ 0 (full confidence)."""
        view_h, ret_h = _make_histories()
        # Make forecasts identical to realised returns
        perfect_forecast = ret_h.copy()
        omega = calibrate_omega_from_track_record(perfect_forecast, ret_h)
        np.testing.assert_allclose(np.diag(omega), 0.0, atol=1e-14)

    def test_large_forecast_error_gives_large_diagonal(self) -> None:
        """Large forecast noise → Ω diagonal >> 0 (low confidence)."""
        rng = np.random.default_rng(42)
        n_obs = 30
        dates = pd.date_range("2020-01-01", periods=n_obs, freq="B")
        realized = pd.DataFrame(
            rng.normal(0.001, 0.01, (n_obs, N_VIEWS)),
            index=dates,
            columns=VIEW_NAMES,
        )
        # Add large random noise (std=0.10) so Var(error) ≈ 0.01
        noisy_forecast = realized + rng.normal(0.0, 0.10, (n_obs, N_VIEWS))
        omega = calibrate_omega_from_track_record(
            pd.DataFrame(noisy_forecast, index=dates, columns=VIEW_NAMES),
            realized,
        )
        assert np.all(np.diag(omega) > 1e-4)

    def test_diagonal_equals_sample_variance_of_errors(self) -> None:
        """Ω_{kk} should equal ddof=1 variance of (Q_k - r_k)."""
        view_h, ret_h = _make_histories(seed=7)
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        errors = view_h - ret_h
        expected_var = errors.var(axis=0, ddof=1).to_numpy()
        np.testing.assert_allclose(np.diag(omega), expected_var, rtol=1e-12)

    def test_more_error_variance_produces_larger_omega(self) -> None:
        """Higher forecast noise → larger diagonal Ω."""
        view_h_low, ret_h = _make_histories(seed=42)
        rng = np.random.default_rng(42)
        dates = ret_h.index
        # High-noise forecasts: add large random error
        view_h_high = ret_h + pd.DataFrame(
            rng.normal(0, 0.1, ret_h.shape), index=dates, columns=VIEW_NAMES
        )
        omega_low = calibrate_omega_from_track_record(view_h_low, ret_h)
        omega_high = calibrate_omega_from_track_record(view_h_high, ret_h)
        assert np.all(np.diag(omega_high) > np.diag(omega_low))

    def test_raises_when_shapes_differ(self) -> None:
        view_h, ret_h = _make_histories()
        bad_ret = ret_h.iloc[:, :2]
        with pytest.raises(ValueError, match="same shape"):
            calibrate_omega_from_track_record(view_h, bad_ret)

    def test_raises_when_columns_differ(self) -> None:
        view_h, ret_h = _make_histories()
        bad_ret = ret_h.copy()
        bad_ret.columns = ["x", "y", "z"]
        with pytest.raises(ValueError, match="same column names"):
            calibrate_omega_from_track_record(view_h, bad_ret)

    def test_raises_when_fewer_than_5_observations(self) -> None:
        view_h, ret_h = _make_histories(n_obs=4)
        with pytest.raises(ValueError, match="at least 5"):
            calibrate_omega_from_track_record(view_h, ret_h)

    def test_raises_when_nan_rows_reduce_below_5(self) -> None:
        view_h, ret_h = _make_histories(n_obs=6)
        view_h.iloc[0] = np.nan
        view_h.iloc[1] = np.nan
        with pytest.raises(ValueError, match="at least 5"):
            calibrate_omega_from_track_record(view_h, ret_h)

    def test_nan_rows_are_dropped_before_computation(self) -> None:
        """NaN rows in either DataFrame are dropped; result uses clean subset."""
        view_h, ret_h = _make_histories(n_obs=20)
        view_h_with_nan = view_h.copy()
        view_h_with_nan.iloc[5] = np.nan  # introduce one NaN row
        omega_full = calibrate_omega_from_track_record(view_h, ret_h)
        omega_dropped = calibrate_omega_from_track_record(view_h_with_nan, ret_h)
        # After dropping the NaN row, compute expected variance on 19 rows
        errors_clean = (view_h_with_nan - ret_h).dropna()
        expected = errors_clean.var(axis=0, ddof=1).to_numpy()
        np.testing.assert_allclose(np.diag(omega_dropped), expected, rtol=1e-12)
        # And the results should differ from the full-data omega
        assert not np.allclose(np.diag(omega_full), np.diag(omega_dropped))

    def test_exactly_5_observations_succeeds(self) -> None:
        """Boundary: exactly 5 observations should not raise."""
        view_h, ret_h = _make_histories(n_obs=5)
        omega = calibrate_omega_from_track_record(view_h, ret_h)
        assert omega.shape == (N_VIEWS, N_VIEWS)
