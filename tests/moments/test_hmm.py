"""Tests for Gaussian HMM regime-conditional moment estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.moments import (
    HMMBlendedCovariance,
    HMMBlendedMu,
    HMMConfig,
    HMMResult,
    blend_moments_by_regime,
    fit_hmm,
    select_hmm_n_states,
)

TICKERS = ["AAPL", "MSFT", "GOOG"]
N_ASSETS = len(TICKERS)


def _make_two_regime_returns(n_obs: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic return panel with two clearly separated regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_obs, freq="B")

    # Alternate between bull (high mean, low vol) and bear (low mean, high vol)
    regime = np.zeros(n_obs, dtype=int)
    regime[n_obs // 2 :] = 1

    means = [
        np.array([0.001, 0.0012, 0.0008]),  # bull
        np.array([-0.002, -0.0025, -0.0015]),  # bear
    ]
    stds = [
        np.array([0.01, 0.012, 0.009]),  # bull: low vol
        np.array([0.03, 0.035, 0.028]),  # bear: high vol
    ]

    rows = []
    for t in range(n_obs):
        s = regime[t]
        rows.append(rng.normal(means[s], stds[s]))

    return pd.DataFrame(rows, index=dates, columns=TICKERS)


@pytest.fixture()
def synthetic_returns() -> pd.DataFrame:
    return _make_two_regime_returns()


@pytest.fixture()
def hmm_result(synthetic_returns: pd.DataFrame) -> HMMResult:
    config = HMMConfig(n_states=2, n_iter=200, random_state=0)
    return fit_hmm(synthetic_returns, config)


# ---------------------------------------------------------------------------
# HMMConfig
# ---------------------------------------------------------------------------


class TestHMMConfig:
    def test_defaults(self) -> None:
        cfg = HMMConfig()
        assert cfg.n_states == 2
        assert cfg.n_iter == 100
        assert cfg.tol == 1e-4
        assert cfg.covariance_type == "full"
        assert cfg.random_state is None

    def test_custom_values(self) -> None:
        cfg = HMMConfig(n_states=3, n_iter=50, random_state=7)
        assert cfg.n_states == 3
        assert cfg.random_state == 7

    def test_frozen(self) -> None:
        cfg = HMMConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.n_states = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# fit_hmm — structural correctness
# ---------------------------------------------------------------------------


class TestFitHMM:
    def test_returns_hmm_result(self, hmm_result: HMMResult) -> None:
        assert isinstance(hmm_result, HMMResult)

    def test_transition_matrix_shape(self, hmm_result: HMMResult) -> None:
        assert hmm_result.transition_matrix.shape == (2, 2)

    def test_transition_matrix_row_stochastic(self, hmm_result: HMMResult) -> None:
        row_sums = hmm_result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-8)

    def test_regime_means_shape(self, hmm_result: HMMResult) -> None:
        assert hmm_result.regime_means.shape == (2, N_ASSETS)

    def test_regime_means_columns_match_tickers(self, hmm_result: HMMResult) -> None:
        assert list(hmm_result.regime_means.columns) == TICKERS

    def test_regime_covariances_shape(self, hmm_result: HMMResult) -> None:
        assert hmm_result.regime_covariances.shape == (2, N_ASSETS, N_ASSETS)

    def test_filtered_probs_shape(
        self, synthetic_returns: pd.DataFrame, hmm_result: HMMResult
    ) -> None:
        assert hmm_result.filtered_probs.shape == (len(synthetic_returns), 2)

    def test_filtered_probs_rows_sum_to_one(self, hmm_result: HMMResult) -> None:
        row_sums = hmm_result.filtered_probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.to_numpy(), np.ones(len(row_sums)), atol=1e-8
        )

    def test_filtered_probs_non_negative(self, hmm_result: HMMResult) -> None:
        assert (hmm_result.filtered_probs >= 0).all().all()

    def test_filtered_probs_index_matches_input(
        self, synthetic_returns: pd.DataFrame, hmm_result: HMMResult
    ) -> None:
        pd.testing.assert_index_equal(
            hmm_result.filtered_probs.index, synthetic_returns.index
        )

    def test_log_likelihood_is_float(self, hmm_result: HMMResult) -> None:
        assert isinstance(hmm_result.log_likelihood, float)

    def test_log_likelihood_is_finite(self, hmm_result: HMMResult) -> None:
        assert np.isfinite(hmm_result.log_likelihood)

    def test_default_config_used_when_none(
        self, synthetic_returns: pd.DataFrame
    ) -> None:
        result = fit_hmm(synthetic_returns)
        assert isinstance(result, HMMResult)

    def test_three_state_model(self, synthetic_returns: pd.DataFrame) -> None:
        config = HMMConfig(n_states=3, random_state=1)
        result = fit_hmm(synthetic_returns, config)
        assert result.transition_matrix.shape == (3, 3)
        assert result.filtered_probs.shape == (len(synthetic_returns), 3)
        row_sums = result.filtered_probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.to_numpy(), np.ones(len(row_sums)), atol=1e-8
        )

    def test_nan_rows_dropped(self) -> None:
        rng = np.random.default_rng(5)
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        df = pd.DataFrame(
            rng.normal(0, 0.01, (200, N_ASSETS)), index=dates, columns=TICKERS
        )
        df.iloc[10] = np.nan
        df.iloc[50] = np.nan
        result = fit_hmm(df, HMMConfig(n_states=2, random_state=0))
        assert result.filtered_probs.shape[0] == 198

    def test_too_few_observations_raises(self) -> None:
        tiny = pd.DataFrame(
            np.random.default_rng(0).normal(0, 0.01, (2, N_ASSETS)),
            columns=TICKERS,
        )
        with pytest.raises(DataError, match="observations"):
            fit_hmm(tiny, HMMConfig(n_states=2))

    def test_smoothed_probs_shape(
        self, synthetic_returns: pd.DataFrame, hmm_result: HMMResult
    ) -> None:
        assert hmm_result.smoothed_probs.shape == (len(synthetic_returns), 2)

    def test_smoothed_probs_rows_sum_to_one(self, hmm_result: HMMResult) -> None:
        row_sums = hmm_result.smoothed_probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.to_numpy(), np.ones(len(row_sums)), atol=1e-8
        )

    def test_filtered_probs_differ_from_smoothed(self) -> None:
        """Forward-filtered probabilities should differ from smoothed posteriors.

        Smoothed posteriors condition on the full sequence (including future
        observations), so they should generally differ from the forward-only
        filtered probabilities, especially at early timesteps where future
        data provides the most additional information.
        """
        returns = _make_two_regime_returns(n_obs=300, seed=7)
        config = HMMConfig(n_states=2, n_iter=200, random_state=0)
        result = fit_hmm(returns, config)

        # Filtered and smoothed should have the same shape and properties
        assert result.filtered_probs.shape == result.smoothed_probs.shape
        # Both should sum to 1 across states
        np.testing.assert_allclose(
            result.filtered_probs.sum(axis=1).to_numpy(),
            np.ones(len(returns)),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.smoothed_probs.sum(axis=1).to_numpy(),
            np.ones(len(returns)),
            atol=1e-8,
        )
        # They should not be identical (smoothed uses backward pass too)
        diff = (result.filtered_probs - result.smoothed_probs).abs().max().max()
        assert diff > 1e-6, (
            "Filtered and smoothed probs should differ for two-regime data"
        )

    def test_two_regime_means_differ(self, hmm_result: HMMResult) -> None:
        """Fitted regime means should be distinguishable (not identical)."""
        m0 = hmm_result.regime_means.iloc[0].to_numpy()
        m1 = hmm_result.regime_means.iloc[1].to_numpy()
        assert not np.allclose(m0, m1, atol=1e-4)

    def test_regime_covariances_positive_definite(self, hmm_result: HMMResult) -> None:
        for s in range(2):
            eigvals = np.linalg.eigvalsh(hmm_result.regime_covariances[s])
            assert (eigvals > -1e-10).all()


# ---------------------------------------------------------------------------
# blend_moments_by_regime
# ---------------------------------------------------------------------------


class TestBlendMomentsByRegime:
    def test_returns_series_and_dataframe(self, hmm_result: HMMResult) -> None:
        mu, cov = blend_moments_by_regime(hmm_result)
        assert isinstance(mu, pd.Series)
        assert isinstance(cov, pd.DataFrame)

    def test_mu_index_matches_tickers(self, hmm_result: HMMResult) -> None:
        mu, _ = blend_moments_by_regime(hmm_result)
        assert list(mu.index) == TICKERS

    def test_cov_shape(self, hmm_result: HMMResult) -> None:
        _, cov = blend_moments_by_regime(hmm_result)
        assert cov.shape == (N_ASSETS, N_ASSETS)

    def test_cov_index_and_columns_match_tickers(self, hmm_result: HMMResult) -> None:
        _, cov = blend_moments_by_regime(hmm_result)
        assert list(cov.index) == TICKERS
        assert list(cov.columns) == TICKERS

    def test_mu_between_regime_extremes(self, hmm_result: HMMResult) -> None:
        """Blended mu must lie between the per-regime min and max means."""
        mu, _ = blend_moments_by_regime(hmm_result)
        mu_min = hmm_result.regime_means.min(axis=0)
        mu_max = hmm_result.regime_means.max(axis=0)
        assert (mu >= mu_min - 1e-12).all()
        assert (mu <= mu_max + 1e-12).all()

    def test_pure_state_returns_regime_moments(self) -> None:
        """If last filtered prob is 100% in state 0, blended = regime 0 moments."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")

        # Build a mock HMMResult with controlled filtered_probs
        means_arr = np.array([[0.001, 0.002, 0.003], [-0.001, -0.002, -0.003]])
        covs = np.stack([np.eye(N_ASSETS) * 0.0001, np.eye(N_ASSETS) * 0.0002])
        probs = pd.DataFrame(
            np.column_stack([np.ones(n), np.zeros(n)]),
            index=dates,
            columns=[0, 1],
        )
        result = HMMResult(
            transition_matrix=np.array([[0.9, 0.1], [0.1, 0.9]]),
            regime_means=pd.DataFrame(means_arr, index=[0, 1], columns=TICKERS),
            regime_covariances=covs,
            filtered_probs=probs,
            smoothed_probs=probs,
            log_likelihood=-100.0,
        )
        mu, cov = blend_moments_by_regime(result)
        np.testing.assert_allclose(mu.to_numpy(), means_arr[0], atol=1e-12)
        np.testing.assert_allclose(cov.to_numpy(), covs[0], atol=1e-12)


# ---------------------------------------------------------------------------
# HMMBlendedMu — skfolio BaseMu estimator
# ---------------------------------------------------------------------------


class TestHMMBlendedMu:
    def test_fit_returns_self(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        result = est.fit(synthetic_returns)
        assert result is est

    def test_mu_attribute_set_after_fit(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert hasattr(est, "mu_")

    def test_mu_shape(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert est.mu_.shape == (N_ASSETS,)

    def test_mu_is_numpy_array(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert isinstance(est.mu_, np.ndarray)

    def test_mu_values_finite(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert np.all(np.isfinite(est.mu_))

    def test_hmm_result_attribute_set(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert hasattr(est, "hmm_result_")
        assert isinstance(est.hmm_result_, HMMResult)

    def test_n_features_in_set(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert est.n_features_in_ == N_ASSETS

    def test_feature_names_preserved(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert list(est.feature_names_in_) == TICKERS

    def test_default_config_used_when_none(
        self, synthetic_returns: pd.DataFrame
    ) -> None:
        est = HMMBlendedMu()
        est.fit(synthetic_returns)
        assert est.mu_.shape == (N_ASSETS,)

    def test_mu_within_regime_range(self, synthetic_returns: pd.DataFrame) -> None:
        """Blended mu must fall within the envelope of per-regime means."""
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        mu_min = est.hmm_result_.regime_means.min(axis=0).to_numpy()
        mu_max = est.hmm_result_.regime_means.max(axis=0).to_numpy()
        assert np.all(est.mu_ >= mu_min - 1e-12)
        assert np.all(est.mu_ <= mu_max + 1e-12)

    def test_accepts_numpy_array_input(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns.to_numpy())
        assert est.mu_.shape == (N_ASSETS,)


# ---------------------------------------------------------------------------
# HMMBlendedCovariance — skfolio BaseCovariance estimator
# ---------------------------------------------------------------------------


class TestHMMBlendedCovariance:
    def test_fit_returns_self(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        result = est.fit(synthetic_returns)
        assert result is est

    def test_covariance_attribute_set_after_fit(
        self, synthetic_returns: pd.DataFrame
    ) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert hasattr(est, "covariance_")

    def test_covariance_shape(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert est.covariance_.shape == (N_ASSETS, N_ASSETS)

    def test_covariance_is_numpy_array(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert isinstance(est.covariance_, np.ndarray)

    def test_covariance_is_symmetric(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        np.testing.assert_allclose(est.covariance_, est.covariance_.T, atol=1e-12)

    def test_covariance_is_psd(self, synthetic_returns: pd.DataFrame) -> None:
        """All eigenvalues must be non-negative (positive semi-definite)."""
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        eigvals = np.linalg.eigvalsh(est.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_covariance_values_finite(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert np.all(np.isfinite(est.covariance_))

    def test_hmm_result_attribute_set(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert hasattr(est, "hmm_result_")
        assert isinstance(est.hmm_result_, HMMResult)

    def test_n_features_in_set(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert est.n_features_in_ == N_ASSETS

    def test_feature_names_preserved(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        assert list(est.feature_names_in_) == TICKERS

    def test_default_config_used_when_none(
        self, synthetic_returns: pd.DataFrame
    ) -> None:
        est = HMMBlendedCovariance()
        est.fit(synthetic_returns)
        assert est.covariance_.shape == (N_ASSETS, N_ASSETS)

    def test_covariance_larger_than_within_regime(
        self, synthetic_returns: pd.DataFrame
    ) -> None:
        """Full law of total variance includes mean-dispersion term, so blended
        diagonal variance >= weighted average of within-regime variances."""
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns)
        weights = est.hmm_result_.filtered_probs.iloc[-1].to_numpy()
        weighted_diag = sum(
            weights[s] * np.diag(est.hmm_result_.regime_covariances[s])
            for s in range(len(weights))
        )
        blended_diag = np.diag(est.covariance_)
        assert np.all(blended_diag >= weighted_diag - 1e-12)

    def test_accepts_numpy_array_input(self, synthetic_returns: pd.DataFrame) -> None:
        est = HMMBlendedCovariance(hmm_config=HMMConfig(n_states=2, random_state=0))
        est.fit(synthetic_returns.to_numpy())
        assert est.covariance_.shape == (N_ASSETS, N_ASSETS)


# ---------------------------------------------------------------------------
# select_hmm_n_states — AIC/BIC model selection (issue #66)
# ---------------------------------------------------------------------------


class TestSelectHMMNStates:
    def test_returns_int(self, synthetic_returns: pd.DataFrame) -> None:
        result = select_hmm_n_states(
            synthetic_returns,
            candidate_n_states=(2, 3),
            hmm_config=HMMConfig(random_state=0),
        )
        assert isinstance(result, int)

    def test_result_in_candidates(self, synthetic_returns: pd.DataFrame) -> None:
        candidates = (2, 3, 4)
        result = select_hmm_n_states(
            synthetic_returns,
            candidate_n_states=candidates,
            hmm_config=HMMConfig(random_state=0),
        )
        assert result in candidates

    def test_bic_selects_two_for_two_regime_data(
        self, synthetic_returns: pd.DataFrame
    ) -> None:
        """BIC should prefer 2 states for clearly two-regime data."""
        result = select_hmm_n_states(
            synthetic_returns,
            candidate_n_states=(2, 3, 4),
            criterion="bic",
            hmm_config=HMMConfig(random_state=0),
        )
        assert result == 2

    def test_aic_works(self, synthetic_returns: pd.DataFrame) -> None:
        result = select_hmm_n_states(
            synthetic_returns,
            candidate_n_states=(2, 3),
            criterion="aic",
            hmm_config=HMMConfig(random_state=0),
        )
        assert result in (2, 3)

    def test_invalid_criterion_raises(self, synthetic_returns: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="criterion"):
            select_hmm_n_states(
                synthetic_returns,
                criterion="hqic",
            )

    def test_default_config_works(self, synthetic_returns: pd.DataFrame) -> None:
        result = select_hmm_n_states(synthetic_returns, candidate_n_states=(2,))
        assert result == 2
