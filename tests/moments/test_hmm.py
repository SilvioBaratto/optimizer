"""Tests for Gaussian HMM regime-conditional moment estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.moments import HMMConfig, HMMResult, blend_moments_by_regime, fit_hmm

TICKERS = ["AAPL", "MSFT", "GOOG"]
N_ASSETS = len(TICKERS)


def _make_two_regime_returns(
    n_obs: int = 300, seed: int = 42
) -> pd.DataFrame:
    """Synthetic return panel with two clearly separated regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_obs, freq="B")

    # Alternate between bull (high mean, low vol) and bear (low mean, high vol)
    regime = np.zeros(n_obs, dtype=int)
    regime[n_obs // 2 :] = 1

    means = [
        np.array([0.001, 0.0012, 0.0008]),   # bull
        np.array([-0.002, -0.0025, -0.0015]), # bear
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
        np.testing.assert_allclose(row_sums.to_numpy(), np.ones(len(row_sums)), atol=1e-8)

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
        np.testing.assert_allclose(row_sums.to_numpy(), np.ones(len(row_sums)), atol=1e-8)

    def test_nan_rows_dropped(self) -> None:
        rng = np.random.default_rng(5)
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        df = pd.DataFrame(rng.normal(0, 0.01, (200, N_ASSETS)), index=dates, columns=TICKERS)
        df.iloc[10] = np.nan
        df.iloc[50] = np.nan
        result = fit_hmm(df, HMMConfig(n_states=2, random_state=0))
        assert result.filtered_probs.shape[0] == 198

    def test_too_few_observations_raises(self) -> None:
        tiny = pd.DataFrame(
            np.random.default_rng(0).normal(0, 0.01, (2, N_ASSETS)),
            columns=TICKERS,
        )
        with pytest.raises(ValueError, match="observations"):
            fit_hmm(tiny, HMMConfig(n_states=2))

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
        rng = np.random.default_rng(3)
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
            log_likelihood=-100.0,
        )
        mu, cov = blend_moments_by_regime(result)
        np.testing.assert_allclose(mu.to_numpy(), means_arr[0], atol=1e-12)
        np.testing.assert_allclose(cov.to_numpy(), covs[0], atol=1e-12)
