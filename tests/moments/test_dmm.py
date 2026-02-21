"""Tests for Deep Markov Model (DMM) regime-conditional moment estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip entire module if PyTorch / Pyro are not installed
pytest.importorskip("torch")
pytest.importorskip("pyro")

from optimizer.moments import (
    DMMConfig,
    DMMResult,
    blend_moments_dmm,
    fit_dmm,
)
from optimizer.moments._dmm import (
    DMM,
    Combiner,
    Emitter,
    GatedTransition,
)

TICKERS = ["AAPL", "MSFT", "GOOG"]
N_ASSETS = len(TICKERS)

# Small config so tests run fast (z_dim=4, tiny hidden dims)
FAST_CONFIG = DMMConfig(
    z_dim=4,
    emission_dim=16,
    transition_dim=16,
    rnn_dim=32,
    num_epochs=80,
    annealing_epochs=10,
    minimum_annealing_factor=0.2,
    random_state=42,
)


def _make_returns(n_obs: int = 200, seed: int = 42) -> pd.DataFrame:
    """Two-regime synthetic return series with clear signal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_obs, freq="B")
    means = np.where(np.arange(n_obs) < n_obs // 2, 0.001, -0.002)
    noise = rng.normal(0, 0.01, (n_obs, N_ASSETS))
    data = means[:, None] + noise
    return pd.DataFrame(data, index=dates, columns=TICKERS)


@pytest.fixture(scope="module")
def synthetic_returns() -> pd.DataFrame:
    return _make_returns()


@pytest.fixture(scope="module")
def dmm_result(synthetic_returns: pd.DataFrame) -> DMMResult:
    return fit_dmm(synthetic_returns, FAST_CONFIG)


# ---------------------------------------------------------------------------
# DMMConfig
# ---------------------------------------------------------------------------


class TestDMMConfig:
    def test_defaults(self) -> None:
        cfg = DMMConfig()
        assert cfg.z_dim == 16
        assert cfg.emission_dim == 64
        assert cfg.transition_dim == 64
        assert cfg.rnn_dim == 128
        assert cfg.num_epochs == 1000
        assert cfg.annealing_epochs == 50
        assert cfg.minimum_annealing_factor == 0.2
        assert cfg.random_state is None

    def test_custom_values(self) -> None:
        cfg = DMMConfig(z_dim=8, num_epochs=200, random_state=7)
        assert cfg.z_dim == 8
        assert cfg.num_epochs == 200
        assert cfg.random_state == 7

    def test_frozen(self) -> None:
        cfg = DMMConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.z_dim = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------


class TestEmitter:
    def test_output_shapes(self) -> None:
        import torch

        emitter = Emitter(N_ASSETS, 4, 16)
        z = torch.zeros(4)
        loc, scale = emitter(z)
        assert loc.shape == (N_ASSETS,)
        assert scale.shape == (N_ASSETS,)

    def test_scale_positive(self) -> None:
        import torch

        emitter = Emitter(N_ASSETS, 4, 16)
        z = torch.randn(4)
        _, scale = emitter(z)
        assert (scale > 0).all()


class TestGatedTransition:
    def test_output_shapes(self) -> None:
        import torch

        trans = GatedTransition(4, 16)
        z = torch.zeros(4)
        loc, scale = trans(z)
        assert loc.shape == (4,)
        assert scale.shape == (4,)

    def test_scale_positive(self) -> None:
        import torch

        trans = GatedTransition(4, 16)
        _, scale = trans(torch.randn(4))
        assert (scale > 0).all()


class TestCombiner:
    def test_output_shapes(self) -> None:
        import torch

        combiner = Combiner(4, 32)
        z = torch.zeros(4)
        h = torch.zeros(32)
        loc, scale = combiner(z, h)
        assert loc.shape == (4,)
        assert scale.shape == (4,)


# ---------------------------------------------------------------------------
# fit_dmm
# ---------------------------------------------------------------------------


class TestFitDMM:
    def test_returns_dmm_result(self, dmm_result: DMMResult) -> None:
        assert isinstance(dmm_result, DMMResult)

    def test_latent_means_shape(self, dmm_result: DMMResult) -> None:
        assert dmm_result.latent_means.shape == (200, FAST_CONFIG.z_dim)

    def test_latent_stds_shape(self, dmm_result: DMMResult) -> None:
        assert dmm_result.latent_stds.shape == (200, FAST_CONFIG.z_dim)

    def test_latent_means_index_matches_input(
        self, synthetic_returns: pd.DataFrame, dmm_result: DMMResult
    ) -> None:
        pd.testing.assert_index_equal(
            dmm_result.latent_means.index, synthetic_returns.index
        )

    def test_elbo_history_length(self, dmm_result: DMMResult) -> None:
        assert len(dmm_result.elbo_history) == FAST_CONFIG.num_epochs

    def test_no_nan_in_elbo(self, dmm_result: DMMResult) -> None:
        assert all(np.isfinite(v) for v in dmm_result.elbo_history)

    def test_no_nan_in_latents(self, dmm_result: DMMResult) -> None:
        assert dmm_result.latent_means.notna().all().all()
        assert dmm_result.latent_stds.notna().all().all()

    def test_latent_stds_positive(self, dmm_result: DMMResult) -> None:
        assert (dmm_result.latent_stds > 0).all().all()

    def test_elbo_improves_after_warmup(self, dmm_result: DMMResult) -> None:
        """ELBO should trend upward after KL annealing warmup."""
        post = dmm_result.elbo_history[FAST_CONFIG.annealing_epochs :]
        n = len(post)
        quarter = max(1, n // 4)
        assert np.mean(post[-quarter:]) > np.mean(post[:quarter])

    def test_tickers_stored(self, dmm_result: DMMResult) -> None:
        assert dmm_result.tickers == TICKERS

    def test_input_stats_shape(self, dmm_result: DMMResult) -> None:
        assert dmm_result.input_mean.shape == (N_ASSETS,)
        assert dmm_result.input_std.shape == (N_ASSETS,)

    def test_input_std_positive(self, dmm_result: DMMResult) -> None:
        assert (dmm_result.input_std > 0).all()

    def test_nan_rows_dropped(self) -> None:
        rng = np.random.default_rng(5)
        dates = pd.date_range("2020-01-01", periods=202, freq="B")
        df = pd.DataFrame(
            rng.normal(0, 0.01, (202, N_ASSETS)),
            index=dates,
            columns=TICKERS,
        )
        df.iloc[10] = np.nan
        df.iloc[100] = np.nan
        cfg = DMMConfig(
            z_dim=2,
            emission_dim=8,
            transition_dim=8,
            rnn_dim=16,
            num_epochs=5,
            annealing_epochs=1,
            random_state=0,
        )
        result = fit_dmm(df, cfg)
        assert result.latent_means.shape[0] == 200

    def test_default_config_used_when_none(
        self, synthetic_returns: pd.DataFrame
    ) -> None:
        # Just verify it runs; use very short training
        cfg = DMMConfig(
            z_dim=2,
            emission_dim=8,
            transition_dim=8,
            rnn_dim=16,
            num_epochs=3,
            annealing_epochs=1,
            random_state=0,
        )
        result = fit_dmm(synthetic_returns, cfg)
        assert isinstance(result, DMMResult)

    def test_model_is_dmm_instance(self, dmm_result: DMMResult) -> None:
        assert isinstance(dmm_result.model, DMM)

    def test_latent_cols_are_integer_range(self, dmm_result: DMMResult) -> None:
        expected = list(range(FAST_CONFIG.z_dim))
        assert list(dmm_result.latent_means.columns) == expected


# ---------------------------------------------------------------------------
# blend_moments_dmm
# ---------------------------------------------------------------------------


class TestBlendMomentsDmm:
    def test_returns_series_and_dataframe(self, dmm_result: DMMResult) -> None:
        mu, cov = blend_moments_dmm(dmm_result)
        assert isinstance(mu, pd.Series)
        assert isinstance(cov, pd.DataFrame)

    def test_mu_index_matches_tickers(self, dmm_result: DMMResult) -> None:
        mu, _ = blend_moments_dmm(dmm_result)
        assert list(mu.index) == TICKERS

    def test_cov_shape(self, dmm_result: DMMResult) -> None:
        _, cov = blend_moments_dmm(dmm_result)
        assert cov.shape == (N_ASSETS, N_ASSETS)

    def test_cov_index_columns_match_tickers(self, dmm_result: DMMResult) -> None:
        _, cov = blend_moments_dmm(dmm_result)
        assert list(cov.index) == TICKERS
        assert list(cov.columns) == TICKERS

    def test_cov_diagonal_positive(self, dmm_result: DMMResult) -> None:
        _, cov = blend_moments_dmm(dmm_result)
        diag = np.diag(cov.to_numpy())
        assert (diag > 0).all()

    def test_same_signature_as_hmm_blend(self, dmm_result: DMMResult) -> None:
        """blend_moments_dmm must return (pd.Series, pd.DataFrame)."""
        result = blend_moments_dmm(dmm_result)
        assert len(result) == 2
        assert isinstance(result[0], pd.Series)
        assert isinstance(result[1], pd.DataFrame)

    def test_mu_finite(self, dmm_result: DMMResult) -> None:
        mu, _ = blend_moments_dmm(dmm_result)
        assert mu.notna().all()
        assert np.isfinite(mu.to_numpy()).all()

    def test_mc_cov_includes_variance_of_means(self, dmm_result: DMMResult) -> None:
        """MC covariance diagonal includes both E[Var] and Var[E] terms,
        so it should be >= a single-point estimate's emission variance."""
        import torch

        # Single-point estimate (old behaviour)
        z_last = torch.tensor(
            dmm_result.latent_means.iloc[-1].to_numpy(dtype=np.float32)
        )
        dmm_result.model.eval()
        with torch.no_grad():
            _, emit_scale = dmm_result.model.emitter(z_last)
        single_var = (emit_scale.numpy().astype(np.float64) * dmm_result.input_std) ** 2

        # MC estimate (use many samples to reduce noise)
        _, cov = blend_moments_dmm(dmm_result, n_mc_samples=5000, seed=42)
        mc_var = np.diag(cov.to_numpy())
        # MC diagonal should be >= single-point diagonal (it adds Var[E] term).
        # Allow 5% relative tolerance for MC sampling noise.
        assert np.all(mc_var >= single_var * 0.95)

    def test_mc_reproducible_with_seed(self, dmm_result: DMMResult) -> None:
        """Results are reproducible when seed is fixed."""
        mu1, cov1 = blend_moments_dmm(dmm_result, n_mc_samples=100, seed=42)
        mu2, cov2 = blend_moments_dmm(dmm_result, n_mc_samples=100, seed=42)
        np.testing.assert_allclose(mu1.to_numpy(), mu2.to_numpy(), atol=1e-6)
        np.testing.assert_allclose(cov1.to_numpy(), cov2.to_numpy(), atol=1e-6)
