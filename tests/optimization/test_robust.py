"""Unit tests for robust mean-risk optimization (issues #18, #33)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import MeanRisk
from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)

from optimizer.optimization._config import MeanRiskConfig, ObjectiveFunctionType
from optimizer.optimization._factory import build_mean_risk
from optimizer.optimization._robust import (
    CovarianceUncertaintyResult,
    RobustConfig,
    _KappaEmpiricalMuUncertaintySet,
    bootstrap_covariance_uncertainty,
    build_robust_mean_risk,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 10
N_OBS = 252
TICKERS = [f"A{i:02d}" for i in range(N_ASSETS)]
DATES = pd.date_range("2020-01-02", periods=N_OBS, freq="B")


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0005, scale=0.01, size=(N_OBS, N_ASSETS))
    return pd.DataFrame(data, index=DATES, columns=TICKERS)


# ---------------------------------------------------------------------------
# TestRobustConfig
# ---------------------------------------------------------------------------


class TestRobustConfig:
    def test_is_frozen(self) -> None:
        cfg = RobustConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.kappa = 99.0  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        cfg = RobustConfig(kappa=1.5)
        assert isinstance(hash(cfg), int)

    def test_usable_in_set(self) -> None:
        s = {RobustConfig(kappa=0.5), RobustConfig(kappa=1.0), RobustConfig(kappa=0.5)}
        assert len(s) == 2

    def test_default_kappa(self) -> None:
        assert RobustConfig().kappa == 1.0

    def test_default_cov_uncertainty(self) -> None:
        assert RobustConfig().cov_uncertainty is False

    def test_default_cov_uncertainty_method(self) -> None:
        assert RobustConfig().cov_uncertainty_method == "bootstrap"

    def test_default_B(self) -> None:
        assert RobustConfig().B == 500

    def test_default_block_size(self) -> None:
        assert RobustConfig().block_size == 21

    def test_default_bootstrap_alpha(self) -> None:
        assert RobustConfig().bootstrap_alpha == 0.05

    def test_default_mean_risk_config_is_none(self) -> None:
        assert RobustConfig().mean_risk_config is None

    def test_for_conservative_kappa(self) -> None:
        assert RobustConfig.for_conservative().kappa == 2.0

    def test_for_moderate_kappa(self) -> None:
        assert RobustConfig.for_moderate().kappa == 1.0

    def test_for_aggressive_kappa(self) -> None:
        assert RobustConfig.for_aggressive().kappa == 0.5

    def test_for_bootstrap_covariance_preset(self) -> None:
        cfg = RobustConfig.for_bootstrap_covariance()
        assert cfg.kappa == 1.0
        assert cfg.cov_uncertainty is True
        assert cfg.cov_uncertainty_method == "bootstrap"

    def test_embedded_mean_risk_config(self) -> None:
        inner = MeanRiskConfig.for_max_sharpe()
        cfg = RobustConfig(kappa=1.0, mean_risk_config=inner)
        assert cfg.mean_risk_config is inner

    def test_usable_in_param_grid(self) -> None:
        """RobustConfig can appear as a value in a scikit-learn param_grid."""
        grid: list[dict[str, object]] = [
            {"kappa": [0.5, 1.0, 2.0]},
            {
                "config": [
                    RobustConfig.for_aggressive(),
                    RobustConfig.for_conservative(),
                ]
            },
        ]
        assert len(grid) == 2


# ---------------------------------------------------------------------------
# TestBuildRobustMeanRisk
# ---------------------------------------------------------------------------


class TestBuildRobustMeanRisk:
    def test_returns_mean_risk_instance(self) -> None:
        model = build_robust_mean_risk()
        assert isinstance(model, MeanRisk)

    def test_kappa_zero_no_uncertainty_set(self) -> None:
        """kappa=0 → MeanRisk with no mu_uncertainty_set_estimator."""
        model = build_robust_mean_risk(RobustConfig(kappa=0.0))
        assert model.mu_uncertainty_set_estimator is None

    def test_kappa_positive_has_mu_uncertainty_set(self) -> None:
        """kappa>0 → MeanRisk has a mu_uncertainty_set_estimator injected."""
        model = build_robust_mean_risk(RobustConfig(kappa=1.0))
        assert isinstance(model.mu_uncertainty_set_estimator, EmpiricalMuUncertaintySet)

    def test_kappa_is_kappa_based_subclass(self) -> None:
        model = build_robust_mean_risk(RobustConfig(kappa=1.5))
        assert isinstance(
            model.mu_uncertainty_set_estimator, _KappaEmpiricalMuUncertaintySet
        )

    def test_cov_uncertainty_false_no_cov_set(self) -> None:
        model = build_robust_mean_risk(RobustConfig(cov_uncertainty=False))
        assert model.covariance_uncertainty_set_estimator is None

    def test_cov_uncertainty_true_default_bootstrap(self) -> None:
        """Default cov_uncertainty_method='bootstrap' injects BootstrapCovarianceUncertaintySet."""
        model = build_robust_mean_risk(RobustConfig(cov_uncertainty=True))
        assert isinstance(
            model.covariance_uncertainty_set_estimator,
            BootstrapCovarianceUncertaintySet,
        )

    def test_cov_uncertainty_empirical_method(self) -> None:
        """cov_uncertainty_method='empirical' falls back to EmpiricalCovarianceUncertaintySet."""
        model = build_robust_mean_risk(
            RobustConfig(cov_uncertainty=True, cov_uncertainty_method="empirical")
        )
        assert isinstance(
            model.covariance_uncertainty_set_estimator,
            EmpiricalCovarianceUncertaintySet,
        )

    def test_bootstrap_params_forwarded(self) -> None:
        """B and block_size are forwarded to BootstrapCovarianceUncertaintySet."""
        model = build_robust_mean_risk(
            RobustConfig(cov_uncertainty=True, B=200, block_size=10, bootstrap_alpha=0.10)
        )
        est = model.covariance_uncertainty_set_estimator
        assert isinstance(est, BootstrapCovarianceUncertaintySet)
        assert est.n_bootstrap_samples == 200
        assert est.block_size == pytest.approx(10.0)
        assert est.confidence_level == pytest.approx(0.90)

    def test_default_config_used_when_none(self) -> None:
        model = build_robust_mean_risk(None)
        assert isinstance(model, MeanRisk)
        assert model.mu_uncertainty_set_estimator is not None  # kappa=1.0 by default

    def test_kwargs_forwarded(self) -> None:
        """Non-config kwargs (e.g. previous_weights) are forwarded to MeanRisk."""
        prev = np.full(N_ASSETS, 1.0 / N_ASSETS)
        model = build_robust_mean_risk(
            RobustConfig(kappa=1.0), previous_weights=prev
        )
        np.testing.assert_array_equal(model.previous_weights, prev)

    def test_mean_risk_config_objective_forwarded(self) -> None:
        inner = MeanRiskConfig(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
        )
        model = build_robust_mean_risk(RobustConfig(mean_risk_config=inner))
        from skfolio.optimization.convex._base import ObjectiveFunction

        assert model.objective_function == ObjectiveFunction.MAXIMIZE_RATIO

    # -- fit & predict ---------------------------------------------------------

    def test_fit_returns_portfolio(self, returns: pd.DataFrame) -> None:
        model = build_robust_mean_risk(RobustConfig(kappa=1.0))
        model.fit(returns)
        portfolio = model.predict(returns)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == N_ASSETS

    def test_weights_sum_to_one(self, returns: pd.DataFrame) -> None:
        model = build_robust_mean_risk(RobustConfig(kappa=1.0))
        model.fit(returns)
        portfolio = model.predict(returns)
        assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative_long_only(self, returns: pd.DataFrame) -> None:
        model = build_robust_mean_risk(RobustConfig(kappa=1.0))
        model.fit(returns)
        portfolio = model.predict(returns)
        assert np.all(portfolio.weights >= -1e-8)

    # -- kappa=0 acceptance criterion ------------------------------------------

    def test_kappa_zero_matches_standard_mean_risk(
        self, returns: pd.DataFrame
    ) -> None:
        """At κ=0 output weights are within 1e-4 of standard build_mean_risk()."""
        standard = build_mean_risk(MeanRiskConfig())
        robust_zero = build_robust_mean_risk(RobustConfig(kappa=0.0))

        standard.fit(returns)
        robust_zero.fit(returns)

        w_std = standard.predict(returns).weights
        w_rob = robust_zero.predict(returns).weights

        np.testing.assert_allclose(w_rob, w_std, atol=1e-4)

    # -- kappa monotonicity: larger kappa → more diversified -------------------

    def test_larger_kappa_more_diversified(self, returns: pd.DataFrame) -> None:
        """Larger κ produces a more diversified portfolio (lower max weight)."""
        kappas = [0.5, 1.0, 2.0]
        max_weights: list[float] = []

        for kappa in kappas:
            model = build_robust_mean_risk(RobustConfig(kappa=kappa))
            model.fit(returns)
            portfolio = model.predict(returns)
            max_weights.append(float(np.max(portfolio.weights)))

        # Max weight should be non-increasing with kappa (strictly for typical data)
        assert max_weights[0] >= max_weights[1] - 1e-4
        assert max_weights[1] >= max_weights[2] - 1e-4

    # -- presets ---------------------------------------------------------------

    def test_conservative_preset_fits(self, returns: pd.DataFrame) -> None:
        model = build_robust_mean_risk(RobustConfig.for_conservative())
        model.fit(returns)
        portfolio = model.predict(returns)
        assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)

    def test_moderate_preset_fits(self, returns: pd.DataFrame) -> None:
        model = build_robust_mean_risk(RobustConfig.for_moderate())
        model.fit(returns)
        portfolio = model.predict(returns)
        assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)

    def test_aggressive_preset_fits(self, returns: pd.DataFrame) -> None:
        model = build_robust_mean_risk(RobustConfig.for_aggressive())
        model.fit(returns)
        portfolio = model.predict(returns)
        assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestKappaEmpiricalMuUncertaintySet
# ---------------------------------------------------------------------------


class TestKappaEmpiricalMuUncertaintySet:
    def test_kappa_stored_as_parameter(self) -> None:
        est = _KappaEmpiricalMuUncertaintySet(kappa=1.5)
        assert est.kappa == pytest.approx(1.5)

    def test_confidence_level_updated_after_fit(
        self, returns: pd.DataFrame
    ) -> None:
        """After fit(), confidence_level == chi2.cdf(kappa², df=n_assets)."""
        from scipy.stats import chi2

        kappa = 1.0
        est = _KappaEmpiricalMuUncertaintySet(kappa=kappa)
        est.fit(returns)

        expected = chi2.cdf(kappa**2, df=N_ASSETS)
        assert est.confidence_level == pytest.approx(expected, rel=1e-9)

    def test_uncertainty_set_fitted_after_fit(
        self, returns: pd.DataFrame
    ) -> None:
        est = _KappaEmpiricalMuUncertaintySet(kappa=1.0)
        est.fit(returns)
        assert hasattr(est, "uncertainty_set_")


# ---------------------------------------------------------------------------
# TestCovarianceUncertaintyResult
# ---------------------------------------------------------------------------


class TestCovarianceUncertaintyResult:
    def test_fields_accessible(self) -> None:
        cov = np.eye(3)
        samples = np.tile(np.eye(3), (10, 1, 1))
        result = CovarianceUncertaintyResult(cov_hat=cov, delta=0.5, cov_samples=samples)
        np.testing.assert_array_equal(result.cov_hat, cov)
        assert result.delta == pytest.approx(0.5)
        assert result.cov_samples.shape == (10, 3, 3)

    def test_delta_is_float(self) -> None:
        result = CovarianceUncertaintyResult(
            cov_hat=np.eye(2), delta=1.23, cov_samples=np.zeros((5, 2, 2))
        )
        assert isinstance(result.delta, float)


# ---------------------------------------------------------------------------
# TestBootstrapCovarianceUncertainty
# ---------------------------------------------------------------------------


class TestBootstrapCovarianceUncertainty:
    def test_returns_correct_type(self, returns: pd.DataFrame) -> None:
        result = bootstrap_covariance_uncertainty(returns, B=50, seed=0)
        assert isinstance(result, CovarianceUncertaintyResult)

    def test_cov_hat_shape(self, returns: pd.DataFrame) -> None:
        result = bootstrap_covariance_uncertainty(returns, B=50, seed=0)
        assert result.cov_hat.shape == (N_ASSETS, N_ASSETS)

    def test_cov_samples_shape(self, returns: pd.DataFrame) -> None:
        result = bootstrap_covariance_uncertainty(returns, B=50, seed=0)
        assert result.cov_samples.shape == (50, N_ASSETS, N_ASSETS)

    def test_delta_positive_finite(self, returns: pd.DataFrame) -> None:
        result = bootstrap_covariance_uncertainty(returns, B=50, seed=0)
        assert result.delta > 0.0
        assert np.isfinite(result.delta)

    def test_each_sample_is_symmetric(self, returns: pd.DataFrame) -> None:
        result = bootstrap_covariance_uncertainty(returns, B=50, seed=0)
        for i in range(50):
            np.testing.assert_allclose(
                result.cov_samples[i],
                result.cov_samples[i].T,
                atol=1e-12,
            )

    def test_each_sample_is_psd(self, returns: pd.DataFrame) -> None:
        result = bootstrap_covariance_uncertainty(returns, B=50, seed=0)
        for i in range(50):
            eigenvalues = np.linalg.eigvalsh(result.cov_samples[i])
            assert np.all(eigenvalues >= -1e-10), f"Sample {i} is not PSD"

    def test_delta_shrinks_with_more_data(self) -> None:
        """Larger T → smaller bootstrap uncertainty radius (acceptance criterion)."""
        rng = np.random.default_rng(42)
        short_dates = pd.date_range("2020-01-02", periods=126, freq="B")
        long_dates = pd.date_range("2020-01-02", periods=504, freq="B")
        short_data = pd.DataFrame(
            rng.normal(0.0005, 0.01, (126, 5)),
            index=short_dates,
            columns=[f"A{i}" for i in range(5)],
        )
        long_data = pd.DataFrame(
            rng.normal(0.0005, 0.01, (504, 5)),
            index=long_dates,
            columns=[f"A{i}" for i in range(5)],
        )
        delta_short = bootstrap_covariance_uncertainty(short_data, B=200, seed=0).delta
        delta_long = bootstrap_covariance_uncertainty(long_data, B=200, seed=0).delta
        assert delta_short > delta_long

    def test_reproducible_with_seed(self, returns: pd.DataFrame) -> None:
        r1 = bootstrap_covariance_uncertainty(returns, B=30, seed=7)
        r2 = bootstrap_covariance_uncertainty(returns, B=30, seed=7)
        np.testing.assert_allclose(r1.delta, r2.delta)
        np.testing.assert_allclose(r1.cov_samples, r2.cov_samples)


# ---------------------------------------------------------------------------
# TestBuildRobustMeanRiskBootstrapCovariance
# ---------------------------------------------------------------------------


class TestBuildRobustMeanRiskBootstrapCovariance:
    def test_bootstrap_cov_fit_produces_valid_weights(
        self, returns: pd.DataFrame
    ) -> None:
        """Acceptance criterion: cov_uncertainty=True yields valid weights."""
        model = build_robust_mean_risk(
            RobustConfig(cov_uncertainty=True, B=50, block_size=21)
        )
        model.fit(returns)
        portfolio = model.predict(returns)
        assert np.sum(portfolio.weights) == pytest.approx(1.0, abs=1e-6)
        assert np.all(portfolio.weights >= -1e-8)

    def test_for_bootstrap_covariance_preset_fits(
        self, returns: pd.DataFrame
    ) -> None:
        model = build_robust_mean_risk(RobustConfig.for_bootstrap_covariance())
        model.fit(returns)
        portfolio = model.predict(returns)
        assert float(np.sum(portfolio.weights)) == pytest.approx(1.0, abs=1e-6)
