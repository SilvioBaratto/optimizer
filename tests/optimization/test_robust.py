"""Tests for the RobustMeanRisk factory."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.optimization import MeanRisk
from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)

from optimizer.optimization import (
    MeanRiskConfig,
    RobustMeanRiskConfig,
    build_mean_risk,
    build_robust_mean_risk,
)
from optimizer.uncertainty_set import (
    CovarianceUncertaintySetConfig,
    MuUncertaintySetConfig,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestRobustMeanRiskConfig:
    def test_when_default_then_uncertainty_configs_none(self) -> None:
        cfg = RobustMeanRiskConfig()
        assert cfg.mu_uncertainty_set_config is None
        assert cfg.covariance_uncertainty_set_config is None

    def test_when_default_then_holds_default_mean_risk_config(self) -> None:
        cfg = RobustMeanRiskConfig()
        assert isinstance(cfg.mean_risk_config, MeanRiskConfig)

    def test_when_constructed_then_frozen(self) -> None:
        cfg = RobustMeanRiskConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.mu_uncertainty_set_config = None  # type: ignore[misc]


class TestPresets:
    def test_when_for_conservative_then_99_confidence(self) -> None:
        cfg = RobustMeanRiskConfig.for_conservative()
        assert cfg.mu_uncertainty_set_config is not None
        assert cfg.mu_uncertainty_set_config.confidence_level == 0.99

    def test_when_for_moderate_then_95_confidence(self) -> None:
        cfg = RobustMeanRiskConfig.for_moderate()
        assert cfg.mu_uncertainty_set_config is not None
        assert cfg.mu_uncertainty_set_config.confidence_level == 0.95

    def test_when_for_aggressive_then_90_confidence(self) -> None:
        cfg = RobustMeanRiskConfig.for_aggressive()
        assert cfg.mu_uncertainty_set_config is not None
        assert cfg.mu_uncertainty_set_config.confidence_level == 0.90

    def test_when_for_bootstrap_covariance_then_bootstrap_kind(self) -> None:
        from optimizer.uncertainty_set import CovarianceUncertaintySetType

        cfg = RobustMeanRiskConfig.for_bootstrap_covariance()
        assert cfg.covariance_uncertainty_set_config is not None
        assert (
            cfg.covariance_uncertainty_set_config.kind
            == CovarianceUncertaintySetType.BOOTSTRAP
        )


class TestBuildRobustMeanRisk:
    def test_when_built_then_returns_skfolio_mean_risk(self) -> None:
        est = build_robust_mean_risk(RobustMeanRiskConfig())
        assert isinstance(est, MeanRisk)

    def test_when_mu_uncertainty_set_config_then_estimator_built(self) -> None:
        cfg = RobustMeanRiskConfig(
            mu_uncertainty_set_config=MuUncertaintySetConfig.for_empirical()
        )
        est = build_robust_mean_risk(cfg)
        assert isinstance(est.mu_uncertainty_set_estimator, EmpiricalMuUncertaintySet)

    def test_when_cov_uncertainty_set_config_then_estimator_built(self) -> None:
        cfg = RobustMeanRiskConfig(
            covariance_uncertainty_set_config=(
                CovarianceUncertaintySetConfig.for_bootstrap(
                    n_bootstrap_samples=50, random_state=0
                )
            )
        )
        est = build_robust_mean_risk(cfg)
        assert isinstance(
            est.covariance_uncertainty_set_estimator,
            BootstrapCovarianceUncertaintySet,
        )

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_robust_mean_risk(RobustMeanRiskConfig(), raise_on_failure=False)
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_robust_mean_risk(None)
        assert isinstance(est, MeanRisk)


class TestFallbackContract:
    """Both uncertainty configs None must reproduce plain build_mean_risk."""

    def test_when_no_uncertainty_then_weights_match_plain_mean_risk(
        self, returns
    ) -> None:
        mr_cfg = MeanRiskConfig.for_min_variance()
        plain = build_mean_risk(mr_cfg)
        plain.fit(returns)

        robust_cfg = RobustMeanRiskConfig(mean_risk_config=mr_cfg)
        robust = build_robust_mean_risk(robust_cfg)
        robust.fit(returns)

        assert np.allclose(plain.weights_, robust.weights_, atol=1e-8)


class TestIntegration:
    def test_when_for_moderate_then_fits(self, returns) -> None:
        est = build_robust_mean_risk(RobustMeanRiskConfig.for_moderate())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-3)

    def test_when_for_conservative_then_fits(self, returns) -> None:
        est = build_robust_mean_risk(RobustMeanRiskConfig.for_conservative())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)

    def test_when_for_bootstrap_covariance_then_fits(self, returns) -> None:
        cfg = RobustMeanRiskConfig.for_bootstrap_covariance()
        # Override with deterministic small bootstrap for test speed.
        cfg = dataclasses.replace(
            cfg,
            covariance_uncertainty_set_config=(
                CovarianceUncertaintySetConfig.for_bootstrap(
                    n_bootstrap_samples=20, random_state=0
                )
            ),
        )
        est = build_robust_mean_risk(cfg)
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
