"""Tests for the DistributionallyRobustCVaR factory."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.measures import RiskMeasure
from skfolio.optimization import DistributionallyRobustCVaR, MeanRisk
from skfolio.optimization.convex._base import ObjectiveFunction

from optimizer.optimization import DRCVaRConfig, build_dr_cvar


@pytest.fixture(scope="module")
def returns():
    """SP500 returns truncated to the most recent 500 obs.

    DR-CVaR with non-zero Wasserstein radius is computationally
    expensive on the full 8312-obs dataset; trimming keeps integration
    tests under one second per fit while preserving the contract.
    """
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices).tail(500)


class TestDRCVaRConfig:
    def test_when_default_then_epsilon_0_05(self) -> None:
        assert DRCVaRConfig().epsilon == 0.05

    def test_when_default_then_cvar_beta_0_95(self) -> None:
        assert DRCVaRConfig().cvar_beta == 0.95

    def test_when_constructed_then_frozen(self) -> None:
        cfg = DRCVaRConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.epsilon = 0.0  # type: ignore[misc]


class TestPresets:
    def test_when_for_default_then_epsilon_0_05(self) -> None:
        assert DRCVaRConfig.for_default().epsilon == 0.05

    def test_when_for_aggressive_then_epsilon_0_01(self) -> None:
        assert DRCVaRConfig.for_aggressive().epsilon == 0.01

    def test_when_for_conservative_then_epsilon_0_10(self) -> None:
        assert DRCVaRConfig.for_conservative().epsilon == 0.10


class TestBuildDRCVaR:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_dr_cvar(DRCVaRConfig())
        assert isinstance(est, DistributionallyRobustCVaR)

    def test_when_epsilon_set_then_wasserstein_radius_forwarded(self) -> None:
        cfg = DRCVaRConfig(epsilon=0.07)
        est = build_dr_cvar(cfg)
        assert est.wasserstein_ball_radius == 0.07

    def test_when_cvar_beta_set_then_forwarded(self) -> None:
        cfg = DRCVaRConfig(cvar_beta=0.99)
        est = build_dr_cvar(cfg)
        assert est.cvar_beta == 0.99

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_dr_cvar(DRCVaRConfig(), raise_on_failure=False)
        # raise_on_failure on DRCVaR
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_dr_cvar(None)
        assert isinstance(est, DistributionallyRobustCVaR)

    def test_when_prior_config_then_inner_prior_built(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        cfg = DRCVaRConfig(prior_config=MomentEstimationConfig())
        est = build_dr_cvar(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)


class TestEpsilonZeroFallback:
    """epsilon=0 must collapse to plain MeanRisk(CVaR) — weights identical."""

    def test_when_epsilon_zero_then_weights_match_plain_meanrisk_cvar(
        self, returns
    ) -> None:
        from optimizer.optimization import RiskMeasureType

        cfg = DRCVaRConfig(epsilon=0.0)
        dr = build_dr_cvar(cfg)
        dr.fit(returns)

        plain = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.CVAR,
            cvar_beta=cfg.cvar_beta,
        )
        plain.fit(returns)

        # Confirm the fallback config reuses CVAR semantics.
        assert RiskMeasureType.CVAR.value == "cvar"
        assert np.allclose(dr.weights_, plain.weights_, atol=1e-8)


class TestIntegration:
    def test_when_default_then_fits(self, returns) -> None:
        est = build_dr_cvar(DRCVaRConfig.for_default())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)

    def test_when_aggressive_then_fits(self, returns) -> None:
        est = build_dr_cvar(DRCVaRConfig.for_aggressive())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)

    def test_when_conservative_then_fits(self, returns) -> None:
        est = build_dr_cvar(DRCVaRConfig.for_conservative())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
