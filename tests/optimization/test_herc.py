"""Tests for HierarchicalEqualRiskContribution factory and configuration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.measures import RiskMeasure
from skfolio.optimization import HierarchicalEqualRiskContribution

from optimizer.cluster import HierarchicalClusteringConfig, LinkageMethodType
from optimizer.distance import DistanceConfig, DistanceEstimatorType
from optimizer.optimization import (
    HERCConfig,
    RiskMeasureType,
    build_herc,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestHERCConfig:
    def test_when_default_then_variance_risk_measure(self) -> None:
        assert HERCConfig().risk_measure == RiskMeasureType.VARIANCE

    def test_when_default_then_solver_clarabel(self) -> None:
        assert HERCConfig().solver == "CLARABEL"

    def test_when_constructed_then_frozen(self) -> None:
        cfg = HERCConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.solver = "OSQP"  # type: ignore[misc]


class TestPresets:
    def test_when_for_default_then_variance(self) -> None:
        cfg = HERCConfig.for_default()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_when_for_cvar_then_cvar(self) -> None:
        cfg = HERCConfig.for_cvar()
        assert cfg.risk_measure == RiskMeasureType.CVAR


class TestBuildHERC:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_herc(HERCConfig())
        assert isinstance(est, HierarchicalEqualRiskContribution)

    def test_when_cvar_preset_then_cvar_forwarded(self) -> None:
        est = build_herc(HERCConfig.for_cvar())
        assert est.risk_measure == RiskMeasure.CVAR

    def test_when_solver_set_then_forwarded(self) -> None:
        cfg = HERCConfig(solver="SCS")
        est = build_herc(cfg)
        assert est.solver == "SCS"

    def test_when_distance_config_then_distance_estimator_built(self) -> None:
        from skfolio.distance import SpearmanDistance

        cfg = HERCConfig(
            distance_config=DistanceConfig(estimator=DistanceEstimatorType.SPEARMAN)
        )
        est = build_herc(cfg)
        assert isinstance(est.distance_estimator, SpearmanDistance)

    def test_when_clustering_config_then_clustering_estimator_built(self) -> None:
        from skfolio.cluster import HierarchicalClustering

        cfg = HERCConfig(
            clustering_config=HierarchicalClusteringConfig(
                linkage_method=LinkageMethodType.AVERAGE
            )
        )
        est = build_herc(cfg)
        assert isinstance(est.hierarchical_clustering_estimator, HierarchicalClustering)

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_herc(HERCConfig(), raise_on_failure=False)
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_herc(None)
        assert isinstance(est, HierarchicalEqualRiskContribution)

    def test_when_prior_config_then_inner_prior_built(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        cfg = HERCConfig(prior_config=MomentEstimationConfig())
        est = build_herc(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)


class TestIntegration:
    def test_when_default_then_fits_and_sums_to_one(self, returns) -> None:
        est = build_herc(HERCConfig.for_default())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_for_cvar_then_fits(self, returns) -> None:
        est = build_herc(HERCConfig.for_cvar())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
