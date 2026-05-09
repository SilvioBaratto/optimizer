"""Tests for HierarchicalRiskParity factory and configuration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.measures import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity

from optimizer.cluster import HierarchicalClusteringConfig, LinkageMethodType
from optimizer.distance import DistanceConfig, DistanceEstimatorType
from optimizer.optimization import (
    HRPConfig,
    RiskMeasureType,
    build_hrp,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestHRPConfig:
    def test_when_default_then_variance_risk_measure(self) -> None:
        cfg = HRPConfig()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_when_default_then_min_weights_zero(self) -> None:
        assert HRPConfig().min_weights == 0.0

    def test_when_default_then_no_distance_or_clustering_config(self) -> None:
        cfg = HRPConfig()
        assert cfg.distance_config is None
        assert cfg.clustering_config is None

    def test_when_constructed_then_frozen(self) -> None:
        cfg = HRPConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.risk_measure = RiskMeasureType.CVAR  # type: ignore[misc]


class TestPresets:
    def test_when_for_default_then_variance(self) -> None:
        cfg = HRPConfig.for_default()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_when_for_cvar_then_cvar(self) -> None:
        cfg = HRPConfig.for_cvar()
        assert cfg.risk_measure == RiskMeasureType.CVAR

    def test_when_for_distance_then_distance_config_set(self) -> None:
        cfg = HRPConfig.for_distance(DistanceEstimatorType.KENDALL)
        assert cfg.distance_config is not None
        assert cfg.distance_config.estimator == DistanceEstimatorType.KENDALL


class TestBuildHRP:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_hrp(HRPConfig())
        assert isinstance(est, HierarchicalRiskParity)

    def test_when_default_then_risk_measure_variance(self) -> None:
        est = build_hrp(HRPConfig())
        assert est.risk_measure == RiskMeasure.VARIANCE

    def test_when_cvar_preset_then_cvar_forwarded(self) -> None:
        est = build_hrp(HRPConfig.for_cvar())
        assert est.risk_measure == RiskMeasure.CVAR

    def test_when_min_weights_set_then_forwarded(self) -> None:
        cfg = HRPConfig(min_weights=0.01)
        est = build_hrp(cfg)
        assert est.min_weights == 0.01

    def test_when_max_weights_set_then_forwarded(self) -> None:
        cfg = HRPConfig(max_weights=0.10)
        est = build_hrp(cfg)
        assert est.max_weights == 0.10

    def test_when_transaction_costs_set_then_forwarded(self) -> None:
        cfg = HRPConfig(transaction_costs=0.001)
        est = build_hrp(cfg)
        assert est.transaction_costs == 0.001

    def test_when_distance_config_then_distance_estimator_built(self) -> None:
        from skfolio.distance import KendallDistance

        cfg = HRPConfig(
            distance_config=DistanceConfig(estimator=DistanceEstimatorType.KENDALL)
        )
        est = build_hrp(cfg)
        assert isinstance(est.distance_estimator, KendallDistance)

    def test_when_clustering_config_then_clustering_estimator_built(self) -> None:
        from skfolio.cluster import HierarchicalClustering, LinkageMethod

        cfg = HRPConfig(
            clustering_config=HierarchicalClusteringConfig(
                linkage_method=LinkageMethodType.SINGLE
            )
        )
        est = build_hrp(cfg)
        assert isinstance(est.hierarchical_clustering_estimator, HierarchicalClustering)
        assert (
            est.hierarchical_clustering_estimator.linkage_method == LinkageMethod.SINGLE
        )

    def test_when_explicit_prior_estimator_then_forwarded(self) -> None:
        from skfolio.prior import EmpiricalPrior

        prior = EmpiricalPrior()
        est = build_hrp(HRPConfig(), prior_estimator=prior)
        assert est.prior_estimator is prior

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_hrp(HRPConfig(), raise_on_failure=False)
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_hrp(None)
        assert isinstance(est, HierarchicalRiskParity)

    def test_when_prior_config_then_inner_prior_built(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        cfg = HRPConfig(prior_config=MomentEstimationConfig())
        est = build_hrp(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)


class TestIntegration:
    def test_when_default_then_fits_and_sums_to_one(self, returns) -> None:
        est = build_hrp(HRPConfig.for_default())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-6)

    def test_when_for_cvar_then_fits(self, returns) -> None:
        est = build_hrp(HRPConfig.for_cvar())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)

    def test_when_for_distance_then_fits(self, returns) -> None:
        est = build_hrp(HRPConfig.for_distance(DistanceEstimatorType.SPEARMAN))
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
