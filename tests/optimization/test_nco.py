"""Tests for NestedClustersOptimization factory and configuration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.optimization import MeanRisk, NestedClustersOptimization

from optimizer.cluster import HierarchicalClusteringConfig, LinkageMethodType
from optimizer.distance import DistanceConfig, DistanceEstimatorType
from optimizer.optimization import (
    MeanRiskConfig,
    NCOConfig,
    build_mean_risk,
    build_nco,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestNCOConfig:
    def test_when_default_then_no_distance_or_clustering_config(self) -> None:
        cfg = NCOConfig()
        assert cfg.distance_config is None
        assert cfg.clustering_config is None

    def test_when_default_then_quantile_half(self) -> None:
        assert NCOConfig().quantile == 0.5

    def test_when_constructed_then_frozen(self) -> None:
        cfg = NCOConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.quantile = 0.25  # type: ignore[misc]


class TestPresets:
    def test_when_for_default_then_returns_default_config(self) -> None:
        cfg = NCOConfig.for_default()
        assert isinstance(cfg, NCOConfig)

    def test_when_for_max_sharpe_outer_then_returns_config(self) -> None:
        cfg = NCOConfig.for_max_sharpe_outer()
        assert isinstance(cfg, NCOConfig)


class TestBuildNCO:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_nco(NCOConfig())
        assert isinstance(est, NestedClustersOptimization)

    def test_when_inner_estimator_passed_then_forwarded(self) -> None:
        inner = build_mean_risk(MeanRiskConfig.for_min_variance())
        est = build_nco(NCOConfig(), inner_estimator=inner)
        assert est.inner_estimator is inner

    def test_when_outer_estimator_passed_then_forwarded(self) -> None:
        outer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
        est = build_nco(NCOConfig(), outer_estimator=outer)
        assert est.outer_estimator is outer

    def test_when_default_estimators_built_then_mean_risk_instances(self) -> None:
        est = build_nco(NCOConfig.for_default())
        assert isinstance(est.inner_estimator, MeanRisk)
        assert isinstance(est.outer_estimator, MeanRisk)

    def test_when_for_max_sharpe_outer_then_built_estimators_dispatched(
        self,
    ) -> None:
        from skfolio.optimization.convex._base import ObjectiveFunction

        est = build_nco(NCOConfig.for_max_sharpe_outer())
        assert isinstance(est.inner_estimator, MeanRisk)
        assert isinstance(est.outer_estimator, MeanRisk)
        assert (
            est.outer_estimator.objective_function == ObjectiveFunction.MAXIMIZE_RATIO
        )

    def test_when_distance_config_then_distance_estimator_built(self) -> None:
        from skfolio.distance import SpearmanDistance

        cfg = NCOConfig(
            distance_config=DistanceConfig(estimator=DistanceEstimatorType.SPEARMAN)
        )
        est = build_nco(cfg)
        assert isinstance(est.distance_estimator, SpearmanDistance)

    def test_when_clustering_config_then_clustering_estimator_built(self) -> None:
        from skfolio.cluster import HierarchicalClustering

        cfg = NCOConfig(
            clustering_config=HierarchicalClusteringConfig(
                linkage_method=LinkageMethodType.AVERAGE
            )
        )
        est = build_nco(cfg)
        assert isinstance(est.clustering_estimator, HierarchicalClustering)

    def test_when_quantile_set_then_forwarded(self) -> None:
        cfg = NCOConfig(quantile=0.3)
        est = build_nco(cfg)
        assert est.quantile == 0.3

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_nco(NCOConfig(), raise_on_failure=False)
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_nco(None)
        assert isinstance(est, NestedClustersOptimization)


class TestInjectedEstimators:
    def test_when_distance_estimator_kwarg_then_it_wins_over_config(self) -> None:
        from skfolio.distance import CovarianceDistance
        from skfolio.moments import LedoitWolf

        prebuilt = CovarianceDistance(covariance_estimator=LedoitWolf())
        cfg = NCOConfig(
            distance_config=DistanceConfig(estimator=DistanceEstimatorType.KENDALL)
        )
        est = build_nco(cfg, distance_estimator=prebuilt)
        assert est.distance_estimator is prebuilt

    def test_when_distance_estimator_kwarg_and_no_config_then_forwarded(self) -> None:
        from skfolio.distance import CovarianceDistance
        from skfolio.moments import LedoitWolf

        prebuilt = CovarianceDistance(covariance_estimator=LedoitWolf())
        est = build_nco(NCOConfig(), distance_estimator=prebuilt)
        assert est.distance_estimator is prebuilt

    def test_when_clustering_estimator_kwarg_then_it_wins_over_config(self) -> None:
        from skfolio.cluster import HierarchicalClustering, LinkageMethod

        prebuilt = HierarchicalClustering(linkage_method=LinkageMethod.COMPLETE)
        cfg = NCOConfig(
            clustering_config=HierarchicalClusteringConfig(
                linkage_method=LinkageMethodType.SINGLE
            )
        )
        est = build_nco(cfg, clustering_estimator=prebuilt)
        assert est.clustering_estimator is prebuilt

    def test_when_clustering_estimator_kwarg_and_no_config_then_forwarded(self) -> None:
        from skfolio.cluster import HierarchicalClustering, LinkageMethod

        prebuilt = HierarchicalClustering(linkage_method=LinkageMethod.COMPLETE)
        est = build_nco(NCOConfig(), clustering_estimator=prebuilt)
        assert est.clustering_estimator is prebuilt

    def test_when_covariance_estimator_kwarg_then_covariance_distance_uses_it(
        self,
    ) -> None:
        from skfolio.distance import CovarianceDistance
        from skfolio.moments import LedoitWolf

        cov = LedoitWolf()
        cfg = NCOConfig(distance_config=DistanceConfig.for_covariance())
        est = build_nco(cfg, covariance_estimator=cov)
        assert isinstance(est.distance_estimator, CovarianceDistance)
        assert est.distance_estimator.covariance_estimator is cov

    def test_when_covariance_estimator_without_covariance_distance_then_raises(
        self,
    ) -> None:
        from skfolio.moments import LedoitWolf

        with pytest.raises(ValueError, match="covariance_estimator"):
            build_nco(NCOConfig(), covariance_estimator=LedoitWolf())

    def test_when_covariance_estimator_and_distance_estimator_then_raises(
        self,
    ) -> None:
        from skfolio.distance import PearsonDistance
        from skfolio.moments import LedoitWolf

        with pytest.raises(ValueError, match="covariance_estimator"):
            build_nco(
                NCOConfig(distance_config=DistanceConfig.for_covariance()),
                distance_estimator=PearsonDistance(),
                covariance_estimator=LedoitWolf(),
            )


class TestIntegration:
    def test_when_default_then_fits_and_sums_to_one(self, returns) -> None:
        est = build_nco(NCOConfig.for_default())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-3)

    def test_when_for_max_sharpe_outer_then_fits(self, returns) -> None:
        est = build_nco(NCOConfig.for_max_sharpe_outer())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
