"""Tests for SchurComplementary factory and configuration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.optimization import SchurComplementary

from optimizer.cluster import HierarchicalClusteringConfig, LinkageMethodType
from optimizer.distance import DistanceConfig, DistanceEstimatorType
from optimizer.exceptions import ConfigurationError
from optimizer.optimization import (
    SchurComplementaryConfig,
    build_schur_complementary,
)


@pytest.fixture(scope="module")
def returns():
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    return prices_to_returns(prices)


class TestSchurComplementaryConfig:
    def test_when_default_then_gamma_is_balanced(self) -> None:
        # skfolio default gamma=0.5 (balanced between HRP and MVP).
        assert SchurComplementaryConfig().gamma == 0.5

    def test_when_default_then_keep_monotonic_true(self) -> None:
        assert SchurComplementaryConfig().keep_monotonic is True

    def test_when_constructed_then_frozen(self) -> None:
        cfg = SchurComplementaryConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.gamma = 0.0  # type: ignore[misc]

    def test_when_gamma_negative_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="gamma"):
            SchurComplementaryConfig(gamma=-0.1)

    def test_when_gamma_above_one_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="gamma"):
            SchurComplementaryConfig(gamma=1.1)


class TestPresets:
    def test_when_for_hrp_anchor_then_gamma_zero(self) -> None:
        cfg = SchurComplementaryConfig.for_hrp_anchor()
        assert cfg.gamma == 0.0

    def test_when_for_mvp_anchor_then_gamma_one(self) -> None:
        cfg = SchurComplementaryConfig.for_mvp_anchor()
        assert cfg.gamma == 1.0

    def test_when_for_balanced_then_gamma_half(self) -> None:
        cfg = SchurComplementaryConfig.for_balanced()
        assert cfg.gamma == 0.5


class TestBuildSchurComplementary:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_schur_complementary(SchurComplementaryConfig())
        assert isinstance(est, SchurComplementary)

    @pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_when_gamma_set_then_forwarded(self, gamma: float) -> None:
        cfg = SchurComplementaryConfig(gamma=gamma)
        est = build_schur_complementary(cfg)
        assert est.gamma == gamma

    def test_when_distance_config_then_distance_estimator_built(self) -> None:
        from skfolio.distance import KendallDistance

        cfg = SchurComplementaryConfig(
            distance_config=DistanceConfig(estimator=DistanceEstimatorType.KENDALL)
        )
        est = build_schur_complementary(cfg)
        assert isinstance(est.distance_estimator, KendallDistance)

    def test_when_clustering_config_then_clustering_estimator_built(self) -> None:
        from skfolio.cluster import HierarchicalClustering

        cfg = SchurComplementaryConfig(
            clustering_config=HierarchicalClusteringConfig(
                linkage_method=LinkageMethodType.SINGLE
            )
        )
        est = build_schur_complementary(cfg)
        assert isinstance(est.hierarchical_clustering_estimator, HierarchicalClustering)

    def test_when_kwargs_passed_then_forwarded(self) -> None:
        est = build_schur_complementary(
            SchurComplementaryConfig(), raise_on_failure=False
        )
        assert est.raise_on_failure is False

    def test_when_config_none_then_default(self) -> None:
        est = build_schur_complementary(None)
        assert isinstance(est, SchurComplementary)

    def test_when_prior_config_then_inner_prior_built(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        cfg = SchurComplementaryConfig(prior_config=MomentEstimationConfig())
        est = build_schur_complementary(cfg)
        assert isinstance(est.prior_estimator, EmpiricalPrior)


class TestIntegration:
    def test_when_balanced_then_fits(self, returns) -> None:
        est = build_schur_complementary(SchurComplementaryConfig.for_balanced())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
        assert np.isclose(est.weights_.sum(), 1.0, atol=1e-3)

    def test_when_hrp_anchor_then_fits(self, returns) -> None:
        est = build_schur_complementary(SchurComplementaryConfig.for_hrp_anchor())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)

    def test_when_mvp_anchor_then_fits(self, returns) -> None:
        est = build_schur_complementary(SchurComplementaryConfig.for_mvp_anchor())
        est.fit(returns)
        assert est.weights_.shape == (returns.shape[1],)
