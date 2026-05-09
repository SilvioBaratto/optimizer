"""Tests for hierarchical clustering factory."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from skfolio.cluster import HierarchicalClustering, LinkageMethod

from optimizer.cluster import (
    HierarchicalClusteringConfig,
    LinkageMethodType,
    build_hierarchical_clustering,
)
from optimizer.distance import DistanceConfig, build_distance
from optimizer.exceptions import ConfigurationError


class TestLinkageMethodType:
    def test_when_listed_then_mirrors_skfolio_linkage_method(self) -> None:
        local_names = {m.name for m in LinkageMethodType}
        skfolio_names = {m.name for m in LinkageMethod}
        assert local_names == skfolio_names

    def test_when_listed_then_values_mirror_skfolio_linkage_method(self) -> None:
        local_pairs = {(m.name, m.value) for m in LinkageMethodType}
        skfolio_pairs = {(m.name, m.value) for m in LinkageMethod}
        assert local_pairs == skfolio_pairs


class TestHierarchicalClusteringConfig:
    def test_when_default_then_ward_linkage(self) -> None:
        cfg = HierarchicalClusteringConfig()
        assert cfg.linkage_method == LinkageMethodType.WARD

    def test_when_default_then_max_clusters_none(self) -> None:
        cfg = HierarchicalClusteringConfig()
        assert cfg.max_clusters is None

    def test_when_default_then_min_cluster_size_one(self) -> None:
        cfg = HierarchicalClusteringConfig()
        assert cfg.min_cluster_size == 1

    def test_when_constructed_then_frozen_dataclass(self) -> None:
        cfg = HierarchicalClusteringConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.linkage_method = LinkageMethodType.SINGLE  # type: ignore[misc]

    def test_when_min_cluster_size_not_one_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="min_cluster_size"):
            HierarchicalClusteringConfig(min_cluster_size=2)

    def test_when_max_clusters_set_then_stored(self) -> None:
        cfg = HierarchicalClusteringConfig(max_clusters=5)
        assert cfg.max_clusters == 5


class TestPresets:
    def test_when_for_default_then_ward_linkage(self) -> None:
        cfg = HierarchicalClusteringConfig.for_default()
        assert cfg.linkage_method == LinkageMethodType.WARD

    def test_when_for_single_linkage_then_single_linkage(self) -> None:
        cfg = HierarchicalClusteringConfig.for_single_linkage()
        assert cfg.linkage_method == LinkageMethodType.SINGLE


class TestBuildHierarchicalClustering:
    @pytest.mark.parametrize("linkage", list(LinkageMethodType))
    def test_when_dispatched_then_linkage_method_forwarded(
        self,
        linkage: LinkageMethodType,
    ) -> None:
        cfg = HierarchicalClusteringConfig(linkage_method=linkage)
        est = build_hierarchical_clustering(cfg)
        assert isinstance(est, HierarchicalClustering)
        assert est.linkage_method == LinkageMethod(linkage.value)

    def test_when_max_clusters_set_then_forwarded(self) -> None:
        cfg = HierarchicalClusteringConfig(max_clusters=4)
        est = build_hierarchical_clustering(cfg)
        assert est.max_clusters == 4

    def test_when_default_then_max_clusters_none(self) -> None:
        est = build_hierarchical_clustering(HierarchicalClusteringConfig())
        assert est.max_clusters is None


class TestIntegrationWithDistance:
    @pytest.fixture(scope="class")
    def returns(self) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        n_obs, n_assets = 250, 8
        cols = [f"A{i:02d}" for i in range(n_assets)]
        return pd.DataFrame(
            rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets)),
            columns=cols,
        )

    def test_when_distance_then_clustering_fits_distance_matrix(
        self,
        returns: pd.DataFrame,
    ) -> None:
        distance_est = build_distance(DistanceConfig.for_pearson())
        distance_est.fit(returns)

        clustering = build_hierarchical_clustering(
            HierarchicalClusteringConfig.for_default()
        )
        clustering.fit(distance_est.distance_)

        n_assets = returns.shape[1]
        assert clustering.linkage_matrix_.shape == (n_assets - 1, 4)
        assert clustering.n_clusters_ >= 1
        assert clustering.labels_.shape == (n_assets,)

    def test_when_max_clusters_then_n_clusters_bounded(
        self,
        returns: pd.DataFrame,
    ) -> None:
        distance_est = build_distance(DistanceConfig.for_pearson())
        distance_est.fit(returns)

        cfg = HierarchicalClusteringConfig(max_clusters=3)
        clustering = build_hierarchical_clustering(cfg)
        clustering.fit(distance_est.distance_)

        assert clustering.n_clusters_ <= 3
