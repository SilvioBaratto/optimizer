"""Shared helpers for the hierarchical optimizer family.

No matrix inversion — robust to small-sample covariance error.
"""

from __future__ import annotations

from skfolio.cluster import HierarchicalClustering
from skfolio.distance._base import BaseDistance

from optimizer.cluster._config import HierarchicalClusteringConfig
from optimizer.cluster._factory import build_hierarchical_clustering
from optimizer.distance._config import DistanceConfig
from optimizer.distance._factory import build_distance


def build_distance_or_none(
    config: DistanceConfig | None,
) -> BaseDistance | None:
    """Compose a distance estimator from config; return ``None`` if unset."""
    if config is None:
        return None
    return build_distance(config)


def build_clustering_or_none(
    config: HierarchicalClusteringConfig | None,
) -> HierarchicalClustering | None:
    """Compose a clustering estimator from config; return ``None`` if unset."""
    if config is None:
        return None
    return build_hierarchical_clustering(config)
