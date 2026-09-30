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
    """Compose a distance estimator from config; return ``None`` if unset.

    Args:
        config: Distance metric configuration. Pass ``None`` to omit the
            distance step entirely (the caller supplies no estimator).

    Returns:
        The constructed estimator, or ``None`` when *config* is ``None``.
    """
    if config is None:
        return None
    return build_distance(config)


def build_clustering_or_none(
    config: HierarchicalClusteringConfig | None,
) -> HierarchicalClustering | None:
    """Compose a clustering estimator from config; return ``None`` if unset.

    Args:
        config: Hierarchical clustering configuration. Pass ``None`` to let the
            optimizer use its built-in default clustering behaviour.

    Returns:
        The constructed estimator, or ``None`` when *config* is ``None``.
    """
    if config is None:
        return None
    return build_hierarchical_clustering(config)
