"""Shared helpers for the hierarchical optimizer family.

No matrix inversion — robust to small-sample covariance error.
"""

from __future__ import annotations

from skfolio.cluster import HierarchicalClustering
from skfolio.distance._base import BaseDistance
from skfolio.moments import BaseCovariance

from optimizer.cluster._config import HierarchicalClusteringConfig
from optimizer.cluster._factory import build_hierarchical_clustering
from optimizer.distance._config import DistanceConfig
from optimizer.distance._factory import build_distance


def build_distance_or_none(
    config: DistanceConfig | None,
    *,
    estimator: BaseDistance | None = None,
    covariance_estimator: BaseCovariance | None = None,
) -> BaseDistance | None:
    """Compose a distance estimator from config; return ``None`` if unset.

    Args:
        config: Distance metric configuration. Pass ``None`` to omit the
            distance step entirely (the caller supplies no estimator).
        estimator: Prebuilt distance estimator. It takes precedence over
            *config*, which is then ignored.
        covariance_estimator: Non-serialisable covariance estimator forwarded
            to a ``COVARIANCE`` distance built from *config*.

    Returns:
        *estimator* when given, otherwise the estimator built from *config*,
        or ``None`` when both are unset.

    Raises:
        ValueError: If *covariance_estimator* has no covariance distance to
            attach to, i.e. it is combined with *estimator*, with a ``None``
            *config*, or with a non-covariance *config*.
    """
    if covariance_estimator is not None and (estimator is not None or config is None):
        raise ValueError(
            "covariance_estimator needs a DistanceConfig with the COVARIANCE "
            "estimator to attach to; it cannot be combined with a prebuilt "
            "distance_estimator or an unset distance_config"
        )
    if estimator is not None:
        return estimator
    if config is None:
        return None
    return build_distance(config, covariance_estimator=covariance_estimator)


def build_clustering_or_none(
    config: HierarchicalClusteringConfig | None,
    *,
    estimator: HierarchicalClustering | None = None,
) -> HierarchicalClustering | None:
    """Compose a clustering estimator from config; return ``None`` if unset.

    Args:
        config: Hierarchical clustering configuration. Pass ``None`` to let the
            optimizer use its built-in default clustering behaviour.
        estimator: Prebuilt clustering estimator. It takes precedence over
            *config*, which is then ignored.

    Returns:
        *estimator* when given, otherwise the estimator built from *config*,
        or ``None`` when both are unset.
    """
    if estimator is not None:
        return estimator
    if config is None:
        return None
    return build_hierarchical_clustering(config)
