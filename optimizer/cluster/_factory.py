"""Factory for skfolio hierarchical clustering."""

from __future__ import annotations

from skfolio.cluster import HierarchicalClustering, LinkageMethod

from optimizer.cluster._config import HierarchicalClusteringConfig


def build_hierarchical_clustering(
    config: HierarchicalClusteringConfig,
) -> HierarchicalClustering:
    """Build a skfolio :class:`HierarchicalClustering` from *config*.

    Parameters
    ----------
    config : HierarchicalClusteringConfig
        Hierarchical clustering configuration.

    Returns
    -------
    HierarchicalClustering
        A fitted-ready estimator. ``.fit(X)`` expects a square distance
        matrix of shape ``(n_assets, n_assets)``; after fitting it
        exposes ``linkage_matrix_``, ``labels_``, and ``n_clusters_``.
    """
    return HierarchicalClustering(
        max_clusters=config.max_clusters,
        linkage_method=LinkageMethod(config.linkage_method.value),
    )
