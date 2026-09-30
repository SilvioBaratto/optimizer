"""Hierarchical clustering for portfolio construction.

Wraps `HierarchicalClustering` behind a typed
`HierarchicalClusteringConfig`. Consumed by HRP/HERC/NCO/Schur
optimizers, which take a fitted clustering estimator as input.
"""

from optimizer.cluster._config import (
    HierarchicalClusteringConfig,
    LinkageMethodType,
)
from optimizer.cluster._factory import build_hierarchical_clustering

__all__ = [
    "HierarchicalClusteringConfig",
    "LinkageMethodType",
    "build_hierarchical_clustering",
]
