"""Configuration for hierarchical clustering selection.

The :class:`HierarchicalClusteringConfig` is a serialisable dataclass
mirroring the parameter surface of
:class:`skfolio.cluster.HierarchicalClustering`. The
:class:`LinkageMethodType` enum mirrors :class:`skfolio.cluster.LinkageMethod`
verbatim — every member is enforced equal in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.exceptions import ConfigurationError


class LinkageMethodType(str, Enum):
    """Linkage method for agglomerative hierarchical clustering."""

    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    CENTROID = "centroid"
    MEDIAN = "median"
    WARD = "ward"


@dataclass(frozen=True)
class HierarchicalClusteringConfig:
    """Immutable configuration for hierarchical clustering construction.

    Parameters
    ----------
    linkage_method : LinkageMethodType
        Agglomerative linkage rule. Default :attr:`LinkageMethodType.WARD`.
    max_clusters : int or None
        Cap on the number of clusters returned by ``fit``. ``None`` lets
        skfolio select the optimum via :func:`compute_optimal_n_clusters`.
    min_cluster_size : int
        Reserved minimum cluster size. Not exposed by skfolio 0.20.1
        :class:`HierarchicalClustering`; must remain ``1``.
    """

    linkage_method: LinkageMethodType = LinkageMethodType.WARD
    max_clusters: int | None = None
    min_cluster_size: int = 1

    def __post_init__(self) -> None:
        if self.min_cluster_size != 1:
            raise ConfigurationError(
                "min_cluster_size is reserved and not supported by "
                "skfolio 0.20.1 HierarchicalClustering; must be 1"
            )

    @classmethod
    def for_default(cls) -> HierarchicalClusteringConfig:
        """Default Ward-linkage preset (skfolio default)."""
        return cls(linkage_method=LinkageMethodType.WARD)

    @classmethod
    def for_single_linkage(cls) -> HierarchicalClusteringConfig:
        """Single-linkage (chaining) preset used by classic HRP."""
        return cls(linkage_method=LinkageMethodType.SINGLE)
