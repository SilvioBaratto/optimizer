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
        When set, must be a positive integer; skfolio passes it to
        ``scipy.cluster.hierarchy.fcluster`` with ``criterion="maxclust"``,
        which requires ``t >= 1``.
    min_cluster_size : int
        Reserved minimum cluster size. Not exposed by skfolio 1.0.6
        :class:`HierarchicalClustering`; must remain ``1``.
    """

    linkage_method: LinkageMethodType = LinkageMethodType.WARD
    max_clusters: int | None = None
    min_cluster_size: int = 1

    def __post_init__(self) -> None:
        if self.min_cluster_size != 1:
            raise ConfigurationError(
                "min_cluster_size is reserved and not supported by "
                "skfolio 1.0.6 HierarchicalClustering; must be 1"
            )
        if self.max_clusters is not None and (
            isinstance(self.max_clusters, bool) or self.max_clusters < 1
        ):
            raise ConfigurationError(
                "max_clusters must be a positive integer or None; "
                f"got {self.max_clusters!r}"
            )

    @classmethod
    def for_default(cls) -> HierarchicalClusteringConfig:
        """Default Ward-linkage preset (skfolio default)."""
        return cls(linkage_method=LinkageMethodType.WARD)

    @classmethod
    def for_single_linkage(cls) -> HierarchicalClusteringConfig:
        """Single-linkage (chaining) preset used by classic HRP."""
        return cls(linkage_method=LinkageMethodType.SINGLE)

    @classmethod
    def for_complete_linkage(cls) -> HierarchicalClusteringConfig:
        """Complete-linkage (farthest-neighbour) preset."""
        return cls(linkage_method=LinkageMethodType.COMPLETE)

    @classmethod
    def for_average_linkage(cls) -> HierarchicalClusteringConfig:
        """Average-linkage (UPGMA) preset."""
        return cls(linkage_method=LinkageMethodType.AVERAGE)

    @classmethod
    def for_ward_linkage(cls) -> HierarchicalClusteringConfig:
        """Ward-linkage (minimum-variance) preset; alias of ``for_default``."""
        return cls(linkage_method=LinkageMethodType.WARD)

    @classmethod
    def with_max_clusters(
        cls,
        max_clusters: int,
        linkage_method: LinkageMethodType = LinkageMethodType.WARD,
    ) -> HierarchicalClusteringConfig:
        """Preset that fixes the cluster count instead of auto-selecting it.

        Parameters
        ----------
        max_clusters : int
            Positive cap on the number of clusters returned by ``fit``.
        linkage_method : LinkageMethodType
            Agglomerative linkage rule. Default :attr:`LinkageMethodType.WARD`.
        """
        return cls(linkage_method=linkage_method, max_clusters=max_clusters)
