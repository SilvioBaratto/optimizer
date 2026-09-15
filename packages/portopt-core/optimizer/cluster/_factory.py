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
        matrix of shape ``(n_assets, n_assets)`` (e.g. a distance
        estimator's ``distance_``); after fitting it exposes
        ``condensed_distance_``, ``linkage_matrix_``, ``labels_``, and
        ``n_clusters_``.

    Notes
    -----
    This wrapper is data-source agnostic: it operates only on the distance
    matrix produced upstream, never on prices or returns. Two upstream
    preconditions must therefore already be satisfied by the caller (the
    ``distance/`` estimator that computes ``distance_``):

    * **Finite / NaN-free.** ``.fit(X)`` runs
      ``sklearn.utils.validation.validate_data`` before clustering, which
      rejects ``NaN``/``inf`` with ``ValueError: Input X contains NaN``.
      Ragged, unequal-length histories (e.g. a 5-year backfill across a
      universe where instruments list/trade on different dates) yield
      ``NaN`` pairwise codependence for non-overlapping pairs, so the
      distance matrix must be repaired/pruned upstream — this estimator
      neither imputes nor drops assets, and doing so here would silently
      corrupt the linkage.
    * **``float64``.** SQL ``Numeric`` columns (e.g. ``price_history``) read
      back as Python ``Decimal``; the ``prices_to_returns`` /
      ``distance``-estimator path must cast to float before this point.
      A ``distance_`` matrix carrying ``object``-dtype ``Decimal`` is not a
      valid input.
    """
    return HierarchicalClustering(
        max_clusters=config.max_clusters,
        linkage_method=LinkageMethod(config.linkage_method.value),
    )
