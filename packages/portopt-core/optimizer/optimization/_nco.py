"""NestedClustersOptimization configuration and factory.

No matrix inversion — robust to small-sample covariance error.

Inner and outer optimizers are non-serialisable estimator instances and
therefore live as factory kwargs (``inner_estimator``, ``outer_estimator``)
rather than Config fields. When the kwargs are omitted, the factory
builds default min-variance / max-Sharpe MeanRisk estimators based on
the active preset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skfolio.optimization import NestedClustersOptimization
from skfolio.optimization._base import BaseOptimization

from optimizer.cluster._config import HierarchicalClusteringConfig
from optimizer.distance._config import DistanceConfig
from optimizer.optimization._config import MeanRiskConfig
from optimizer.optimization._factory import build_mean_risk
from optimizer.optimization._hierarchical_common import (
    build_clustering_or_none,
    build_distance_or_none,
)


@dataclass(frozen=True)
class NCOConfig:
    """Immutable configuration for :class:`NestedClustersOptimization`.

    Inner and outer estimators are NOT stored on the Config because
    skfolio estimator instances are not serialisable. Pass them as
    factory kwargs to :func:`build_nco`.

    Parameters
    ----------
    distance_config : DistanceConfig or None
        Distance estimator configuration.
    clustering_config : HierarchicalClusteringConfig or None
        Hierarchical-clustering configuration.
    quantile : float
        Quantile threshold passed to the wrapped optimizer.
    inner_default : MeanRiskConfig
        Preset used to construct the default inner ``MeanRisk`` when
        the ``inner_estimator`` factory kwarg is omitted.
    outer_default : MeanRiskConfig
        Preset used to construct the default outer ``MeanRisk`` when
        the ``outer_estimator`` factory kwarg is omitted.
    """

    distance_config: DistanceConfig | None = None
    clustering_config: HierarchicalClusteringConfig | None = None
    quantile: float = 0.5
    inner_default: MeanRiskConfig = field(default_factory=MeanRiskConfig)
    outer_default: MeanRiskConfig = field(default_factory=MeanRiskConfig)

    @classmethod
    def for_default(cls) -> NCOConfig:
        """Min-variance inner + outer preset (skfolio default behaviour)."""
        return cls(
            inner_default=MeanRiskConfig.for_min_variance(),
            outer_default=MeanRiskConfig.for_min_variance(),
        )

    @classmethod
    def for_max_sharpe_outer(cls) -> NCOConfig:
        """Min-variance inner with max-Sharpe outer."""
        return cls(
            inner_default=MeanRiskConfig.for_min_variance(),
            outer_default=MeanRiskConfig.for_max_sharpe(),
        )


def build_nco(
    config: NCOConfig | None = None,
    *,
    inner_estimator: BaseOptimization | None = None,
    outer_estimator: BaseOptimization | None = None,
    **kwargs: Any,
) -> NestedClustersOptimization:
    """Build a skfolio :class:`NestedClustersOptimization` from *config*.

    Parameters
    ----------
    config : NCOConfig or None
        NCO configuration. ``None`` triggers default.
    inner_estimator : BaseOptimization or None
        Within-cluster optimizer. When ``None``, ``build_mean_risk`` is
        called with ``config.inner_default``.
    outer_estimator : BaseOptimization or None
        Across-cluster optimizer. When ``None``, ``build_mean_risk`` is
        called with ``config.outer_default``.
    **kwargs
        Additional kwargs forwarded to the wrapped optimizer.

    Returns
    -------
    NestedClustersOptimization
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = NCOConfig()

    if inner_estimator is None:
        inner_estimator = build_mean_risk(config.inner_default)
    if outer_estimator is None:
        outer_estimator = build_mean_risk(config.outer_default)

    return NestedClustersOptimization(
        inner_estimator=inner_estimator,
        outer_estimator=outer_estimator,
        distance_estimator=build_distance_or_none(config.distance_config),
        clustering_estimator=build_clustering_or_none(config.clustering_config),
        quantile=config.quantile,
        **kwargs,
    )


__all__ = ["NCOConfig", "build_nco"]
