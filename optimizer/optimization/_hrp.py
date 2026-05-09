"""HierarchicalRiskParity configuration and factory.

No matrix inversion — robust to small-sample covariance error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.optimization import HierarchicalRiskParity
from skfolio.prior._base import BasePrior

from optimizer.cluster._config import HierarchicalClusteringConfig
from optimizer.distance._config import DistanceConfig, DistanceEstimatorType
from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import RiskMeasureType
from optimizer.optimization._factory import _RISK_MEASURE_MAP
from optimizer.optimization._hierarchical_common import (
    build_clustering_or_none,
    build_distance_or_none,
)


@dataclass(frozen=True)
class HRPConfig:
    """Immutable configuration for :class:`HierarchicalRiskParity`.

    Parameters
    ----------
    risk_measure : RiskMeasureType
        Risk measure used in recursive bisection. Default ``VARIANCE``.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration. ``None`` defers to skfolio default.
    distance_config : DistanceConfig or None
        Distance estimator configuration. ``None`` lets skfolio pick the
        default (Pearson). Built via :func:`build_distance` when set.
    clustering_config : HierarchicalClusteringConfig or None
        Hierarchical-clustering configuration. ``None`` lets skfolio
        pick the default. Built via :func:`build_hierarchical_clustering`
        when set.
    min_weights : float
        Lower bound on asset weights.
    max_weights : float
        Upper bound on asset weights.
    transaction_costs : float
        Linear transaction costs penalising turnover.
    management_fees : float
        Linear management fees proportional to position size.
    """

    risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE
    prior_config: MomentEstimationConfig | None = None
    distance_config: DistanceConfig | None = None
    clustering_config: HierarchicalClusteringConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    transaction_costs: float = 0.0
    management_fees: float = 0.0

    @classmethod
    def for_default(cls) -> HRPConfig:
        """Default variance-risk preset."""
        return cls()

    @classmethod
    def for_cvar(cls) -> HRPConfig:
        """CVaR-risk preset."""
        return cls(risk_measure=RiskMeasureType.CVAR)

    @classmethod
    def for_distance(cls, estimator: DistanceEstimatorType) -> HRPConfig:
        """Custom distance-estimator preset."""
        return cls(distance_config=DistanceConfig(estimator=estimator))


def build_hrp(
    config: HRPConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> HierarchicalRiskParity:
    """Build a skfolio :class:`HierarchicalRiskParity` optimiser from *config*.

    Parameters
    ----------
    config : HRPConfig or None
        HRP configuration. ``None`` triggers default.
    prior_estimator : BasePrior or None
        Prior estimator. When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional kwargs forwarded to the wrapped optimizer.

    Returns
    -------
    HierarchicalRiskParity
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = HRPConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return HierarchicalRiskParity(
        risk_measure=_RISK_MEASURE_MAP[config.risk_measure],
        prior_estimator=prior_estimator,
        distance_estimator=build_distance_or_none(config.distance_config),
        hierarchical_clustering_estimator=build_clustering_or_none(
            config.clustering_config
        ),
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        **kwargs,
    )


__all__ = ["HRPConfig", "build_hrp"]
