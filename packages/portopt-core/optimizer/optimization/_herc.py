"""HierarchicalEqualRiskContribution configuration and factory.

No matrix inversion — robust to small-sample covariance error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.optimization import HierarchicalEqualRiskContribution
from skfolio.prior._base import BasePrior

from optimizer.cluster._config import HierarchicalClusteringConfig
from optimizer.distance._config import DistanceConfig
from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import RiskMeasureType
from optimizer.optimization._factory import _RISK_MEASURE_MAP
from optimizer.optimization._hierarchical_common import (
    build_clustering_or_none,
    build_distance_or_none,
)


@dataclass(frozen=True)
class HERCConfig:
    """Immutable configuration for :class:`HierarchicalEqualRiskContribution`.

    Parameters
    ----------
    risk_measure : RiskMeasureType
        Risk measure for risk-contribution equalisation across clusters.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    distance_config : DistanceConfig or None
        Distance estimator configuration.
    clustering_config : HierarchicalClusteringConfig or None
        Hierarchical-clustering configuration.
    min_weights : float
        Lower bound on asset weights.
    max_weights : float
        Upper bound on asset weights.
    transaction_costs : float
        Linear transaction costs penalising turnover.
    management_fees : float
        Linear management fees proportional to position size.
    solver : str
        CVXPY solver used by the inner ERC step.
    solver_params : dict or None
        Additional solver parameters.
    """

    risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE
    prior_config: MomentEstimationConfig | None = None
    distance_config: DistanceConfig | None = None
    clustering_config: HierarchicalClusteringConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    transaction_costs: float = 0.0
    management_fees: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None

    @classmethod
    def for_default(cls) -> HERCConfig:
        """Default variance-risk preset."""
        return cls()

    @classmethod
    def for_cvar(cls) -> HERCConfig:
        """CVaR-risk preset."""
        return cls(risk_measure=RiskMeasureType.CVAR)


def build_herc(
    config: HERCConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> HierarchicalEqualRiskContribution:
    """Build a skfolio :class:`HierarchicalEqualRiskContribution` from *config*.

    Parameters
    ----------
    config : HERCConfig or None
        HERC configuration. ``None`` triggers default.
    prior_estimator : BasePrior or None
        Prior estimator. When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional kwargs forwarded to the wrapped optimizer.

    Returns
    -------
    HierarchicalEqualRiskContribution
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = HERCConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return HierarchicalEqualRiskContribution(
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
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


__all__ = ["HERCConfig", "build_herc"]
