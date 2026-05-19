"""SchurComplementary configuration and factory.

No matrix inversion — robust to small-sample covariance error.

The ``gamma`` parameter linearly interpolates between two anchors:

* ``gamma=0.0`` recovers Hierarchical Risk Parity (HRP).
* ``gamma=1.0`` recovers the Minimum Variance Portfolio (MVP).
* ``gamma=0.5`` is the skfolio default (balanced).

This default behaviour is non-obvious — callers expecting pure HRP must
explicitly request ``for_hrp_anchor()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.optimization import SchurComplementary
from skfolio.prior._base import BasePrior

from optimizer.cluster._config import HierarchicalClusteringConfig
from optimizer.distance._config import DistanceConfig
from optimizer.exceptions import ConfigurationError
from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.optimization._hierarchical_common import (
    build_clustering_or_none,
    build_distance_or_none,
)


@dataclass(frozen=True)
class SchurComplementaryConfig:
    """Immutable configuration for :class:`SchurComplementary`.

    Parameters
    ----------
    gamma : float
        HRP↔MVP interpolation parameter in :math:`[0, 1]`.
        ``0.0`` recovers HRP, ``1.0`` recovers MVP, ``0.5`` is the
        balanced skfolio default.
    keep_monotonic : bool
        Forwarded to skfolio; preserves monotonicity of risk
        contributions across the dendrogram.
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
    """

    gamma: float = 0.5
    keep_monotonic: bool = True
    prior_config: MomentEstimationConfig | None = None
    distance_config: DistanceConfig | None = None
    clustering_config: HierarchicalClusteringConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    transaction_costs: float = 0.0
    management_fees: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ConfigurationError(f"gamma must be in [0, 1] (got {self.gamma})")

    @classmethod
    def for_hrp_anchor(cls) -> SchurComplementaryConfig:
        """``gamma=0`` — recovers HRP exactly."""
        return cls(gamma=0.0)

    @classmethod
    def for_mvp_anchor(cls) -> SchurComplementaryConfig:
        """``gamma=1`` — recovers Minimum Variance Portfolio."""
        return cls(gamma=1.0)

    @classmethod
    def for_balanced(cls) -> SchurComplementaryConfig:
        """``gamma=0.5`` — balanced HRP/MVP interpolation (skfolio default)."""
        return cls(gamma=0.5)


def build_schur_complementary(
    config: SchurComplementaryConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> SchurComplementary:
    """Build a skfolio :class:`SchurComplementary` optimiser from *config*.

    Parameters
    ----------
    config : SchurComplementaryConfig or None
        Configuration. ``None`` triggers default ``gamma=0.5``.
    prior_estimator : BasePrior or None
        Prior estimator. When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional kwargs forwarded to the wrapped optimizer.

    Returns
    -------
    SchurComplementary
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = SchurComplementaryConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return SchurComplementary(
        gamma=config.gamma,
        keep_monotonic=config.keep_monotonic,
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


__all__ = ["SchurComplementaryConfig", "build_schur_complementary"]
