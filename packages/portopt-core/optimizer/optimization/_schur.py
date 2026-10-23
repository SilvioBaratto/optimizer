"""SchurComplementary configuration and factory.

No matrix inversion — robust to small-sample covariance error.

The ``gamma`` parameter moves the allocation from Hierarchical Risk Parity
(HRP) toward the Minimum Variance Portfolio (MVP):

* ``gamma=0.0`` recovers HRP.
* ``gamma=1.0`` approaches the MVP but does not reach it in general. With
  ``keep_monotonic=True`` (default) skfolio caps gamma at ``effective_gamma_``,
  the turning point where variance stops decreasing; on strongly correlated
  universes that cap can be low and the result close to HRP. With
  ``keep_monotonic=False`` skfolio can fail on ill-conditioned covariances.
  Use a ``MeanRisk`` minimum-variance optimiser when the exact MVP is needed.
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
    """Immutable configuration for `SchurComplementary`.

    Attributes:
        gamma: HRP-to-MVP regularisation parameter in [0, 1]. 0.0 recovers
            HRP, values toward 1.0 approach the MVP without reaching it in
            general, 0.5 is the balanced skfolio default.
        keep_monotonic: Forwarded to skfolio. When ``True``, gamma is capped
            at the fitted ``effective_gamma_`` so variance decreases
            monotonically with gamma; the requested gamma may then not be
            applied in full.
        prior_config: Inner prior configuration.
        distance_config: Distance estimator configuration.
        clustering_config: Hierarchical-clustering configuration.
        min_weights: Lower bound on asset weights.
        max_weights: Upper bound on asset weights.
        transaction_costs: Linear transaction costs penalising turnover.
        management_fees: Linear management fees proportional to position size.
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
        """Return a config with gamma=0.0 to recover pure HRP."""
        return cls(gamma=0.0)

    @classmethod
    def for_mvp_anchor(cls) -> SchurComplementaryConfig:
        """Return a config with gamma=1.0, the closest anchor to the MVP.

        This approaches the Minimum Variance Portfolio but does not recover
        it in general: with ``keep_monotonic=True`` gamma is capped at the
        fitted ``effective_gamma_`` (possibly 0.0, i.e. plain HRP), and
        ``keep_monotonic=False`` can raise on ill-conditioned covariances.
        Use a ``MeanRisk`` minimum-variance optimiser for the exact MVP.
        """
        return cls(gamma=1.0)

    @classmethod
    def for_balanced(cls) -> SchurComplementaryConfig:
        """Return a config with gamma=0.5 (skfolio default) balancing HRP and MVP."""
        return cls(gamma=0.5)


def build_schur_complementary(
    config: SchurComplementaryConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> SchurComplementary:
    """Build a skfolio `SchurComplementary` optimiser from `config`.

    Args:
        config: Configuration. ``None`` triggers default ``gamma=0.5``.
        prior_estimator: When ``None``, one is built from
            ``config.prior_config`` (or the skfolio default).
        **kwargs: Additional kwargs forwarded to the wrapped optimizer.

    Returns:
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
