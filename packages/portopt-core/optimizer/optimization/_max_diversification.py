"""MaximumDiversification configuration and factory.

Maximises the diversification ratio (weighted sum of individual asset
volatilities divided by portfolio volatility). The ratio is undefined
for short positions, so the default config is long-only
(``min_weights=0.0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.optimization import MaximumDiversification
from skfolio.prior._base import BasePrior

from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior


@dataclass(frozen=True)
class MaxDiversificationConfig:
    """Immutable configuration for `MaximumDiversification`.

    Default is long-only (``min_weights=0.0``); the diversification
    ratio is undefined for short positions.

    Attributes:
        prior_config: Inner prior configuration. None defers to the skfolio default.
        min_weights: Lower bound on asset weights. Default 0.0 (long-only).
        max_weights: Upper bound on asset weights.
        transaction_costs: Linear transaction costs penalising turnover.
        management_fees: Linear management fees proportional to position size.
        l1_coef: L1 regularisation coefficient.
        l2_coef: L2 regularisation coefficient.
        solver: CVXPY solver name.
        solver_params: Additional solver keyword arguments passed through to CVXPY.
    """

    prior_config: MomentEstimationConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    transaction_costs: float = 0.0
    management_fees: float = 0.0
    l1_coef: float = 0.0
    l2_coef: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None

    @classmethod
    def for_default(cls) -> MaxDiversificationConfig:
        """Default long-only preset."""
        return cls()

    @classmethod
    def for_long_only_capped(
        cls,
        max_weight: float = 0.10,
    ) -> MaxDiversificationConfig:
        """Long-only with a per-asset weight cap (default 10%)."""
        return cls(min_weights=0.0, max_weights=max_weight)


def build_max_diversification(
    config: MaxDiversificationConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> MaximumDiversification:
    """Build a skfolio `MaximumDiversification` from `config`.

    Args:
        config: Max-diversification configuration. None triggers the default
            long-only preset.
        prior_estimator: Pre-built prior estimator. When None, one is
            constructed from ``config.prior_config`` (or the skfolio default
            when that is also None).
        **kwargs: Additional keyword arguments forwarded verbatim to the
            underlying skfolio optimiser.

    Returns:
        A fitted-ready skfolio optimiser instance.
    """
    if config is None:
        config = MaxDiversificationConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return MaximumDiversification(
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        l1_coef=config.l1_coef,
        l2_coef=config.l2_coef,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


__all__ = ["MaxDiversificationConfig", "build_max_diversification"]
