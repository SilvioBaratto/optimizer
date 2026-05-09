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
    """Immutable configuration for :class:`MaximumDiversification`.

    Default is long-only (``min_weights=0.0``); the diversification
    ratio is undefined for short positions.

    Parameters
    ----------
    prior_config : MomentEstimationConfig or None
        Inner prior configuration. ``None`` defers to skfolio default.
    min_weights : float
        Lower bound on asset weights. Default ``0.0`` (long-only).
    max_weights : float
        Upper bound on asset weights.
    transaction_costs : float
        Linear transaction costs penalising turnover.
    management_fees : float
        Linear management fees proportional to position size.
    l1_coef : float
        L1 regularisation coefficient.
    l2_coef : float
        L2 regularisation coefficient.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
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
    """Build a skfolio :class:`MaximumDiversification` from *config*.

    Parameters
    ----------
    config : MaxDiversificationConfig or None
        Max-diversification configuration. ``None`` triggers default.
    prior_estimator : BasePrior or None
        Prior estimator. When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional kwargs forwarded to the wrapped optimizer.

    Returns
    -------
    MaximumDiversification
        A fitted-ready skfolio optimiser.
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
