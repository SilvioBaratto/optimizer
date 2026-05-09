"""DistributionallyRobustCVaR configuration and factory.

``epsilon=0`` falls back to empirical CVaR.

The skfolio class accepts ``wasserstein_ball_radius=epsilon``; the
local Config exposes ``epsilon`` to keep the parameter naming
consistent with the rest of the optimization module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.measures import RiskMeasure
from skfolio.optimization import DistributionallyRobustCVaR, MeanRisk
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.prior._base import BasePrior

from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior


@dataclass(frozen=True)
class DRCVaRConfig:
    """Immutable configuration for :class:`DistributionallyRobustCVaR`.

    Parameters
    ----------
    epsilon : float
        Wasserstein-ball radius. ``0`` falls back to empirical CVaR.
        Default ``0.05``.
    cvar_beta : float
        CVaR confidence level. Default ``0.95``.
    risk_aversion : float
        Risk-aversion coefficient.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    min_weights : float
        Lower bound on asset weights.
    max_weights : float
        Upper bound on asset weights.
    risk_free_rate : float
        Risk-free rate.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
    """

    epsilon: float = 0.05
    cvar_beta: float = 0.95
    risk_aversion: float = 1.0
    prior_config: MomentEstimationConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    risk_free_rate: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None

    @classmethod
    def for_default(cls) -> DRCVaRConfig:
        """Default preset (eps=0.05)."""
        return cls()

    @classmethod
    def for_aggressive(cls) -> DRCVaRConfig:
        """Aggressive preset (eps=0.01) — narrow Wasserstein ball."""
        return cls(epsilon=0.01)

    @classmethod
    def for_conservative(cls) -> DRCVaRConfig:
        """Conservative preset (eps=0.10) — wide Wasserstein ball."""
        return cls(epsilon=0.10)


def build_dr_cvar(
    config: DRCVaRConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> BaseOptimization:
    """Build a skfolio :class:`DistributionallyRobustCVaR` from *config*.

    When ``epsilon == 0``, this falls back to plain ``MeanRisk(CVaR)``
    so the empirical-CVaR fallback is exact (the DRCVaR formulation
    and ``MeanRisk(CVaR)`` use different solver paths and produce
    weights that differ at ~1e-3 even at ``wasserstein_ball_radius=0``).

    Parameters
    ----------
    config : DRCVaRConfig or None
        DR-CVaR configuration. ``None`` triggers default.
    prior_estimator : BasePrior or None
        Prior estimator. When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional kwargs forwarded to the wrapped optimizer.

    Returns
    -------
    BaseOptimization
        A fitted-ready skfolio optimiser. Concrete type is
        :class:`DistributionallyRobustCVaR` when ``epsilon > 0``, or
        :class:`MeanRisk` when ``epsilon == 0`` (empirical-CVaR
        fallback).
    """
    if config is None:
        config = DRCVaRConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    if config.epsilon == 0.0:
        return _build_empirical_cvar_fallback(config, prior_estimator, **kwargs)

    return DistributionallyRobustCVaR(
        wasserstein_ball_radius=config.epsilon,
        cvar_beta=config.cvar_beta,
        risk_aversion=config.risk_aversion,
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        risk_free_rate=config.risk_free_rate,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


def _build_empirical_cvar_fallback(
    config: DRCVaRConfig,
    prior_estimator: BasePrior | None,
    **kwargs: Any,
) -> MeanRisk:
    """Empirical-CVaR fallback for ``epsilon=0`` — plain :class:`MeanRisk`."""
    return MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.CVAR,
        cvar_beta=config.cvar_beta,
        risk_aversion=config.risk_aversion,
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        risk_free_rate=config.risk_free_rate,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


__all__ = ["DRCVaRConfig", "build_dr_cvar"]
