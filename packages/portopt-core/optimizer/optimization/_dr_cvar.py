"""DistributionallyRobustCVaR configuration and factory.

``epsilon=0`` falls back to the empirical CVaR utility.

The skfolio class accepts ``wasserstein_ball_radius=epsilon``; the
local Config exposes ``epsilon`` to keep the parameter naming
consistent with the rest of the optimization module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skfolio.measures import RiskMeasure
from skfolio.optimization import DistributionallyRobustCVaR, MeanRisk
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.prior._base import BasePrior

from optimizer.exceptions import ConfigurationError
from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior


@dataclass(frozen=True)
class DRCVaRConfig:
    """Immutable configuration for `DistributionallyRobustCVaR`.

    Attributes:
        epsilon: Wasserstein-ball radius, in the units of the returns the
            optimiser is fitted on (per period, not annualised).  ``0`` falls
            back to the empirical CVaR utility.  The radius is added to the
            CVaR as a penalty on the largest weight, so a radius of the same
            order as per-period asset volatility drives the solution to equal
            weight whatever the data say; daily returns call for roughly
            ``1e-4`` to ``2e-3``.  Default ``5e-4``.
        cvar_beta: CVaR confidence level. Default ``0.95``.
        risk_aversion: Risk-aversion coefficient.
        prior_config: Inner prior configuration.
        min_weights: Lower bound on asset weights.
        max_weights: Upper bound on asset weights.
        risk_free_rate: Risk-free rate.
        solver: CVXPY solver name.
        solver_params: Additional solver parameters.
    """

    epsilon: float = 5e-4
    cvar_beta: float = 0.95
    risk_aversion: float = 1.0
    prior_config: MomentEstimationConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    risk_free_rate: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.epsilon < 0.0:
            raise ConfigurationError(
                f"epsilon (Wasserstein-ball radius) must be >= 0 (got {self.epsilon})"
            )
        if not 0.0 < self.cvar_beta < 1.0:
            raise ConfigurationError(
                f"cvar_beta must be in (0, 1) (got {self.cvar_beta})"
            )

    @classmethod
    def for_default(cls) -> DRCVaRConfig:
        """Default preset (eps=5e-4, sized for daily returns)."""
        return cls()

    @classmethod
    def for_aggressive(cls) -> DRCVaRConfig:
        """Aggressive preset (eps=1e-4) — narrow Wasserstein ball."""
        return cls(epsilon=1e-4)

    @classmethod
    def for_conservative(cls) -> DRCVaRConfig:
        """Conservative preset (eps=2e-3) — wide Wasserstein ball."""
        return cls(epsilon=2e-3)


def build_dr_cvar(
    config: DRCVaRConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> BaseOptimization:
    """Build a skfolio `DistributionallyRobustCVaR` from `config`.

    When ``epsilon == 0``, this falls back to ``MeanRisk`` maximising
    the ``return - risk_aversion * CVaR`` utility, which is the exact
    radius -> 0 limit of the DR-CVaR objective, so a sweep over radii
    that includes ``0`` stays continuous and ``risk_aversion`` keeps its
    meaning.  The return term is the (sample-weighted) mean of the prior's
    return distribution rather than the prior's ``mu``, because skfolio's
    DR-CVaR ignores ``mu``; a caller-supplied ``overwrite_expected_return``
    takes precedence over that choice and then forfeits the continuity.

    Args:
        config: DR-CVaR configuration. ``None`` triggers default.
        prior_estimator: Prior estimator. When ``None``, one is built from
            ``config.prior_config`` (or skfolio default).
        **kwargs: Additional kwargs forwarded to the wrapped optimizer.

    Returns:
        A fitted-ready skfolio optimiser. Concrete type is
        `DistributionallyRobustCVaR` when ``epsilon > 0``, or
        `MeanRisk` when ``epsilon == 0`` (empirical-CVaR fallback).
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
    """Empirical-CVaR fallback for ``epsilon=0`` — utility-maximising `MeanRisk`."""
    kwargs.setdefault("overwrite_expected_return", _sample_mean_expected_return)
    return MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
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


def _sample_mean_expected_return(w: Any, model: Any) -> Any:
    """Return term of `DistributionallyRobustCVaR` at radius zero.

    skfolio's DR-CVaR evaluates ``returns @ w`` on the raw scenarios and never
    reads the prior's ``mu``; `MeanRisk` defaults to ``mu @ w``.  Any prior
    whose ``mu`` is not the scenario mean (EW, shrunk, equilibrium, ...) would
    otherwise make the fallback optimise a different objective.
    """
    dist = model.prior_estimator_.return_distribution_
    scenario_mean = np.average(dist.returns, axis=0, weights=dist.sample_weight)
    return scenario_mean @ w


__all__ = ["DRCVaRConfig", "build_dr_cvar"]
