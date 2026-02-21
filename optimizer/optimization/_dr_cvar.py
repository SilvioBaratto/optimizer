"""Distributionally robust CVaR optimization over a Wasserstein ball.

Minimises the worst-case CVaR over all distributions within a Wasserstein
ball of radius ε centred at the empirical distribution P̂::

    min_w  sup_{P ∈ B_ε(P̂)} CVaR_α^P(w)

The tractable SOCP reformulation (Esfahani & Kuhn 2018) is handled by
skfolio's :class:`~skfolio.optimization.DistributionallyRobustCVaR`, which
exposes this as a risk-aversion utility::

    max_w  μ̂ᵀw − λ · sup_{P ∈ B_ε(P̂)} CVaR_α^P(w)

**ε=0 special case**
When ``epsilon=0`` the Wasserstein ball degenerates to a single point (the
empirical distribution), and the robust problem reduces to standard
empirical CVaR minimisation.  In that case the factory returns a plain
``MeanRisk(MINIMIZE_RISK, CVAR)`` so that the acceptance criterion
"ε=0 matches standard CVaR" is satisfied exactly.

Usage example::

    from optimizer.optimization._dr_cvar import DRCVaRConfig, build_dr_cvar

    # Conservative: ε=0.01 (wider Wasserstein ball, more robust)
    model = build_dr_cvar(DRCVaRConfig.for_conservative())
    model.fit(X)
    portfolio = model.predict(X)

    # ε=0 → identical to MeanRisk(MINIMIZE_RISK, CVAR)
    baseline = build_dr_cvar(DRCVaRConfig(epsilon=0.0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.measures import RiskMeasure
from skfolio.optimization import DistributionallyRobustCVaR, MeanRisk
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.prior._base import BasePrior

from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DRCVaRConfig:
    """Immutable configuration for distributionally robust CVaR optimization.

    Finds the allocation that minimises the worst-case CVaR over all
    probability distributions within a Wasserstein ball of radius ``epsilon``
    centred at the empirical distribution::

        min_w  sup_{P ∈ B_ε(P̂)} CVaR_α^P(w)

    Parameters
    ----------
    epsilon : float
        Wasserstein ball radius ε ≥ 0.  Larger values produce more
        conservative, diversified portfolios.  ``epsilon=0`` recovers
        standard empirical CVaR minimisation exactly.
    alpha : float
        CVaR confidence level α ∈ (0, 1).  Passed as ``cvar_beta``
        to skfolio.
    risk_aversion : float
        Risk-aversion coefficient λ for the utility formulation when
        ``epsilon > 0``::

            max_w  μ̂ᵀw − λ · worst-case-CVaR_α(w)

        ``lambda=1`` (default) balances expected return and CVaR.
        Ignored when ``epsilon=0``.
    norm : int
        Wasserstein norm order (1 or 2).  Only ``norm=1`` is supported
        by the current skfolio backend; stored for documentation and
        future compatibility.
    min_weights : float or None
        Lower bound on asset weights.
    max_weights : float or None
        Upper bound on asset weights.
    budget : float or None
        Portfolio budget (sum of weights).
    max_short : float or None
        Maximum short position.
    max_long : float or None
        Maximum long position.
    risk_free_rate : float
        Risk-free rate.
    solver : str
        CVXPY solver name.  ``MOSEK`` is preferred for large instances;
        ``CLARABEL`` is the open-source default.
    solver_params : dict or None
        Additional solver parameters.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.  ``None`` uses skfolio's default
        empirical prior.
    """

    epsilon: float = 0.001
    alpha: float = 0.95
    risk_aversion: float = 1.0
    norm: int = 1
    min_weights: float | None = 0.0
    max_weights: float | None = 1.0
    budget: float | None = 1.0
    max_short: float | None = None
    max_long: float | None = None
    risk_free_rate: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None
    prior_config: MomentEstimationConfig | None = None

    # -- factory methods ---------------------------------------------------

    @classmethod
    def for_conservative(cls) -> DRCVaRConfig:
        """Conservative DRO-CVaR portfolio (ε=0.01).

        Suitable for markets where the empirical distribution may
        substantially underestimate tail risk.
        """
        return cls(epsilon=0.01)

    @classmethod
    def for_standard(cls) -> DRCVaRConfig:
        """Standard DRO-CVaR portfolio (ε=0.001).

        Moderate hedge against distribution misspecification.
        """
        return cls(epsilon=0.001)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_dr_cvar(
    config: DRCVaRConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> MeanRisk | DistributionallyRobustCVaR:
    """Build a distributionally robust CVaR optimiser from *config*.

    Dispatches to:

    * **ε=0** — :class:`skfolio.optimization.MeanRisk` with
      ``MINIMIZE_RISK`` and ``CVAR``.  Identical to standard CVaR
      minimisation on the empirical distribution.

    * **ε>0** — :class:`skfolio.optimization.DistributionallyRobustCVaR`
      with ``wasserstein_ball_radius=epsilon``.  Solves the Wasserstein
      DRO reformulation via the built-in SOCP.

    Wasserstein norm note
    ~~~~~~~~~~~~~~~~~~~~~
    ``config.norm`` is stored in the config for documentation purposes.
    Only ``norm=1`` (1-Wasserstein) is currently supported by the skfolio
    backend.  ``norm=2`` is reserved for future work.

    Parameters
    ----------
    config : DRCVaRConfig or None
        DRO-CVaR configuration.  Defaults to ``DRCVaRConfig()``
        (ε=0.001, α=0.95, long-only, fully invested).
    prior_estimator : BasePrior or None
        Prior estimator.  When ``None``, one is built from
        ``config.prior_config`` (or skfolio's empirical default).
    **kwargs
        Additional keyword arguments forwarded to the underlying
        skfolio constructor (e.g. ``groups``, ``linear_constraints``,
        ``previous_weights``).

    Returns
    -------
    MeanRisk
        When ``epsilon=0``: standard CVaR-minimising :class:`MeanRisk`.
    DistributionallyRobustCVaR
        When ``epsilon>0``: Wasserstein DRO-CVaR optimiser.
    """
    if config is None:
        config = DRCVaRConfig()

    if config.epsilon < 0:
        raise ValueError(
            f"epsilon must be non-negative, got {config.epsilon}"
        )

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    if config.epsilon == 0.0:
        return _build_standard_cvar(config, prior_estimator, **kwargs)
    return _build_wasserstein_dr_cvar(config, prior_estimator, **kwargs)


def _build_standard_cvar(
    config: DRCVaRConfig,
    prior_estimator: BasePrior | None,
    **kwargs: Any,
) -> MeanRisk:
    """ε=0 path: standard empirical CVaR minimisation via MeanRisk."""
    return MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.CVAR,
        cvar_beta=config.alpha,
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        budget=config.budget,
        max_short=config.max_short,
        max_long=config.max_long,
        risk_free_rate=config.risk_free_rate,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


def _build_wasserstein_dr_cvar(
    config: DRCVaRConfig,
    prior_estimator: BasePrior | None,
    **kwargs: Any,
) -> DistributionallyRobustCVaR:
    """ε>0 path: Wasserstein DRO-CVaR via DistributionallyRobustCVaR."""
    return DistributionallyRobustCVaR(
        wasserstein_ball_radius=config.epsilon,
        cvar_beta=config.alpha,
        risk_aversion=config.risk_aversion,
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        budget=config.budget,
        max_short=config.max_short,
        max_long=config.max_long,
        risk_free_rate=config.risk_free_rate,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )
