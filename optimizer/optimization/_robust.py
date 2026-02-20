"""Robust mean-risk optimization with ellipsoidal uncertainty sets for μ and Σ.

Implements the ellipsoidal uncertainty set for the expected returns vector::

    U_μ = { μ : (μ - μ̂)ᵀ · S_μ⁻¹ · (μ - μ̂) ≤ κ² }

where:
- ``μ̂``    = estimated mean vector (sample or shrinkage estimate)
- ``S_μ``  = Σ̂ / T  (estimation error covariance of the sample mean)
- ``κ``    = robustness parameter (larger → more conservative)

The robust counterpart optimises the worst-case expected return within U_μ,
which introduces a penalty ``κ · ‖S_μ^(1/2) · w‖₂`` into the objective.

κ is related to the chi-squared confidence level β via::

    κ² = χ²_{n_assets}(β)  →  β = F_{χ²_{n_assets}}(κ²)

where ``n_assets`` is only known at fit time, so the conversion is deferred.

Usage example::

    from optimizer.optimization._robust import RobustConfig, build_robust_mean_risk

    # Conservative: κ=2.0, min-variance objective
    model = build_robust_mean_risk(RobustConfig.for_conservative())
    model.fit(X)
    portfolio = model.predict(X)

    # κ=0 → identical to standard MeanRisk (no penalty)
    baseline = build_robust_mean_risk(RobustConfig(kappa=0.0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2
from skfolio.optimization import MeanRisk
from skfolio.prior._base import BasePrior
from skfolio.uncertainty_set import (
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)

from optimizer.moments._factory import build_prior
from optimizer.optimization._config import MeanRiskConfig
from optimizer.optimization._factory import _OBJECTIVE_MAP, _RISK_MEASURE_MAP

# ---------------------------------------------------------------------------
# Private: kappa-parameterised mu uncertainty set
# ---------------------------------------------------------------------------


class _KappaEmpiricalMuUncertaintySet(EmpiricalMuUncertaintySet):  # type: ignore[misc]
    """Empirical mu uncertainty set parameterised by kappa (ellipsoid radius).

    Defers the chi-squared CDF conversion to ``fit()`` when ``n_assets`` is
    known.  The confidence level is computed as::

        β = F_{χ²_{n_assets}}(κ²)

    so that the ellipsoid radius satisfies ``κ² = χ²_{n_assets}(β)``.

    Parameters
    ----------
    kappa : float
        Ellipsoid half-width κ.
    prior_estimator : BasePrior or None
        Forwarded to :class:`EmpiricalMuUncertaintySet`.
    diagonal : bool
        If True (default) the off-diagonal elements of S_μ are set to zero.
    n_eff : float or None
        Effective number of observations for the mean estimator.  ``None``
        uses the actual observation count T.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        prior_estimator: BasePrior | None = None,
        diagonal: bool = True,
        n_eff: float | None = None,
    ) -> None:
        super().__init__(
            prior_estimator=prior_estimator,
            confidence_level=0.95,  # placeholder; overwritten in fit()
            diagonal=diagonal,
            n_eff=n_eff,
        )
        self.kappa = kappa

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> _KappaEmpiricalMuUncertaintySet:
        """Fit the uncertainty set, converting kappa to confidence_level first."""
        n_assets = np.asarray(X).shape[1]
        self.confidence_level = float(chi2.cdf(self.kappa**2, df=n_assets))
        return super().fit(X, y, **fit_params)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustConfig:
    """Immutable configuration for robust mean-risk optimization.

    Implements ellipsoidal uncertainty sets for μ (and optionally Σ) to
    hedge against estimation error in the input parameters.

    The robust objective modifies the standard mean-risk problem by assuming
    the true mean lies within the ellipsoid U_μ and optimising for the
    worst-case expected return::

        min_w  ρ(w)       subject to  min_{μ ∈ U_μ} μᵀw ≥ target

    The worst-case mean within U_μ is::

        min_{μ ∈ U_μ} μᵀw  =  μ̂ᵀw − κ · ‖S_μ^(1/2) · w‖₂

    so the penalty grows with κ and with the portfolio's exposure to
    estimation uncertainty.

    Parameters
    ----------
    kappa : float
        Ellipsoidal uncertainty radius for μ.  Larger κ produces more
        conservative (diversified) portfolios.  ``kappa=0`` recovers the
        standard (non-robust) MeanRisk exactly.
    cov_uncertainty : bool
        If True, also apply an empirical covariance uncertainty set
        ``EmpiricalCovarianceUncertaintySet(confidence_level=0.95)``
        to hedge against covariance estimation error.
    mean_risk_config : MeanRiskConfig or None
        Embedded mean-risk configuration (objective, risk measure,
        weight bounds, …).  ``None`` uses ``MeanRiskConfig()`` defaults
        (minimum-variance, long-only, fully invested).
    """

    kappa: float = 1.0
    cov_uncertainty: bool = False
    mean_risk_config: MeanRiskConfig | None = None

    # -- factory methods ---------------------------------------------------

    @classmethod
    def for_conservative(cls) -> RobustConfig:
        """Conservative robust portfolio (κ=2.0).

        Suitable when estimation uncertainty is high (short history,
        non-stationary markets).
        """
        return cls(kappa=2.0)

    @classmethod
    def for_moderate(cls) -> RobustConfig:
        """Moderate robust portfolio (κ=1.0).

        Balanced trade-off between estimation uncertainty and expected
        return.
        """
        return cls(kappa=1.0)

    @classmethod
    def for_aggressive(cls) -> RobustConfig:
        """Mildly robust portfolio (κ=0.5).

        Closer to the standard mean-risk solution; retains most of the
        in-sample expected return while still penalising extreme
        concentration.
        """
        return cls(kappa=0.5)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_robust_mean_risk(
    config: RobustConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> MeanRisk:
    """Build a robust :class:`MeanRisk` optimiser with ellipsoidal uncertainty sets.

    Injects skfolio's ``mu_uncertainty_set_estimator`` into a standard
    ``MeanRisk`` to hedge against estimation error in the expected return
    vector μ.  Optionally adds covariance uncertainty as well.

    Kappa–confidence_level mapping
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    At fit time, ``n_assets`` is known and the conversion is::

        confidence_level = F_{χ²_{n_assets}}(κ²)

    This ensures the ellipsoid half-width equals κ regardless of n_assets.

    Special case ``kappa=0``
    ~~~~~~~~~~~~~~~~~~~~~~~~
    When ``kappa=0``, no uncertainty set is injected, so the optimiser is
    identical to the output of :func:`build_mean_risk` with the same
    ``mean_risk_config``.

    Parameters
    ----------
    config : RobustConfig or None
        Robust configuration.  Defaults to ``RobustConfig()`` (κ=1.0,
        min-variance objective, no covariance uncertainty).
    prior_estimator : BasePrior or None
        Prior estimator.  When ``None``, one is built from
        ``config.mean_risk_config.prior_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the :class:`MeanRisk`
        constructor (e.g. ``previous_weights``, ``groups``,
        ``linear_constraints``).

    Returns
    -------
    MeanRisk
        A fitted-ready skfolio optimiser with ellipsoidal uncertainty sets.
    """
    if config is None:
        config = RobustConfig()

    mean_risk_cfg = config.mean_risk_config or MeanRiskConfig()

    if prior_estimator is None and mean_risk_cfg.prior_config is not None:
        prior_estimator = build_prior(mean_risk_cfg.prior_config)

    # kappa=0 → no uncertainty set → identical to standard MeanRisk
    mu_uncertainty: EmpiricalMuUncertaintySet | None = (
        _KappaEmpiricalMuUncertaintySet(kappa=config.kappa)
        if config.kappa > 0.0
        else None
    )

    cov_uncertainty: EmpiricalCovarianceUncertaintySet | None = (
        EmpiricalCovarianceUncertaintySet(confidence_level=0.95)
        if config.cov_uncertainty
        else None
    )

    return MeanRisk(
        objective_function=_OBJECTIVE_MAP[mean_risk_cfg.objective],
        risk_measure=_RISK_MEASURE_MAP[mean_risk_cfg.risk_measure],
        risk_aversion=mean_risk_cfg.risk_aversion,
        efficient_frontier_size=mean_risk_cfg.efficient_frontier_size,
        prior_estimator=prior_estimator,
        mu_uncertainty_set_estimator=mu_uncertainty,
        covariance_uncertainty_set_estimator=cov_uncertainty,
        min_weights=mean_risk_cfg.min_weights,
        max_weights=mean_risk_cfg.max_weights,
        budget=mean_risk_cfg.budget,
        max_short=mean_risk_cfg.max_short,
        max_long=mean_risk_cfg.max_long,
        cardinality=mean_risk_cfg.cardinality,
        transaction_costs=mean_risk_cfg.transaction_costs,
        management_fees=mean_risk_cfg.management_fees,
        max_tracking_error=mean_risk_cfg.max_tracking_error,
        l1_coef=mean_risk_cfg.l1_coef,
        l2_coef=mean_risk_cfg.l2_coef,
        risk_free_rate=mean_risk_cfg.risk_free_rate,
        cvar_beta=mean_risk_cfg.cvar_beta,
        evar_beta=mean_risk_cfg.evar_beta,
        cdar_beta=mean_risk_cfg.cdar_beta,
        edar_beta=mean_risk_cfg.edar_beta,
        solver=mean_risk_cfg.solver,
        solver_params=mean_risk_cfg.solver_params,
        **kwargs,
    )
