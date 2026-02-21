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

Covariance uncertainty uses a stationary block bootstrap to construct an
ellipsoidal uncertainty set around Σ̂.  The standalone utility
:func:`bootstrap_covariance_uncertainty` additionally reports the
Frobenius-norm confidence radius::

    δ = quantile_{1-α}  ‖Σ_b - Σ̂‖_F ,  b = 1…B

Usage example::

    from optimizer.optimization._robust import (
        RobustConfig,
        build_robust_mean_risk,
        bootstrap_covariance_uncertainty,
    )

    # Conservative: κ=2.0, min-variance objective
    model = build_robust_mean_risk(RobustConfig.for_conservative())
    model.fit(X)
    portfolio = model.predict(X)

    # κ=0 → identical to standard MeanRisk (no penalty)
    baseline = build_robust_mean_risk(RobustConfig(kappa=0.0))

    # Bootstrap covariance uncertainty with stationary block bootstrap
    result = bootstrap_covariance_uncertainty(returns, B=500, block_size=21)
    print(result.delta)         # Frobenius-norm confidence radius
    print(result.cov_samples)   # (500, n, n) bootstrap covariance samples
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from scipy.stats import chi2
from skfolio.optimization import MeanRisk
from skfolio.prior._base import BasePrior
from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)

from optimizer.exceptions import ConfigurationError
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import MeanRiskConfig
from optimizer.optimization._factory import _OBJECTIVE_MAP, _RISK_MEASURE_MAP

# ---------------------------------------------------------------------------
# Bootstrap covariance uncertainty
# ---------------------------------------------------------------------------


@dataclass
class CovarianceUncertaintyResult:
    """Result of bootstrap covariance uncertainty estimation.

    Attributes
    ----------
    cov_hat : np.ndarray, shape (n_assets, n_assets)
        Sample covariance matrix estimated from the full return series.
    delta : float
        Frobenius-norm confidence radius: the ``(1-alpha)`` quantile of
        ``‖Σ_b - Σ̂‖_F`` across B stationary bootstrap samples.
    cov_samples : np.ndarray, shape (B, n_assets, n_assets)
        Bootstrap covariance estimates, one per bootstrap resample.
    """

    cov_hat: np.ndarray
    delta: float
    cov_samples: np.ndarray


def bootstrap_covariance_uncertainty(
    returns: pd.DataFrame,
    B: int = 500,
    block_size: int = 21,
    alpha: float = 0.05,
    seed: int | None = None,
) -> CovarianceUncertaintyResult:
    """Estimate covariance uncertainty via stationary block bootstrap.

    Draws *B* bootstrap resamples from *returns* using a stationary block
    bootstrap (preserving autocorrelation structure), estimates the sample
    covariance for each resample, and reports the Frobenius-norm confidence
    radius as the ``(1-alpha)`` quantile of ``‖Σ_b - Σ̂‖_F``.

    The Frobenius-norm confidence set is::

        { Σ : ‖Σ - Σ̂‖_F ≤ δ }

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, n_assets)
        Linear asset returns (rows = observations, columns = assets).
    B : int, default=500
        Number of bootstrap resamples.
    block_size : int, default=21
        Expected block length for the stationary bootstrap (≈ 1 trading
        month).  Larger values preserve more autocorrelation at the cost of
        bootstrap variance.
    alpha : float, default=0.05
        Significance level; ``delta`` is the ``(1-alpha)`` quantile of
        Frobenius distances.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    CovarianceUncertaintyResult
        Dataclass with ``cov_hat``, ``delta``, and ``cov_samples``.
    """
    X = returns.to_numpy()
    n = X.shape[1]
    cov_hat = np.cov(X.T)

    bs = StationaryBootstrap(block_size, X, seed=seed)
    cov_samples = np.empty((B, n, n))
    for i, (pos, _) in enumerate(bs.bootstrap(B)):
        cov_samples[i] = np.cov(pos[0].T)

    frob_distances = np.linalg.norm((cov_samples - cov_hat).reshape(B, -1), axis=1)
    delta = float(np.quantile(frob_distances, 1.0 - alpha))

    return CovarianceUncertaintyResult(
        cov_hat=cov_hat,
        delta=delta,
        cov_samples=cov_samples,
    )


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

    When ``cov_uncertainty=True`` and ``cov_uncertainty_method="bootstrap"``
    a stationary block bootstrap is used (via ``arch``) to construct a
    skfolio ``BootstrapCovarianceUncertaintySet``.  The ``"empirical"``
    method falls back to ``EmpiricalCovarianceUncertaintySet``.

    Parameters
    ----------
    kappa : float
        Ellipsoidal uncertainty radius for μ.  Larger κ produces more
        conservative (diversified) portfolios.  ``kappa=0`` recovers the
        standard (non-robust) MeanRisk exactly.
    cov_uncertainty : bool
        If True, also apply a covariance uncertainty set to hedge against
        covariance estimation error.  The method is controlled by
        ``cov_uncertainty_method``.
    cov_uncertainty_method : str
        Method for covariance uncertainty set construction.
        ``"bootstrap"`` (default) uses stationary block bootstrap via
        ``BootstrapCovarianceUncertaintySet``; ``"empirical"`` uses the
        formula-based ``EmpiricalCovarianceUncertaintySet``.
    B : int
        Number of bootstrap resamples for the stationary block bootstrap
        (only used when ``cov_uncertainty_method="bootstrap"``).
    block_size : int
        Expected block length for the stationary bootstrap (≈ 1 trading
        month).  Only used when ``cov_uncertainty_method="bootstrap"``.
    bootstrap_alpha : float
        Significance level for the covariance uncertainty ellipsoid; the
        ``BootstrapCovarianceUncertaintySet`` confidence level is set to
        ``1 - bootstrap_alpha``.  Only used when
        ``cov_uncertainty_method="bootstrap"``.
    mean_risk_config : MeanRiskConfig or None
        Embedded mean-risk configuration (objective, risk measure,
        weight bounds, …).  ``None`` uses ``MeanRiskConfig()`` defaults
        (minimum-variance, long-only, fully invested).
    """

    kappa: float = 1.0
    cov_uncertainty: bool = False
    cov_uncertainty_method: str = "bootstrap"
    B: int = 500
    block_size: int = 21
    bootstrap_alpha: float = 0.05
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

    @classmethod
    def for_bootstrap_covariance(cls) -> RobustConfig:
        """Robust portfolio with bootstrap covariance uncertainty (κ=1.0).

        Combines mean-vector robustness (κ=1.0) with stationary block
        bootstrap covariance uncertainty to hedge against both sources of
        estimation error.
        """
        return cls(kappa=1.0, cov_uncertainty=True, cov_uncertainty_method="bootstrap")


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
    vector μ.  Optionally adds covariance uncertainty via stationary block
    bootstrap (``cov_uncertainty_method="bootstrap"``, default) or the
    analytic formula (``cov_uncertainty_method="empirical"``).

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

    if config.kappa < 0:
        raise ValueError(
            f"kappa must be non-negative, got {config.kappa}"
        )

    mean_risk_cfg = config.mean_risk_config or MeanRiskConfig()

    if prior_estimator is None and mean_risk_cfg.prior_config is not None:
        prior_estimator = build_prior(mean_risk_cfg.prior_config)

    # kappa=0 → no uncertainty set → identical to standard MeanRisk
    mu_uncertainty: EmpiricalMuUncertaintySet | None = (
        _KappaEmpiricalMuUncertaintySet(kappa=config.kappa)
        if config.kappa > 0.0
        else None
    )

    cov_uncertainty: (
        BootstrapCovarianceUncertaintySet | EmpiricalCovarianceUncertaintySet | None
    )
    if not config.cov_uncertainty:
        cov_uncertainty = None
    elif config.cov_uncertainty_method == "bootstrap":
        cov_uncertainty = BootstrapCovarianceUncertaintySet(
            n_bootstrap_samples=config.B,
            block_size=float(config.block_size),
            confidence_level=1.0 - config.bootstrap_alpha,
        )
    elif config.cov_uncertainty_method == "empirical":
        cov_uncertainty = EmpiricalCovarianceUncertaintySet(confidence_level=0.95)
    else:
        raise ConfigurationError(
            f"Unknown cov_uncertainty_method {config.cov_uncertainty_method!r}. "
            "Valid options are 'bootstrap' and 'empirical'."
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
