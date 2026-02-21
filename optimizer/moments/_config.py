"""Configuration for moment estimation and prior construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from optimizer.moments._hmm import HMMConfig


class MuEstimatorType(str, Enum):
    """Expected return estimator selection."""

    EMPIRICAL = "empirical"
    SHRUNK = "shrunk"
    EW = "ew"
    EQUILIBRIUM = "equilibrium"
    HMM_BLENDED = "hmm_blended"


class CovEstimatorType(str, Enum):
    """Covariance estimator selection."""

    EMPIRICAL = "empirical"
    LEDOIT_WOLF = "ledoit_wolf"
    OAS = "oas"
    SHRUNK = "shrunk"
    EW = "ew"
    GERBER = "gerber"
    GRAPHICAL_LASSO_CV = "graphical_lasso_cv"
    DENOISE = "denoise"
    DETONE = "detone"
    IMPLIED = "implied"
    HMM_BLENDED = "hmm_blended"


class ShrinkageMethod(str, Enum):
    """Shrinkage method for :class:`ShrunkMu`.

    Maps to :class:`skfolio.moments.expected_returns._shrunk_mu.ShrunkMuMethods`.
    """

    JAMES_STEIN = "james_stein"
    BAYES_STEIN = "bayes_stein"
    BODNAR_OKHRIN = "bodnar_okhrin"


@dataclass(frozen=True)
class MomentEstimationConfig:
    """Immutable configuration for moment estimation and prior construction.

    All parameters map 1:1 to skfolio estimator constructor arguments,
    making the config serialisable and suitable for hyperparameter sweeps.

    Parameters
    ----------
    mu_estimator : MuEstimatorType
        Which expected return estimator to use.
    shrinkage_method : ShrinkageMethod
        Shrinkage flavour when ``mu_estimator`` is ``SHRUNK``.
    ew_mu_alpha : float
        Exponential weighting decay for ``EWMu``.
    risk_aversion : float
        Risk-aversion coefficient for ``EquilibriumMu``.
    cov_estimator : CovEstimatorType
        Which covariance estimator to use.
    ew_cov_alpha : float
        Exponential weighting decay for ``EWCovariance``.
    shrunk_cov_shrinkage : float
        Shrinkage intensity for ``ShrunkCovariance``.
    gerber_threshold : float
        Threshold for ``GerberCovariance``.
    is_log_normal : bool
        Whether returns are log-normal (for multi-period scaling in
        ``EmpiricalPrior``).
    investment_horizon : float or None
        Investment horizon forwarded to ``EmpiricalPrior``.
    use_factor_model : bool
        If ``True``, wrap the prior in a ``FactorModel``.
    residual_variance : bool
        Whether to include residual variance in ``FactorModel``.
    """

    # -- Expected return estimator --
    mu_estimator: MuEstimatorType = MuEstimatorType.EMPIRICAL
    shrinkage_method: ShrinkageMethod = ShrinkageMethod.JAMES_STEIN
    ew_mu_alpha: float = 0.2
    risk_aversion: float = 1.0

    # -- Covariance estimator --
    cov_estimator: CovEstimatorType = CovEstimatorType.LEDOIT_WOLF
    ew_cov_alpha: float = 0.2
    shrunk_cov_shrinkage: float = 0.1
    gerber_threshold: float = 0.5

    # -- Prior assembly --
    is_log_normal: bool = False
    investment_horizon: float | None = None

    # -- HMM blended estimators --
    hmm_config: HMMConfig = field(default_factory=HMMConfig)

    # -- Factor model --
    use_factor_model: bool = False
    residual_variance: bool = True

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_equilibrium_ledoitwolf(cls) -> MomentEstimationConfig:
        """Black-Litterman-ready prior: EquilibriumMu + LedoitWolf."""
        return cls(
            mu_estimator=MuEstimatorType.EQUILIBRIUM,
            cov_estimator=CovEstimatorType.LEDOIT_WOLF,
        )

    @classmethod
    def for_shrunk_denoised(cls) -> MomentEstimationConfig:
        """Conservative prior: ShrunkMu (James-Stein) + DenoiseCovariance."""
        return cls(
            mu_estimator=MuEstimatorType.SHRUNK,
            shrinkage_method=ShrinkageMethod.JAMES_STEIN,
            cov_estimator=CovEstimatorType.DENOISE,
        )

    @classmethod
    def for_adaptive(cls) -> MomentEstimationConfig:
        """Responsive prior: EW on both mu and covariance."""
        return cls(
            mu_estimator=MuEstimatorType.EW,
            cov_estimator=CovEstimatorType.EW,
        )

    @classmethod
    def for_hmm_blended(cls, n_states: int = 2) -> MomentEstimationConfig:
        """HMM-blended prior: regime-probability-weighted mu and covariance."""
        return cls(
            mu_estimator=MuEstimatorType.HMM_BLENDED,
            cov_estimator=CovEstimatorType.HMM_BLENDED,
            hmm_config=HMMConfig(n_states=n_states),
        )
