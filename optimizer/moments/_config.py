"""Configuration for moment estimation and prior construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MuEstimatorType(str, Enum):
    """Expected return estimator selection."""

    EMPIRICAL = "empirical"
    SHRUNK = "shrunk"
    EW = "ew"
    EQUILIBRIUM = "equilibrium"


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
    REGIME_ADJUSTED_EW = "regime_adjusted_ew"


class VarianceEstimatorType(str, Enum):
    """1-D variance estimator selection.

    ``VarianceEstimator`` instances expose a 1-D ``variance_`` attribute,
    NOT the 2-D ``covariance_`` attribute. They are not interchangeable
    with covariance estimators inside priors that require a full
    covariance matrix.
    """

    EMPIRICAL = "empirical"
    EW = "ew"
    REGIME_ADJUSTED_EW = "regime_adjusted_ew"


class RegimeAdjustmentTargetType(str, Enum):
    """Target structure for the regime-adjustment STVU multiplier."""

    PORTFOLIO = "portfolio"
    DIAGONAL = "diagonal"
    MAHALANOBIS = "mahalanobis"


class RegimeAdjustmentMethodType(str, Enum):
    """Regime-adjustment scaling method."""

    LOG = "log"
    FIRST_MOMENT = "first_moment"
    RMS = "rms"


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
        If ``True``, wrap the prior in a ``TimeSeriesFactorModel``.
    residual_variance : bool
        Whether to include residual variance in ``TimeSeriesFactorModel``.
    variance_estimator : VarianceEstimatorType or None
        Which 1-D variance estimator to build via
        :func:`build_variance_estimator`. Independent of ``cov_estimator``.
    variance_half_life : float
        Half-life forwarded to ``EWVariance``, ``RegimeAdjustedEWVariance``
        and ``RegimeAdjustedEWCovariance`` ``half_life`` argument.
    corr_half_life : float or None
        Correlation half-life forwarded to ``RegimeAdjustedEWCovariance``.
    regime_half_life : float or None
        Half-life of the regime detector forwarded to
        ``RegimeAdjustedEW*`` ``regime_half_life`` argument.
    regime_target : RegimeAdjustmentTargetType
        STVU target structure for ``RegimeAdjustedEWCovariance``.
    regime_method : RegimeAdjustmentMethodType
        STVU scaling method.
    regime_multiplier_clip : tuple[float, float]
        Lower/upper bounds for the STVU multiplier (skfolio default
        ``(0.7, 1.6)``).
    hac_lags : int
        Newey-West HAC lag count for the regime detector.
    min_observations : int or None
        Minimum number of observations required by EW-family
        estimators before producing output.
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

    # -- Factor model --
    use_factor_model: bool = False
    residual_variance: bool = True

    # -- Variance estimator + regime adjustment --
    variance_estimator: VarianceEstimatorType | None = None
    variance_half_life: float = 40.0
    corr_half_life: float | None = None
    regime_half_life: float | None = None
    regime_target: RegimeAdjustmentTargetType = RegimeAdjustmentTargetType.PORTFOLIO
    regime_method: RegimeAdjustmentMethodType = RegimeAdjustmentMethodType.FIRST_MOMENT
    regime_multiplier_clip: tuple[float, float] = (0.7, 1.6)
    hac_lags: int = 5
    min_observations: int | None = None

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
    def for_regime_adjusted_ew(cls) -> MomentEstimationConfig:
        """Regime-adjusted EW prior with STVU multiplier."""
        return cls(
            cov_estimator=CovEstimatorType.REGIME_ADJUSTED_EW,
            variance_estimator=VarianceEstimatorType.REGIME_ADJUSTED_EW,
            variance_half_life=23.0,
            corr_half_life=50.0,
        )
