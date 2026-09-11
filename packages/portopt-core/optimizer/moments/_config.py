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


class FactorModelType(str, Enum):
    """Factor-model estimator selection (skfolio 1.0).

    ``TIME_SERIES`` -> :class:`skfolio.prior.TimeSeriesFactorModel`, which
    regresses asset returns on observed factor return series (fit with
    ``factors=``).  ``CHARACTERISTICS`` ->
    :class:`skfolio.prior.CharacteristicsFactorModel`, a cross-sectional
    (BARRA-style) model driven by fundamental/price descriptors over an
    ``AssetPanel`` (fit with ``characteristics=``).

    In skfolio 1.0 ``FactorModel`` itself is the *fitted result container*,
    not an estimator, so neither value maps to it directly.
    """

    TIME_SERIES = "time_series"
    CHARACTERISTICS = "characteristics"


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
    ew_mu_half_life : float
        Exponential-weighting half-life (in observations) for ``EWMu``.
        skfolio 1.0 replaced the former ``alpha`` argument with
        ``half_life``; convert via ``half_life = -1 / log2(1 - alpha)``.
    risk_aversion : float
        Risk-aversion coefficient for ``EquilibriumMu``.
    cov_estimator : CovEstimatorType
        Which covariance estimator to use.
    ew_cov_half_life : float
        Exponential-weighting half-life (in observations) for
        ``EWCovariance``. Replaces the former ``alpha`` (skfolio 1.0).
    shrunk_cov_shrinkage : float
        Shrinkage intensity for ``ShrunkCovariance``.
    gerber_threshold : float
        Threshold for ``GerberCovariance``.
    implied_annualization_factor : float or None
        Number of periods per year forwarded to ``ImpliedCovariance``
        as ``annualization_factor`` (skfolio 1.0 renamed the former
        ``annualized_factor``; the old name is deprecated and removed in
        2.0). ``None`` lets skfolio infer it from the index frequency.
    implied_window_size : int
        Rolling window (in observations) used by ``ImpliedCovariance`` to
        regress realized on implied volatility.
    is_log_normal : bool
        Whether returns are log-normal (for multi-period scaling in
        ``EmpiricalPrior``).
    investment_horizon : float or None
        Investment horizon forwarded to ``EmpiricalPrior``.
    use_factor_model : bool
        If ``True``, wrap the prior in a factor model. When
        ``factor_model_type`` is ``TIME_SERIES`` (the default),
        :func:`build_prior` returns a ``TimeSeriesFactorModel`` fit with
        factor returns via the keyword ``factors=`` (skfolio 1.0; the
        former positional ``y`` factor argument was removed). When it is
        ``CHARACTERISTICS`` the model must be built via
        :func:`build_characteristics_factor_model` (needs an
        ``AssetPanel`` + factor-exposure estimators that cannot live in a
        serialisable config).
    factor_model_type : FactorModelType
        Which factor-model estimator to build when ``use_factor_model`` is
        ``True``.
    exposure_lag : int
        Number of periods to lag descriptor exposures in
        ``CharacteristicsFactorModel`` (guards against look-ahead when
        characteristics are only known after the fact).
    min_regression_assets : int or None
        Minimum cross-sectional assets required per date for the
        ``CharacteristicsFactorModel`` regression. ``None`` uses the
        skfolio default.
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
    # 3.11 ≈ former alpha=0.2 (skfolio 1.0 dropped alpha; half_life = -1/log2(1-alpha))
    ew_mu_half_life: float = 3.11
    risk_aversion: float = 1.0

    # -- Covariance estimator --
    cov_estimator: CovEstimatorType = CovEstimatorType.LEDOIT_WOLF
    ew_cov_half_life: float = 3.11
    shrunk_cov_shrinkage: float = 0.1
    gerber_threshold: float = 0.5
    implied_annualization_factor: float | None = None
    implied_window_size: int = 20

    # -- Prior assembly --
    is_log_normal: bool = False
    investment_horizon: float | None = None

    # -- Factor model --
    use_factor_model: bool = False
    factor_model_type: FactorModelType = FactorModelType.TIME_SERIES
    exposure_lag: int = 1
    min_regression_assets: int | None = None

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
