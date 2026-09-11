"""Factory functions for building skfolio moment estimators and priors.

``VarianceEstimator`` instances expose a 1-D ``variance_`` attribute, NOT
the 2-D ``covariance_`` attribute. Not interchangeable with covariance
estimators inside priors that need a full covariance matrix.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skfolio.moments import (
    OAS,
    DenoiseCovariance,
    DetoneCovariance,
    EmpiricalCovariance,
    EmpiricalMu,
    EmpiricalVariance,
    EquilibriumMu,
    EWCovariance,
    EWMu,
    EWVariance,
    GerberCovariance,
    GraphicalLassoCV,
    ImpliedCovariance,
    LedoitWolf,
    RegimeAdjustedEWCovariance,
    RegimeAdjustedEWVariance,
    ShrunkCovariance,
    ShrunkMu,
)
from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.covariance._regime_adjusted_ew_covariance import (
    RegimeAdjustmentMethod,
    RegimeAdjustmentTarget,
)
from skfolio.moments.expected_returns._base import BaseMu
from skfolio.moments.expected_returns._shrunk_mu import ShrunkMuMethods
from skfolio.moments.variance._base import BaseVariance
from skfolio.prior import (
    CharacteristicsFactorModel,
    EmpiricalPrior,
    TimeSeriesFactorModel,
)
from skfolio.prior._base import BasePrior

from optimizer.exceptions import ConfigurationError
from optimizer.moments._config import (
    CovEstimatorType,
    FactorModelType,
    MomentEstimationConfig,
    MuEstimatorType,
    RegimeAdjustmentMethodType,
    RegimeAdjustmentTargetType,
    ShrinkageMethod,
    VarianceEstimatorType,
)

if TYPE_CHECKING:
    from skfolio.factor_exposure import BaseFactorExposure

logger = logging.getLogger(__name__)

_SHRINKAGE_MAP: dict[ShrinkageMethod, ShrunkMuMethods] = {
    ShrinkageMethod.JAMES_STEIN: ShrunkMuMethods.JAMES_STEIN,
    ShrinkageMethod.BAYES_STEIN: ShrunkMuMethods.BAYES_STEIN,
    ShrinkageMethod.BODNAR_OKHRIN: ShrunkMuMethods.BODNAR_OKHRIN,
}

_REGIME_TARGET_MAP: dict[RegimeAdjustmentTargetType, RegimeAdjustmentTarget] = {
    RegimeAdjustmentTargetType.PORTFOLIO: RegimeAdjustmentTarget.PORTFOLIO,
    RegimeAdjustmentTargetType.DIAGONAL: RegimeAdjustmentTarget.DIAGONAL,
    RegimeAdjustmentTargetType.MAHALANOBIS: RegimeAdjustmentTarget.MAHALANOBIS,
}

_REGIME_METHOD_MAP: dict[RegimeAdjustmentMethodType, RegimeAdjustmentMethod] = {
    RegimeAdjustmentMethodType.LOG: RegimeAdjustmentMethod.LOG,
    RegimeAdjustmentMethodType.FIRST_MOMENT: RegimeAdjustmentMethod.FIRST_MOMENT,
    RegimeAdjustmentMethodType.RMS: RegimeAdjustmentMethod.RMS,
}


def build_mu_estimator(config: MomentEstimationConfig) -> BaseMu:
    """Build a skfolio expected return estimator from *config*.

    Parameters
    ----------
    config : MomentEstimationConfig
        Moment estimation configuration.

    Returns
    -------
    BaseMu
        A fitted-ready skfolio expected return estimator.
    """
    match config.mu_estimator:
        case MuEstimatorType.EMPIRICAL:
            return EmpiricalMu()
        case MuEstimatorType.SHRUNK:
            return ShrunkMu(method=_SHRINKAGE_MAP[config.shrinkage_method])
        case MuEstimatorType.EW:
            return EWMu(
                half_life=config.ew_mu_half_life,
                min_observations=config.min_observations,
            )
        case MuEstimatorType.EQUILIBRIUM:
            return EquilibriumMu(risk_aversion=config.risk_aversion)
        case _:
            raise ConfigurationError(
                f"Unsupported mu_estimator: {config.mu_estimator!r}"
            )


def build_cov_estimator(config: MomentEstimationConfig) -> BaseCovariance:
    """Build a skfolio covariance estimator from *config*.

    Parameters
    ----------
    config : MomentEstimationConfig
        Moment estimation configuration.

    Returns
    -------
    BaseCovariance
        A fitted-ready skfolio covariance estimator.
    """
    match config.cov_estimator:
        case CovEstimatorType.EMPIRICAL:
            return EmpiricalCovariance()
        case CovEstimatorType.LEDOIT_WOLF:
            return LedoitWolf()
        case CovEstimatorType.OAS:
            return OAS()
        case CovEstimatorType.SHRUNK:
            return ShrunkCovariance(shrinkage=config.shrunk_cov_shrinkage)
        case CovEstimatorType.EW:
            return EWCovariance(
                half_life=config.ew_cov_half_life,
                min_observations=config.min_observations,
            )
        case CovEstimatorType.GERBER:
            return GerberCovariance(threshold=config.gerber_threshold)
        case CovEstimatorType.GRAPHICAL_LASSO_CV:
            return GraphicalLassoCV()
        case CovEstimatorType.DENOISE:
            return DenoiseCovariance(
                covariance_estimator=EmpiricalCovariance(),
            )
        case CovEstimatorType.DETONE:
            return DetoneCovariance(
                covariance_estimator=EmpiricalCovariance(),
            )
        case CovEstimatorType.IMPLIED:
            # skfolio 1.0 renamed ``annualized_factor`` -> ``annualization_factor``
            # (old name deprecated, removed in 2.0). Always use the new name.
            return ImpliedCovariance(
                annualization_factor=config.implied_annualization_factor,
                window_size=config.implied_window_size,
            )
        case CovEstimatorType.REGIME_ADJUSTED_EW:
            return RegimeAdjustedEWCovariance(
                half_life=config.variance_half_life,
                corr_half_life=config.corr_half_life,
                hac_lags=config.hac_lags,
                regime_half_life=config.regime_half_life,
                regime_target=_REGIME_TARGET_MAP[config.regime_target],
                regime_method=_REGIME_METHOD_MAP[config.regime_method],
                regime_multiplier_clip=config.regime_multiplier_clip,
                min_observations=config.min_observations,
            )
        case _:
            raise ConfigurationError(
                f"Unsupported cov_estimator: {config.cov_estimator!r}"
            )


def build_variance_estimator(config: MomentEstimationConfig) -> BaseVariance:
    """Build a skfolio variance estimator from *config*.

    Parameters
    ----------
    config : MomentEstimationConfig
        Moment estimation configuration. ``config.variance_estimator``
        must be set.

    Returns
    -------
    BaseVariance
        A fitted-ready 1-D variance estimator. Exposes ``variance_`` of
        shape ``(n_assets,)`` after ``.fit(X)``.

    Raises
    ------
    ConfigurationError
        If ``config.variance_estimator`` is ``None``.
    """
    if config.variance_estimator is None:
        raise ConfigurationError(
            "variance_estimator must be set on MomentEstimationConfig"
        )
    match config.variance_estimator:
        case VarianceEstimatorType.EMPIRICAL:
            return EmpiricalVariance()
        case VarianceEstimatorType.EW:
            return EWVariance(
                half_life=config.variance_half_life,
                min_observations=config.min_observations,
            )
        case VarianceEstimatorType.REGIME_ADJUSTED_EW:
            return RegimeAdjustedEWVariance(
                half_life=config.variance_half_life,
                hac_lags=config.hac_lags,
                regime_half_life=config.regime_half_life,
                regime_method=_REGIME_METHOD_MAP[config.regime_method],
                regime_multiplier_clip=config.regime_multiplier_clip,
                min_observations=config.min_observations,
            )
        case _:
            raise ConfigurationError(
                f"Unsupported variance_estimator: {config.variance_estimator!r}"
            )


def build_prior(config: MomentEstimationConfig | None = None) -> BasePrior:
    """Build a complete prior estimator from *config*.

    Composes expected return and covariance estimators into an
    ``EmpiricalPrior``, optionally wrapping it in a ``TimeSeriesFactorModel``
    when ``config.use_factor_model`` is ``True`` and
    ``config.factor_model_type`` is ``TIME_SERIES``.

    Parameters
    ----------
    config : MomentEstimationConfig or None
        Moment estimation configuration.  Defaults to
        ``MomentEstimationConfig()`` (EmpiricalMu + LedoitWolf).

    Returns
    -------
    BasePrior
        A fitted-ready skfolio prior estimator.

    Raises
    ------
    ConfigurationError
        If ``config.use_factor_model`` is ``True`` and
        ``config.factor_model_type`` is ``CHARACTERISTICS`` — a
        ``CharacteristicsFactorModel`` needs an ``AssetPanel`` and
        factor-exposure estimators, so build it with
        :func:`build_characteristics_factor_model` instead.
    """
    if config is None:
        config = MomentEstimationConfig()

    mu = build_mu_estimator(config)
    cov = build_cov_estimator(config)

    empirical_prior = EmpiricalPrior(
        mu_estimator=mu,
        covariance_estimator=cov,
        is_log_normal=config.is_log_normal,
        investment_horizon=config.investment_horizon,
    )

    if config.use_factor_model:
        if config.factor_model_type is FactorModelType.CHARACTERISTICS:
            raise ConfigurationError(
                "CharacteristicsFactorModel cannot be built from config alone; "
                "use build_characteristics_factor_model(config, factors=...) "
                "with an AssetPanel and factor-exposure estimators."
            )
        return TimeSeriesFactorModel(
            factor_prior_estimator=empirical_prior,
        )

    return empirical_prior


def build_characteristics_factor_model(
    config: MomentEstimationConfig | None = None,
    *,
    factors: list[tuple[str, BaseFactorExposure]],
    currency_factor: BaseFactorExposure | None = None,
    neutralize_against: dict[str, list[str]] | None = None,
) -> CharacteristicsFactorModel:
    """Build a cross-sectional (BARRA-style) characteristics factor model.

    New in skfolio 1.0.  ``CharacteristicsFactorModel`` estimates asset
    exposures cross-sectionally from fundamental/price *descriptors* carried
    on an :class:`skfolio.containers.AssetPanel` (fit with
    ``characteristics=``), rather than by regressing on observed factor
    return series (that is ``TimeSeriesFactorModel``).

    The ``factors`` list and ``currency_factor`` hold
    ``BaseFactorExposure`` estimator instances (and ``neutralize_against``
    references live factor names), so they are non-serialisable and are
    passed as keyword arguments rather than living on the frozen config.
    Serialisable knobs (``exposure_lag``, ``min_regression_assets``) and the
    factor prior (mu/covariance estimators) are read from *config*.

    Parameters
    ----------
    config : MomentEstimationConfig or None
        Moment estimation configuration.  Defaults to
        ``MomentEstimationConfig()``.  Its mu/covariance estimators become
        the model's ``factor_prior_estimator``; ``exposure_lag`` and
        ``min_regression_assets`` are forwarded.
    factors : list[tuple[str, BaseFactorExposure]]
        Named factor-exposure estimators (keyword-only, required), e.g.
        ``[("market", GlobalFactor()), ("value", FixedWeightedFactor(...))]``.
    currency_factor : BaseFactorExposure or None
        Optional currency factor-exposure estimator.
    neutralize_against : dict[str, list[str]] or None
        Optional map of factor name -> factors to neutralise it against
        (e.g. ``{"non_linear_size": ["size"]}``).

    Returns
    -------
    CharacteristicsFactorModel
        A fitted-ready cross-sectional factor model.
    """
    if config is None:
        config = MomentEstimationConfig()

    factor_prior = EmpiricalPrior(
        mu_estimator=build_mu_estimator(config),
        covariance_estimator=build_cov_estimator(config),
    )

    return CharacteristicsFactorModel(
        factors=list(factors),
        currency_factor=currency_factor,
        neutralize_against=neutralize_against,
        exposure_lag=config.exposure_lag,
        min_regression_assets=config.min_regression_assets,
        factor_prior_estimator=factor_prior,
    )
