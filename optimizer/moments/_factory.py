"""Factory functions for building skfolio moment estimators and priors."""

from __future__ import annotations

import logging

from skfolio.moments import (
    OAS,
    DenoiseCovariance,
    DetoneCovariance,
    EmpiricalCovariance,
    EmpiricalMu,
    EquilibriumMu,
    EWCovariance,
    EWMu,
    GerberCovariance,
    GraphicalLassoCV,
    ImpliedCovariance,
    LedoitWolf,
    ShrunkCovariance,
    ShrunkMu,
)
from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.expected_returns._base import BaseMu
from skfolio.moments.expected_returns._shrunk_mu import ShrunkMuMethods
from skfolio.prior import EmpiricalPrior, FactorModel
from skfolio.prior._base import BasePrior

from optimizer.moments._config import (
    CovEstimatorType,
    MomentEstimationConfig,
    MuEstimatorType,
    ShrinkageMethod,
)
from optimizer.moments._hmm import HMMBlendedCovariance, HMMBlendedMu

logger = logging.getLogger(__name__)

_SHRINKAGE_MAP: dict[ShrinkageMethod, ShrunkMuMethods] = {
    ShrinkageMethod.JAMES_STEIN: ShrunkMuMethods.JAMES_STEIN,
    ShrinkageMethod.BAYES_STEIN: ShrunkMuMethods.BAYES_STEIN,
    ShrinkageMethod.BODNAR_OKHRIN: ShrunkMuMethods.BODNAR_OKHRIN,
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
            return EWMu(alpha=config.ew_mu_alpha)
        case MuEstimatorType.EQUILIBRIUM:
            return EquilibriumMu(risk_aversion=config.risk_aversion)
        case MuEstimatorType.HMM_BLENDED:
            return HMMBlendedMu(hmm_config=config.hmm_config)


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
            return EWCovariance(alpha=config.ew_cov_alpha)
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
            return ImpliedCovariance()
        case CovEstimatorType.HMM_BLENDED:
            return HMMBlendedCovariance(hmm_config=config.hmm_config)


def build_prior(config: MomentEstimationConfig | None = None) -> BasePrior:
    """Build a complete prior estimator from *config*.

    Composes expected return and covariance estimators into an
    ``EmpiricalPrior``, optionally wrapping it in a ``FactorModel``
    when ``config.use_factor_model`` is ``True``.

    Parameters
    ----------
    config : MomentEstimationConfig or None
        Moment estimation configuration.  Defaults to
        ``MomentEstimationConfig()`` (EmpiricalMu + LedoitWolf).

    Returns
    -------
    BasePrior
        A fitted-ready skfolio prior estimator.
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
        return FactorModel(
            factor_prior_estimator=empirical_prior,
            residual_variance=config.residual_variance,
        )

    return empirical_prior
