"""Factories for skfolio uncertainty-set estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
    OrthogonalCovarianceUncertaintySet,
    OrthogonalMuUncertaintySet,
)
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
)
from skfolio.uncertainty_set._orthogonal import CSWeighting

from optimizer.exceptions import ConfigurationError
from optimizer.uncertainty_set._config import (
    CovarianceUncertaintySetConfig,
    CovarianceUncertaintySetType,
    MuUncertaintySetConfig,
    MuUncertaintySetType,
)

if TYPE_CHECKING:
    from skfolio.prior import BasePrior


def build_mu_uncertainty_set(
    config: MuUncertaintySetConfig,
    *,
    prior_estimator: BasePrior | None = None,
) -> BaseMuUncertaintySet:
    """Build a skfolio mu uncertainty-set estimator from *config*.

    Parameters
    ----------
    config : MuUncertaintySetConfig
        Serialisable configuration selecting the variant and its knobs.
    prior_estimator : BasePrior or None
        Non-serialisable prior estimator (e.g. a factor model or a
        shrinkage prior) forwarded to the empirical/bootstrap sets. Not
        supported for the orthogonal set, which reads its factor model from
        the ``return_distribution`` supplied at fit time.
    """
    if config.kind == MuUncertaintySetType.EMPIRICAL:
        return EmpiricalMuUncertaintySet(
            prior_estimator=prior_estimator,
            confidence_level=config.confidence_level,
            diagonal=config.diagonal,
            n_eff=config.n_eff,
        )
    if config.kind == MuUncertaintySetType.BOOTSTRAP:
        return BootstrapMuUncertaintySet(
            prior_estimator=prior_estimator,
            confidence_level=config.confidence_level,
            diagonal=config.diagonal,
            n_bootstrap_samples=config.n_bootstrap_samples,
            block_size=config.block_size,
            seed=config.random_state,
        )
    if prior_estimator is not None:
        raise ConfigurationError(
            "prior_estimator is not supported for ORTHOGONAL uncertainty sets; "
            "supply a factor-model return_distribution at fit time"
        )
    return OrthogonalMuUncertaintySet(
        confidence_level=config.confidence_level,
        cs_weighting=CSWeighting(config.cs_weighting.value),
        uncertainty_shape=config.uncertainty_shape.value,
    )


def build_covariance_uncertainty_set(
    config: CovarianceUncertaintySetConfig,
    *,
    prior_estimator: BasePrior | None = None,
) -> BaseCovarianceUncertaintySet:
    """Build a skfolio covariance uncertainty-set estimator from *config*.

    Parameters
    ----------
    config : CovarianceUncertaintySetConfig
        Serialisable configuration selecting the variant and its knobs.
    prior_estimator : BasePrior or None
        Non-serialisable prior estimator forwarded to the
        empirical/bootstrap sets. Not supported for the orthogonal set.
    """
    if config.kind == CovarianceUncertaintySetType.EMPIRICAL:
        return EmpiricalCovarianceUncertaintySet(
            prior_estimator=prior_estimator,
            confidence_level=config.confidence_level,
            diagonal=config.diagonal,
            n_eff=config.n_eff,
        )
    if config.kind == CovarianceUncertaintySetType.BOOTSTRAP:
        return BootstrapCovarianceUncertaintySet(
            prior_estimator=prior_estimator,
            confidence_level=config.confidence_level,
            diagonal=config.diagonal,
            n_bootstrap_samples=config.n_bootstrap_samples,
            block_size=config.block_size,
            seed=config.random_state,
        )
    if prior_estimator is not None:
        raise ConfigurationError(
            "prior_estimator is not supported for ORTHOGONAL uncertainty sets; "
            "supply a factor-model return_distribution at fit time"
        )
    return OrthogonalCovarianceUncertaintySet(
        radius=config.radius,
        cs_weighting=CSWeighting(config.cs_weighting.value),
    )
