"""Factories for skfolio uncertainty-set estimators."""

from __future__ import annotations

from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)
from skfolio.uncertainty_set._base import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
)

from optimizer.uncertainty_set._config import (
    CovarianceUncertaintySetConfig,
    CovarianceUncertaintySetType,
    MuUncertaintySetConfig,
    MuUncertaintySetType,
)


def build_mu_uncertainty_set(
    config: MuUncertaintySetConfig,
) -> BaseMuUncertaintySet:
    """Build a skfolio mu uncertainty-set estimator from *config*."""
    if config.kind == MuUncertaintySetType.EMPIRICAL:
        return EmpiricalMuUncertaintySet(confidence_level=config.confidence_level)
    return BootstrapMuUncertaintySet(
        confidence_level=config.confidence_level,
        n_bootstrap_samples=config.n_bootstrap_samples,
        block_size=config.block_size,
        seed=config.random_state,
    )


def build_covariance_uncertainty_set(
    config: CovarianceUncertaintySetConfig,
) -> BaseCovarianceUncertaintySet:
    """Build a skfolio covariance uncertainty-set estimator from *config*."""
    if config.kind == CovarianceUncertaintySetType.EMPIRICAL:
        return EmpiricalCovarianceUncertaintySet(
            confidence_level=config.confidence_level
        )
    return BootstrapCovarianceUncertaintySet(
        confidence_level=config.confidence_level,
        n_bootstrap_samples=config.n_bootstrap_samples,
        block_size=config.block_size,
        seed=config.random_state,
    )
