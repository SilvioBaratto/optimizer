"""Uncertainty-set estimators for robust portfolio optimization.

Bootstrap uncertainty sets call ``arch.StationaryBootstrap`` under the
hood — block size defaults to a Politis-White rule of thumb when
``block_size=None``.
"""

from optimizer.uncertainty_set._config import (
    CovarianceUncertaintySetConfig,
    CovarianceUncertaintySetType,
    MuUncertaintySetConfig,
    MuUncertaintySetType,
)
from optimizer.uncertainty_set._factory import (
    build_covariance_uncertainty_set,
    build_mu_uncertainty_set,
)

__all__ = [
    "CovarianceUncertaintySetConfig",
    "CovarianceUncertaintySetType",
    "MuUncertaintySetConfig",
    "MuUncertaintySetType",
    "build_covariance_uncertainty_set",
    "build_mu_uncertainty_set",
]
