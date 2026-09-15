"""Uncertainty-set estimators for robust portfolio optimization.

Three variants per side (mu / covariance):

* empirical closed-form confidence sets,
* stationary-bootstrap confidence sets (skfolio's native pure-numpy
  ``stationary_bootstrap`` — no ``arch`` dependency; block size defaults
  to a Politis-White rule of thumb when ``block_size=None``), and
* orthogonal factor-model sets (new in skfolio 1.0), which require a
  factor-model return distribution at fit time.
"""

from optimizer.uncertainty_set._config import (
    CovarianceUncertaintySetConfig,
    CovarianceUncertaintySetType,
    CrossSectionalWeighting,
    MuUncertaintySetConfig,
    MuUncertaintySetType,
    OrthogonalUncertaintyShape,
)
from optimizer.uncertainty_set._factory import (
    build_covariance_uncertainty_set,
    build_mu_uncertainty_set,
)

__all__ = [
    "CovarianceUncertaintySetConfig",
    "CovarianceUncertaintySetType",
    "CrossSectionalWeighting",
    "MuUncertaintySetConfig",
    "MuUncertaintySetType",
    "OrthogonalUncertaintyShape",
    "build_covariance_uncertainty_set",
    "build_mu_uncertainty_set",
]
