"""Moment estimation and prior construction.

``VarianceEstimator`` instances expose a 1-D ``variance_`` attribute, NOT
the 2-D ``covariance_`` attribute. Not interchangeable with covariance
estimators inside priors that need a full covariance matrix.
"""

from optimizer.moments._config import (
    CovEstimatorType,
    MomentEstimationConfig,
    MuEstimatorType,
    RegimeAdjustmentMethodType,
    RegimeAdjustmentTargetType,
    ShrinkageMethod,
    VarianceEstimatorType,
)
from optimizer.moments._factory import (
    build_cov_estimator,
    build_mu_estimator,
    build_prior,
    build_variance_estimator,
)
from optimizer.moments._scaling import (
    apply_lognormal_correction,
    scale_moments_to_horizon,
)

__all__ = [
    "CovEstimatorType",
    "MomentEstimationConfig",
    "MuEstimatorType",
    "RegimeAdjustmentMethodType",
    "RegimeAdjustmentTargetType",
    "ShrinkageMethod",
    "VarianceEstimatorType",
    "apply_lognormal_correction",
    "build_cov_estimator",
    "build_mu_estimator",
    "build_prior",
    "build_variance_estimator",
    "scale_moments_to_horizon",
]
