"""Moment estimation and prior construction."""

from optimizer.moments._config import (
    CovEstimatorType,
    MomentEstimationConfig,
    MuEstimatorType,
    ShrinkageMethod,
)
from optimizer.moments._factory import (
    build_cov_estimator,
    build_mu_estimator,
    build_prior,
)
from optimizer.moments._hmm import (
    HMMConfig,
    HMMResult,
    blend_moments_by_regime,
    fit_hmm,
)
from optimizer.moments._scaling import (
    apply_lognormal_correction,
    scale_moments_to_horizon,
)

try:
    from optimizer.moments._dmm import (
        DMMConfig,
        DMMResult,
        blend_moments_dmm,
        fit_dmm,
    )
except ImportError:
    pass

__all__ = [
    "CovEstimatorType",
    "MomentEstimationConfig",
    "MuEstimatorType",
    "ShrinkageMethod",
    "build_cov_estimator",
    "build_mu_estimator",
    "build_prior",
    "apply_lognormal_correction",
    "scale_moments_to_horizon",
    "HMMConfig",
    "HMMResult",
    "fit_hmm",
    "blend_moments_by_regime",
    "DMMConfig",
    "DMMResult",
    "fit_dmm",
    "blend_moments_dmm",
]
