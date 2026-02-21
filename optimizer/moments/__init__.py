"""Moment estimation and prior construction."""

import contextlib

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
    HMMBlendedCovariance,
    HMMBlendedMu,
    HMMConfig,
    HMMResult,
    blend_moments_by_regime,
    fit_hmm,
)
from optimizer.moments._scaling import (
    apply_lognormal_correction,
    scale_moments_to_horizon,
)

with contextlib.suppress(ImportError):
    from optimizer.moments._dmm import (
        DMMConfig,
        DMMResult,
        blend_moments_dmm,
        fit_dmm,
    )

__all__ = [
    "CovEstimatorType",
    "DMMConfig",
    "DMMResult",
    "HMMBlendedCovariance",
    "HMMBlendedMu",
    "HMMConfig",
    "HMMResult",
    "MomentEstimationConfig",
    "MuEstimatorType",
    "ShrinkageMethod",
    "apply_lognormal_correction",
    "blend_moments_by_regime",
    "blend_moments_dmm",
    "build_cov_estimator",
    "build_mu_estimator",
    "build_prior",
    "fit_dmm",
    "fit_hmm",
    "scale_moments_to_horizon",
]
