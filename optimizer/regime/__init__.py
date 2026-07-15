"""Statistical (Gaussian HMM) regime detection.

Produces filtered (causal) regime probabilities suitable for feeding into
:func:`optimizer.optimization.build_regime_blended_mean_risk` via its
``regime_probabilities`` argument. Architecturally independent from the
macro-indicator regime system in ``optimizer.factors``.
"""

from optimizer.regime._config import (
    HMMCovarianceType,
    HMMFeatureType,
    HMMRegimeConfig,
)
from optimizer.regime._hmm import fit_hmm_regime_probabilities

__all__ = [
    "HMMCovarianceType",
    "HMMFeatureType",
    "HMMRegimeConfig",
    "fit_hmm_regime_probabilities",
]
