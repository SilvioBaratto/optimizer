"""Configuration for statistical (Gaussian HMM) regime detection.

The :class:`HMMRegimeConfig` is a serialisable dataclass driving
:func:`optimizer.regime._hmm.fit_hmm_regime_probabilities`. Feature
choice, covariance structure, and EM convergence knobs are all
config fields; the fitted ``hmmlearn.hmm.GaussianHMM`` instance itself
is never stored on the config (non-serialisable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from optimizer.exceptions import ConfigurationError


class HMMFeatureType(str, Enum):
    """Feature(s) the Gaussian HMM is fit on."""

    RETURN = "return"
    RETURN_VOL = "return_vol"
    RETURN_VOL_SKEW = "return_vol_skew"


class HMMCovarianceType(str, Enum):
    """Covariance structure of the per-regime Gaussian emissions.

    Mirrors ``hmmlearn.hmm.GaussianHMM``'s ``covariance_type``.
    """

    FULL = "full"
    DIAG = "diag"
    TIED = "tied"
    SPHERICAL = "spherical"


@dataclass(frozen=True)
class HMMRegimeConfig:
    """Immutable configuration for :func:`fit_hmm_regime_probabilities`.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states. Defaults to 2 (low-vol / high-vol).
    feature : HMMFeatureType
        Which cross-sectional feature(s) of the return panel to fit on.
        ``RETURN`` uses the equal-weighted cross-sectional mean return.
        ``RETURN_VOL`` adds a rolling realised-volatility feature.
        ``RETURN_VOL_SKEW`` adds a rolling skewness feature on top.
    vol_window : int
        Rolling window (trading days) for the volatility feature.
        Ignored when ``feature == HMMFeatureType.RETURN``.
    skew_window : int
        Rolling window (trading days) for the skewness feature.
        Ignored unless ``feature == HMMFeatureType.RETURN_VOL_SKEW``.
    covariance_type : HMMCovarianceType
        Emission covariance structure. Defaults to ``DIAG`` (robust
        with few features / short history).
    n_iter : int
        Maximum Baum-Welch EM iterations.
    tol : float
        EM convergence tolerance on the log-likelihood.
    random_state : int
        Seed for the EM initialisation (k-means init is otherwise
        non-deterministic).
    min_observations : int
        Minimum number of rows required to fit; raises
        :class:`~optimizer.exceptions.ConfigurationError` below this.
    """

    n_regimes: int = 2
    feature: HMMFeatureType = HMMFeatureType.RETURN_VOL
    vol_window: int = 21
    skew_window: int = 63
    covariance_type: HMMCovarianceType = HMMCovarianceType.DIAG
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42
    min_observations: int = 252

    def __post_init__(self) -> None:
        if self.n_regimes < 2:
            raise ConfigurationError(
                f"n_regimes must be >= 2, got {self.n_regimes}."
            )
        if self.vol_window < 2:
            raise ConfigurationError(
                f"vol_window must be >= 2, got {self.vol_window}."
            )
        if self.skew_window < 2:
            raise ConfigurationError(
                f"skew_window must be >= 2, got {self.skew_window}."
            )
        if self.min_observations < 30:
            raise ConfigurationError(
                f"min_observations must be >= 30, got {self.min_observations}."
            )

    @classmethod
    def for_two_regime(cls) -> HMMRegimeConfig:
        """Default 2-regime (calm / stressed) detector on vol-augmented returns."""
        return cls(n_regimes=2, feature=HMMFeatureType.RETURN_VOL)

    @classmethod
    def for_three_regime(cls) -> HMMRegimeConfig:
        """3-regime (calm / transitional / stressed) detector."""
        return cls(n_regimes=3, feature=HMMFeatureType.RETURN_VOL)
