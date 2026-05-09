"""Configuration for distance estimator selection.

The :class:`DistanceConfig` is a serialisable dataclass with a
:class:`DistanceEstimatorType` enum field plus optional MI-only knobs.
Non-MI estimators reject ``n_bins`` and ``bandwidth`` to surface
configuration mistakes early.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.exceptions import ConfigurationError


class DistanceEstimatorType(str, Enum):
    """Distance estimator selection."""

    PEARSON = "pearson"
    KENDALL = "kendall"
    SPEARMAN = "spearman"
    COVARIANCE = "covariance"
    DISTANCE_CORRELATION = "distance_correlation"
    MUTUAL_INFORMATION = "mutual_information"


@dataclass(frozen=True)
class DistanceConfig:
    """Immutable configuration for distance estimator construction.

    Parameters
    ----------
    estimator : DistanceEstimatorType
        Which skfolio distance estimator to instantiate.
    n_bins : int or None
        Histogram bin count for :class:`skfolio.distance.MutualInformation`.
        Only valid when ``estimator`` is ``MUTUAL_INFORMATION``.
    bandwidth : float or None
        Reserved for kernel-based mutual-information estimators. Not
        supported by skfolio 0.20.1 — must be ``None``.
    """

    estimator: DistanceEstimatorType = DistanceEstimatorType.PEARSON
    n_bins: int | None = None
    bandwidth: float | None = None

    def __post_init__(self) -> None:
        is_mi = self.estimator == DistanceEstimatorType.MUTUAL_INFORMATION
        if not is_mi and (self.n_bins is not None or self.bandwidth is not None):
            raise ConfigurationError(
                "n_bins and bandwidth are only valid for "
                "DistanceEstimatorType.MUTUAL_INFORMATION"
            )
        if self.bandwidth is not None:
            raise ConfigurationError(
                "bandwidth is reserved and not supported by skfolio 0.20.1 "
                "MutualInformation"
            )

    @classmethod
    def for_pearson(cls) -> DistanceConfig:
        """Pearson correlation distance preset."""
        return cls(estimator=DistanceEstimatorType.PEARSON)

    @classmethod
    def for_kendall(cls) -> DistanceConfig:
        """Kendall rank correlation distance preset."""
        return cls(estimator=DistanceEstimatorType.KENDALL)

    @classmethod
    def for_spearman(cls) -> DistanceConfig:
        """Spearman rank correlation distance preset."""
        return cls(estimator=DistanceEstimatorType.SPEARMAN)

    @classmethod
    def for_covariance(cls) -> DistanceConfig:
        """Covariance-induced distance preset."""
        return cls(estimator=DistanceEstimatorType.COVARIANCE)

    @classmethod
    def for_distance_correlation(cls) -> DistanceConfig:
        """Szekely-Rizzo distance correlation preset."""
        return cls(estimator=DistanceEstimatorType.DISTANCE_CORRELATION)

    @classmethod
    def for_mutual_information(cls, n_bins: int = 10) -> DistanceConfig:
        """Mutual-information distance preset with histogram binning."""
        return cls(
            estimator=DistanceEstimatorType.MUTUAL_INFORMATION,
            n_bins=n_bins,
        )
