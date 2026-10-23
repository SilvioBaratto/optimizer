"""Configuration for distance estimator selection.

``DistanceConfig`` is a serialisable dataclass with a
``DistanceEstimatorType`` enum field plus estimator-specific knobs.
Every field holds only primitives / enums so the config round-trips through
serialisation. Non-serialisable objects (e.g. a covariance estimator instance
for ``skfolio.distance.CovarianceDistance``) are passed as factory kwargs,
never stored on the config.

skfolio 1.0.6 distance estimators expose a common codependence-to-distance
transform via ``absolute`` and ``power``:

* ``absolute`` — take the absolute value of the codependence before mapping it
  to a distance (treats strong negative and positive correlation as equally
  "close").
* ``power`` — raise the codependence itself to this power before mapping it to
  a distance. A fractional power of a negative codependence is NaN, so it
  requires ``absolute=True``; an even integer power without ``absolute`` makes
  strong negative and positive correlation equally "close" (squared angular
  distance).

These apply to the correlation / covariance family (Pearson, Kendall, Spearman,
Covariance). ``DistanceCorrelation`` has no tunable knob (its ``threshold`` is
stored by skfolio but never read) and ``MutualInformation`` exposes ``n_bins`` /
``n_bins_method`` / ``normalize``. Setting a knob that does not belong to the
selected estimator, or one that skfolio ignores, raises a ``ConfigurationError``
to surface configuration mistakes early.
"""

from __future__ import annotations

import math
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


class NBinsMethod(str, Enum):
    """Histogram bin-count rule for ``skfolio.distance.MutualInformation``.

    Mirrors ``skfolio.distance.NBinsMethod``. Only consulted when ``n_bins`` is
    ``None`` (an explicit ``n_bins`` overrides the rule).
    """

    FREEDMAN = "freedman"
    KNUTH = "knuth"


# Only these estimators expose the ``absolute``/``power`` codependence-to-distance
# transform.
_CORR_FAMILY = frozenset(
    {
        DistanceEstimatorType.PEARSON,
        DistanceEstimatorType.KENDALL,
        DistanceEstimatorType.SPEARMAN,
        DistanceEstimatorType.COVARIANCE,
    }
)


@dataclass(frozen=True)
class DistanceConfig:
    """Immutable configuration for distance estimator construction.

    Attributes:
        estimator: Which skfolio distance estimator to instantiate.
        absolute: Take the absolute value of the codependence before mapping to a
            distance. Only valid for the correlation / covariance family
            (Pearson, Kendall, Spearman, Covariance).
        power: Exponent applied to the codependence (not to ``1 - codependence``).
            Must be finite and strictly positive, and must be a whole number unless
            ``absolute`` is ``True`` (a fractional power of a negative codependence
            is NaN). Only valid for the correlation / covariance family.
        threshold: Reserved for ``skfolio.distance.DistanceCorrelation``. skfolio
            stores it but never reads it when fitting, so any value would be a
            silent no-op — it must be ``None``. Retained for backward
            compatibility only.
        n_bins: Explicit histogram bin count for ``skfolio.distance.MutualInformation``.
            Only valid when ``estimator`` is ``MUTUAL_INFORMATION``. ``None`` defers to
            ``n_bins_method``.
        n_bins_method: Bin-count rule used when ``n_bins`` is ``None``. Only valid when
            ``estimator`` is ``MUTUAL_INFORMATION``. ``None`` keeps the skfolio default
            (Freedman-Diaconis).
        normalize: Whether ``skfolio.distance.MutualInformation`` normalises the mutual
            information into [0, 1]. Only valid when ``estimator`` is
            ``MUTUAL_INFORMATION``. ``None`` keeps the skfolio default (``True``).
        bandwidth: Reserved for kernel-based mutual-information estimators. Not
            supported by skfolio 1.0.6 — must be ``None``. Retained for backward
            compatibility only.
    """

    estimator: DistanceEstimatorType = DistanceEstimatorType.PEARSON
    absolute: bool = False
    power: float = 1.0
    threshold: float | None = None
    n_bins: int | None = None
    n_bins_method: NBinsMethod | None = None
    normalize: bool | None = None
    bandwidth: float | None = None

    def __post_init__(self) -> None:
        is_mi = self.estimator == DistanceEstimatorType.MUTUAL_INFORMATION
        in_corr_family = self.estimator in _CORR_FAMILY

        # absolute / power belong to the correlation / covariance family only.
        if not in_corr_family and (self.absolute or self.power != 1.0):
            raise ConfigurationError(
                "absolute and power are only valid for the correlation/"
                "covariance distance family (PEARSON, KENDALL, SPEARMAN, "
                "COVARIANCE)"
            )
        if not math.isfinite(self.power) or self.power <= 0.0:
            raise ConfigurationError("power must be finite and strictly positive")
        if not self.absolute and not float(self.power).is_integer():
            raise ConfigurationError(
                "a fractional power requires absolute=True: it is applied to the "
                "signed codependence and is NaN for negative correlations"
            )

        # skfolio's DistanceCorrelation never reads ``threshold``; accepting a
        # value would let callers tune a knob that changes nothing.
        if self.threshold is not None:
            raise ConfigurationError(
                "threshold has no effect: skfolio DistanceCorrelation stores it "
                "but never reads it; leave it as None"
            )

        mi_only_set = (
            self.n_bins is not None
            or self.n_bins_method is not None
            or self.normalize is not None
            or self.bandwidth is not None
        )
        if not is_mi and mi_only_set:
            raise ConfigurationError(
                "n_bins, n_bins_method, normalize and bandwidth are only "
                "valid for DistanceEstimatorType.MUTUAL_INFORMATION"
            )
        if self.n_bins is not None and self.n_bins < 1:
            raise ConfigurationError("n_bins must be a positive integer")
        if self.bandwidth is not None:
            raise ConfigurationError(
                "bandwidth is reserved and not supported by skfolio 1.0.6 "
                "MutualInformation"
            )

    @classmethod
    def for_pearson(
        cls, *, absolute: bool = False, power: float = 1.0
    ) -> DistanceConfig:
        """Pearson correlation distance preset."""
        return cls(
            estimator=DistanceEstimatorType.PEARSON,
            absolute=absolute,
            power=power,
        )

    @classmethod
    def for_kendall(
        cls, *, absolute: bool = False, power: float = 1.0
    ) -> DistanceConfig:
        """Kendall rank correlation distance preset."""
        return cls(
            estimator=DistanceEstimatorType.KENDALL,
            absolute=absolute,
            power=power,
        )

    @classmethod
    def for_spearman(
        cls, *, absolute: bool = False, power: float = 1.0
    ) -> DistanceConfig:
        """Spearman rank correlation distance preset."""
        return cls(
            estimator=DistanceEstimatorType.SPEARMAN,
            absolute=absolute,
            power=power,
        )

    @classmethod
    def for_covariance(
        cls, *, absolute: bool = False, power: float = 1.0
    ) -> DistanceConfig:
        """Covariance-induced distance preset.

        A non-default covariance estimator instance is passed to
        ``optimizer.distance.build_distance`` as a kwarg, not stored here.
        """
        return cls(
            estimator=DistanceEstimatorType.COVARIANCE,
            absolute=absolute,
            power=power,
        )

    @classmethod
    def for_distance_correlation(
        cls, *, threshold: float | None = None
    ) -> DistanceConfig:
        """Szekely-Rizzo distance correlation preset.

        ``threshold`` is accepted only so a stale value fails with a clear
        ``ConfigurationError`` instead of a ``TypeError``; skfolio ignores it.
        """
        return cls(
            estimator=DistanceEstimatorType.DISTANCE_CORRELATION,
            threshold=threshold,
        )

    @classmethod
    def for_mutual_information(
        cls,
        n_bins: int | None = None,
        *,
        n_bins_method: NBinsMethod | None = None,
        normalize: bool | None = None,
    ) -> DistanceConfig:
        """Mutual-information distance preset.

        By default the bin count is chosen per asset by ``n_bins_method``
        (skfolio default: Freedman-Diaconis). An explicit ``n_bins`` takes
        precedence and disables the rule, so ``n_bins`` and ``n_bins_method``
        should not both be given.
        """
        return cls(
            estimator=DistanceEstimatorType.MUTUAL_INFORMATION,
            n_bins=n_bins,
            n_bins_method=n_bins_method,
            normalize=normalize,
        )
