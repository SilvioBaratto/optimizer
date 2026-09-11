"""Configuration for distance estimator selection.

The :class:`DistanceConfig` is a serialisable dataclass with a
:class:`DistanceEstimatorType` enum field plus estimator-specific knobs.
Every field holds only primitives / enums so the config round-trips through
serialisation. Non-serialisable objects (e.g. a covariance estimator instance
for :class:`skfolio.distance.CovarianceDistance`) are passed as factory kwargs,
never stored on the config.

skfolio 1.0.6 distance estimators expose a common codependence-to-distance
transform via ``absolute`` and ``power``:

* ``absolute`` — take the absolute value of the codependence before mapping it
  to a distance (treats strong negative and positive correlation as equally
  "close").
* ``power`` — raise the ``(1 - codependence)`` term to this power, sharpening
  or softening the resulting dendrogram.

These apply to the correlation / covariance family (Pearson, Kendall, Spearman,
Covariance). ``DistanceCorrelation`` instead exposes a ``threshold`` and
``MutualInformation`` exposes ``n_bins`` / ``n_bins_method`` / ``normalize``.
Setting a knob that does not belong to the selected estimator raises a
:class:`ConfigurationError` to surface configuration mistakes early.
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


class NBinsMethod(str, Enum):
    """Histogram bin-count rule for :class:`skfolio.distance.MutualInformation`.

    Mirrors ``skfolio.distance.NBinsMethod``. Only consulted when ``n_bins`` is
    ``None`` (an explicit ``n_bins`` overrides the rule).
    """

    FREEDMAN = "freedman"
    KNUTH = "knuth"


#: Estimators that support the ``absolute`` / ``power`` codependence transform.
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

    Parameters
    ----------
    estimator : DistanceEstimatorType
        Which skfolio distance estimator to instantiate.
    absolute : bool
        Take the absolute value of the codependence before mapping to a
        distance. Only valid for the correlation / covariance family
        (Pearson, Kendall, Spearman, Covariance).
    power : float
        Exponent applied to the ``(1 - codependence)`` term. Must be strictly
        positive. Only valid for the correlation / covariance family.
    threshold : float or None
        Codependence threshold for
        :class:`skfolio.distance.DistanceCorrelation`. Only valid when
        ``estimator`` is ``DISTANCE_CORRELATION``. ``None`` keeps the skfolio
        default (``0.5``).
    n_bins : int or None
        Explicit histogram bin count for
        :class:`skfolio.distance.MutualInformation`. Only valid when
        ``estimator`` is ``MUTUAL_INFORMATION``. ``None`` defers to
        ``n_bins_method``.
    n_bins_method : NBinsMethod or None
        Bin-count rule used when ``n_bins`` is ``None``. Only valid when
        ``estimator`` is ``MUTUAL_INFORMATION``. ``None`` keeps the skfolio
        default (Freedman-Diaconis).
    normalize : bool or None
        Whether :class:`skfolio.distance.MutualInformation` normalises the
        mutual information into ``[0, 1]``. Only valid when ``estimator`` is
        ``MUTUAL_INFORMATION``. ``None`` keeps the skfolio default (``True``).
    bandwidth : float or None
        Reserved for kernel-based mutual-information estimators. Not supported
        by skfolio 1.0.6 — must be ``None``. Retained for backward
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
        is_dcorr = self.estimator == DistanceEstimatorType.DISTANCE_CORRELATION
        in_corr_family = self.estimator in _CORR_FAMILY

        # absolute / power belong to the correlation / covariance family only.
        if not in_corr_family and (self.absolute or self.power != 1.0):
            raise ConfigurationError(
                "absolute and power are only valid for the correlation/"
                "covariance distance family (PEARSON, KENDALL, SPEARMAN, "
                "COVARIANCE)"
            )
        if self.power <= 0.0:
            raise ConfigurationError("power must be strictly positive")

        # threshold belongs to DISTANCE_CORRELATION only.
        if not is_dcorr and self.threshold is not None:
            raise ConfigurationError(
                "threshold is only valid for DistanceEstimatorType.DISTANCE_CORRELATION"
            )
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ConfigurationError("threshold must lie in [0, 1]")

        # MI-only knobs.
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
        :func:`optimizer.distance.build_distance` as a kwarg, not stored here.
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
        """Szekely-Rizzo distance correlation preset."""
        return cls(
            estimator=DistanceEstimatorType.DISTANCE_CORRELATION,
            threshold=threshold,
        )

    @classmethod
    def for_mutual_information(
        cls,
        n_bins: int | None = 10,
        *,
        n_bins_method: NBinsMethod | None = None,
        normalize: bool | None = None,
    ) -> DistanceConfig:
        """Mutual-information distance preset.

        Pass ``n_bins=None`` together with ``n_bins_method`` to let skfolio
        choose the bin count via the Freedman-Diaconis or Knuth rule.
        """
        return cls(
            estimator=DistanceEstimatorType.MUTUAL_INFORMATION,
            n_bins=n_bins,
            n_bins_method=n_bins_method,
            normalize=normalize,
        )
