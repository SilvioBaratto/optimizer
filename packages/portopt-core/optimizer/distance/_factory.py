"""Factory for skfolio distance estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skfolio.distance import (
    CovarianceDistance,
    DistanceCorrelation,
    KendallDistance,
    MutualInformation,
    PearsonDistance,
    SpearmanDistance,
)
from skfolio.distance import (
    NBinsMethod as SkNBinsMethod,
)
from skfolio.distance._base import BaseDistance

from optimizer.distance._config import DistanceConfig, DistanceEstimatorType

if TYPE_CHECKING:
    from skfolio.moments import BaseCovariance

_CORR_FAMILY_MAP: dict[DistanceEstimatorType, type[BaseDistance]] = {
    DistanceEstimatorType.PEARSON: PearsonDistance,
    DistanceEstimatorType.KENDALL: KendallDistance,
    DistanceEstimatorType.SPEARMAN: SpearmanDistance,
}


def build_distance(
    config: DistanceConfig,
    *,
    covariance_estimator: BaseCovariance | None = None,
) -> BaseDistance:
    """Build a skfolio distance estimator from *config*.

    Parameters
    ----------
    config : DistanceConfig
        Distance estimator configuration (serialisable primitives / enums).
    covariance_estimator : BaseCovariance or None, optional
        Non-serialisable covariance estimator instance forwarded to
        :class:`skfolio.distance.CovarianceDistance`. Only accepted when
        ``config.estimator`` is ``COVARIANCE``; ``None`` uses the skfolio
        default (:class:`skfolio.moments.GerberCovariance`, an outlier-robust
        co-movement statistic — a sensible default for noisy, heavy-tailed
        market returns).

    Returns
    -------
    BaseDistance
        A fit-ready skfolio distance estimator. After ``.fit(X)`` the estimator
        stores a square ``distance_`` matrix and a ``codependence_`` matrix on
        the columns of ``X``.

    Raises
    ------
    ValueError
        If ``covariance_estimator`` is supplied for a non-covariance estimator.

    Notes
    -----
    The returned estimator expects a clean, aligned **linear**-return matrix
    ``X`` of shape ``(n_observations, n_assets)`` with tickers as columns. All
    six skfolio distance estimators validate ``X`` through sklearn and **reject
    NaN / non-finite values**, and the correlation family needs a full matrix
    (equal-length columns). Real ingestion data (5y ragged history, ``Numeric``
    columns read back as ``Decimal``, mixed ``price_unit`` scales) is therefore
    *not* fit-ready as-is: cast to ``float`` and align/drop NaN upstream — e.g.
    normalise currency/scale, then drop incomplete-history assets via a
    ``SelectComplete`` pre-selection step or ``DataFrame.dropna`` — before
    calling ``.fit(X)``. This estimator is a pure builder and performs no such
    cleaning itself.
    """
    estimator = config.estimator

    if (
        covariance_estimator is not None
        and estimator != DistanceEstimatorType.COVARIANCE
    ):
        raise ValueError(
            "covariance_estimator is only valid for DistanceEstimatorType.COVARIANCE"
        )

    if estimator in _CORR_FAMILY_MAP:
        return _CORR_FAMILY_MAP[estimator](
            absolute=config.absolute,
            power=config.power,
        )

    if estimator == DistanceEstimatorType.COVARIANCE:
        return CovarianceDistance(
            covariance_estimator=covariance_estimator,
            absolute=config.absolute,
            power=config.power,
        )

    if estimator == DistanceEstimatorType.DISTANCE_CORRELATION:
        if config.threshold is None:
            return DistanceCorrelation()
        return DistanceCorrelation(threshold=config.threshold)

    if estimator == DistanceEstimatorType.MUTUAL_INFORMATION:
        kwargs: dict[str, object] = {"n_bins": config.n_bins}
        if config.n_bins_method is not None:
            kwargs["n_bins_method"] = SkNBinsMethod(config.n_bins_method.value)
        if config.normalize is not None:
            kwargs["normalize"] = config.normalize
        return MutualInformation(**kwargs)

    raise ValueError(f"Unsupported distance estimator: {estimator!r}")
