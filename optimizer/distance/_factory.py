"""Factory for skfolio distance estimators."""

from __future__ import annotations

from skfolio.distance import (
    CovarianceDistance,
    DistanceCorrelation,
    KendallDistance,
    MutualInformation,
    PearsonDistance,
    SpearmanDistance,
)
from skfolio.distance._base import BaseDistance

from optimizer.distance._config import DistanceConfig, DistanceEstimatorType

_NO_ARG_MAP: dict[DistanceEstimatorType, type[BaseDistance]] = {
    DistanceEstimatorType.PEARSON: PearsonDistance,
    DistanceEstimatorType.KENDALL: KendallDistance,
    DistanceEstimatorType.SPEARMAN: SpearmanDistance,
    DistanceEstimatorType.COVARIANCE: CovarianceDistance,
    DistanceEstimatorType.DISTANCE_CORRELATION: DistanceCorrelation,
}


def build_distance(config: DistanceConfig) -> BaseDistance:
    """Build a skfolio distance estimator from *config*.

    Parameters
    ----------
    config : DistanceConfig
        Distance estimator configuration.

    Returns
    -------
    BaseDistance
        A fitted-ready skfolio distance estimator. After ``.fit(X)`` the
        estimator stores a square ``distance_`` matrix on the columns of
        ``X``.
    """
    if config.estimator == DistanceEstimatorType.MUTUAL_INFORMATION:
        return MutualInformation(n_bins=config.n_bins)
    return _NO_ARG_MAP[config.estimator]()
