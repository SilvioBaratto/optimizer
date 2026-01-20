"""
Covariance Estimation Module - Robust covariance matrix estimation.

Provides multiple covariance estimators:
- LedoitWolfEstimator: Shrinkage-based robust estimation
- SampleCovarianceEstimator: Simple sample covariance
- ExponentialWeightedEstimator: Time-weighted covariance

All estimators implement the CovarianceEstimator protocol.

Usage:
    from src.black_litterman.covariance import LedoitWolfEstimator

    estimator = LedoitWolfEstimator()
    cov_matrix = estimator.estimate(prices_df)
"""

from optimizer.src.black_litterman.covariance.calculator import (
    LedoitWolfEstimator,
    SampleCovarianceEstimator,
    ExponentialWeightedEstimator,
    get_covariance_estimator,
)
from optimizer.src.black_litterman.covariance.shrinkage import (
    ShrinkageTarget,
    compute_shrinkage_intensity,
)

__all__ = [
    "LedoitWolfEstimator",
    "SampleCovarianceEstimator",
    "ExponentialWeightedEstimator",
    "get_covariance_estimator",
    "ShrinkageTarget",
    "compute_shrinkage_intensity",
]
