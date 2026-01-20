"""
Covariance Calculator - Implementations of covariance estimators.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from optimizer.domain.value_objects.covariance import CovarianceMatrix

logger = logging.getLogger(__name__)


class BaseCovarianceEstimator(ABC):
    """
    Abstract base class for covariance estimators.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._last_shrinkage: Optional[float] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable estimator name."""
        pass

    @abstractmethod
    def _estimate_internal(
        self,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, Optional[float]]:
        """
        Internal estimation method.
        """
        pass

    def estimate(
        self,
        prices: pd.DataFrame,
        frequency: int = 252,
    ) -> CovarianceMatrix:
        """
        Estimate the covariance matrix from price data.
        """
        # Calculate returns
        returns = prices.pct_change().dropna()

        if len(returns) < 30:
            self._logger.warning(
                f"Only {len(returns)} observations for covariance estimation. "
                "Results may be unreliable."
            )

        # Estimate covariance
        cov_matrix, shrinkage = self._estimate_internal(returns)
        self._last_shrinkage = shrinkage

        # Annualize
        cov_matrix_annualized = cov_matrix * frequency

        self._logger.info(
            f"{self.name}: Estimated {len(prices.columns)}x{len(prices.columns)} "
            f"covariance matrix from {len(returns)} observations"
        )
        if shrinkage is not None:
            self._logger.info(f"  Shrinkage intensity: {shrinkage:.4f}")

        return CovarianceMatrix(
            matrix=cov_matrix_annualized,
            tickers=tuple(prices.columns.tolist()),
            annualized=True,
            estimation_method=self.name.lower().replace(" ", "_"),
            shrinkage_factor=shrinkage,
        )

    def get_shrinkage_factor(self) -> Optional[float]:
        """Get the shrinkage factor used (if applicable)."""
        return self._last_shrinkage


class LedoitWolfEstimator(BaseCovarianceEstimator):
    """
    Ledoit-Wolf shrinkage estimator.
    """

    @property
    def name(self) -> str:
        return "Ledoit-Wolf"

    def _estimate_internal(
        self,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, Optional[float]]:
        """
        Estimate covariance using Ledoit-Wolf shrinkage.
        """
        lw = LedoitWolf()
        lw.fit(returns.values)

        return lw.covariance_, lw.shrinkage_


class SampleCovarianceEstimator(BaseCovarianceEstimator):
    """
    Simple sample covariance estimator.

    Computes the standard sample covariance matrix. Not recommended for
    portfolios with many assets relative to observations due to:
    - High estimation error
    - Potential ill-conditioning
    """

    @property
    def name(self) -> str:
        return "Sample Covariance"

    def _estimate_internal(
        self,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, Optional[float]]:
        """
        Estimate sample covariance.
        """
        cov_matrix = returns.cov().values
        return cov_matrix, None


class ExponentialWeightedEstimator(BaseCovarianceEstimator):
    """
    Exponentially weighted moving average covariance estimator.
    """

    def __init__(self, halflife: int = 63):
        """
        Initialize exponential weighted estimator.

        Args:
            halflife: Decay halflife in trading days
        """
        super().__init__()
        self._halflife = halflife

    @property
    def name(self) -> str:
        return f"Exponential Weighted (halflife={self._halflife})"

    def _estimate_internal(
        self,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, Optional[float]]:
        """
        Estimate exponentially weighted covariance.
        """
        cov_matrix = returns.ewm(halflife=self._halflife).cov().iloc[-len(returns.columns) :]
        cov_matrix = cov_matrix.values.reshape(len(returns.columns), len(returns.columns))
        return cov_matrix, None


def get_covariance_estimator(method: str, **kwargs) -> BaseCovarianceEstimator:
    """
    Factory function to get a covariance estimator by name.
    """
    methods = {
        "ledoit_wolf": LedoitWolfEstimator,
        "sample": SampleCovarianceEstimator,
        "exponential": ExponentialWeightedEstimator,
    }

    method_lower = method.lower().replace("-", "_").replace(" ", "_")

    if method_lower not in methods:
        raise ValueError(
            f"Unknown covariance method: {method}. " f"Available: {list(methods.keys())}"
        )

    return methods[method_lower](**kwargs)
