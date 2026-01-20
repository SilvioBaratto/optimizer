"""
Covariance Matrix Value Object - Immutable covariance representation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass(frozen=True)
class CovarianceMatrix:
    """
    Immutable covariance matrix representation.

    Ensures the matrix is symmetric and positive semi-definite.

    Attributes:
        matrix: 2D numpy array of covariances
        tickers: Ordered list of ticker symbols
        annualized: Whether the matrix is annualized
        estimation_method: Method used for estimation (e.g., 'ledoit_wolf')
        shrinkage_factor: Shrinkage factor if applicable
    """

    matrix: np.ndarray
    tickers: tuple  # Tuple for immutability
    annualized: bool = True
    estimation_method: str = "sample"
    shrinkage_factor: Optional[float] = None

    def __post_init__(self):
        """Validate covariance matrix."""
        if self.matrix.ndim != 2:
            raise ValueError("Covariance matrix must be 2D")

        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")

        if self.matrix.shape[0] != len(self.tickers):
            raise ValueError(
                f"Matrix size {self.matrix.shape[0]} doesn't match "
                f"number of tickers {len(self.tickers)}"
            )

        # Check symmetry (allow small numerical tolerance)
        if not np.allclose(self.matrix, self.matrix.T, atol=1e-10):
            raise ValueError("Covariance matrix must be symmetric")

    @property
    def n_assets(self) -> int:
        """Number of assets in the matrix."""
        return len(self.tickers)

    @property
    def shape(self) -> tuple:
        """Shape of the matrix."""
        return self.matrix.shape

    @property
    def ticker_list(self) -> List[str]:
        """Get tickers as list."""
        return list(self.tickers)

    def __getitem__(self, key) -> Any:
        """Allow array-like indexing."""
        return self.matrix[key]

    def get_variance(self, ticker: str) -> float:
        """Get variance for a specific ticker."""
        idx = self.tickers.index(ticker)
        return self.matrix[idx, idx]

    def get_volatility(self, ticker: str) -> float:
        """Get annualized volatility for a specific ticker."""
        return np.sqrt(self.get_variance(ticker))

    def get_covariance(self, ticker1: str, ticker2: str) -> float:
        """Get covariance between two tickers."""
        idx1 = self.tickers.index(ticker1)
        idx2 = self.tickers.index(ticker2)
        return self.matrix[idx1, idx2]

    def get_correlation(self, ticker1: str, ticker2: str) -> float:
        """Get correlation between two tickers."""
        cov = self.get_covariance(ticker1, ticker2)
        vol1 = self.get_volatility(ticker1)
        vol2 = self.get_volatility(ticker2)
        if vol1 > 0 and vol2 > 0:
            return cov / (vol1 * vol2)
        return 0.0

    def to_correlation_matrix(self) -> np.ndarray:
        """Convert to correlation matrix."""
        vols = np.sqrt(np.diag(self.matrix))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = self.matrix / np.outer(vols, vols)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
        return corr

    def to_dataframe(self) -> Any:
        """
        Convert to pandas DataFrame.

        Returns:
            DataFrame with tickers as both index and columns
        """
        import pandas as pd
        return pd.DataFrame(
            self.matrix,
            index=self.ticker_list,
            columns=self.ticker_list
        )

    def subset(self, tickers: List[str]) -> "CovarianceMatrix":
        """
        Create a subset covariance matrix for given tickers.

        Args:
            tickers: List of tickers to include

        Returns:
            New CovarianceMatrix with only specified tickers
        """
        indices = [self.tickers.index(t) for t in tickers if t in self.tickers]
        subset_matrix = self.matrix[np.ix_(indices, indices)]

        return CovarianceMatrix(
            matrix=subset_matrix,
            tickers=tuple(tickers),
            annualized=self.annualized,
            estimation_method=self.estimation_method,
            shrinkage_factor=self.shrinkage_factor
        )

    @property
    def condition_number(self) -> float:
        """
        Calculate condition number (measure of numerical stability).

        High condition numbers (>1000) indicate ill-conditioning.
        """
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if eigenvalues.min() <= 0:
            return float('inf')
        return eigenvalues.max() / eigenvalues.min()

    @property
    def is_positive_definite(self) -> bool:
        """Check if matrix is positive definite."""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return bool(np.all(eigenvalues > 0))

    @property
    def is_positive_semidefinite(self) -> bool:
        """Check if matrix is positive semi-definite."""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return bool(np.all(eigenvalues >= -1e-10))

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio variance for given weights.

        Args:
            weights: 1D array of weights (same order as tickers)

        Returns:
            Portfolio variance
        """
        return float(weights @ self.matrix @ weights)

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility for given weights.

        Args:
            weights: 1D array of weights (same order as tickers)

        Returns:
            Portfolio volatility (annualized if matrix is annualized)
        """
        return np.sqrt(self.portfolio_variance(weights))

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pd.DataFrame
        annualized: bool = True,
        estimation_method: str = "sample",
        shrinkage_factor: Optional[float] = None
    ) -> "CovarianceMatrix":
        """
        Create from pandas DataFrame.

        Args:
            df: Square DataFrame with tickers as index and columns
            annualized: Whether the matrix is annualized
            estimation_method: Method used for estimation
            shrinkage_factor: Shrinkage factor if applicable

        Returns:
            CovarianceMatrix object
        """
        return cls(
            matrix=df.values.copy(),
            tickers=tuple(df.index.tolist()),
            annualized=annualized,
            estimation_method=estimation_method,
            shrinkage_factor=shrinkage_factor
        )
