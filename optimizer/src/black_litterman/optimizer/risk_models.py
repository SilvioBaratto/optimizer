import warnings
from typing import Union

import numpy as np
import pandas as pd

from .enums import FixMethod
from .expected_returns import returns_from_prices


def _is_positive_semidefinite(matrix: Union[np.ndarray, pd.DataFrame]) -> bool:
    """
    Check if a matrix is positive semidefinite (PSD).
    """
    try:
        # Significantly more efficient than checking eigenvalues
        # Add small epsilon for numerical stability
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(
    matrix: Union[np.ndarray, pd.DataFrame],
    fix_method: Union[str, FixMethod] = FixMethod.SPECTRAL,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Validate and repair a covariance matrix to ensure it is positive semidefinite.
    """
    # Allow string or enum
    if isinstance(fix_method, str):
        try:
            fix_method = FixMethod(fix_method)
        except ValueError:
            raise NotImplementedError(f"Method {fix_method} not implemented")

    if _is_positive_semidefinite(matrix):
        return matrix

    warnings.warn("The covariance matrix is non positive semidefinite. Amending eigenvalues.")

    # Eigendecomposition
    q, V = np.linalg.eigh(matrix)

    if fix_method == FixMethod.SPECTRAL:
        # Remove negative eigenvalues
        q = np.where(q > 0, q, 0)
        # Reconstruct matrix
        fixed_matrix = V @ np.diag(q) @ V.T
    elif fix_method == FixMethod.DIAG:
        # Add small positive value to diagonal to make PSD
        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
    else:
        raise NotImplementedError(f"Method {fix_method} not implemented")

    if not _is_positive_semidefinite(fixed_matrix):  # pragma: no cover
        warnings.warn("Could not fix matrix. Please try a different risk model.", UserWarning)

    # Rebuild labels if provided
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix


def sample_cov(
    prices: Union[pd.DataFrame, pd.Series],
    returns_data: bool = False,
    frequency: int = 252,
    log_returns: bool = False,
    fix_method: Union[str, FixMethod] = FixMethod.SPECTRAL,
) -> pd.DataFrame:
    """
    Calculate the annualized sample covariance matrix of asset returns.
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    # Calculate covariance and annualize
    cov_matrix = returns.cov() * frequency

    # fix_nonpositive_semidefinite preserves DataFrame type when input is DataFrame
    fixed_cov = fix_nonpositive_semidefinite(cov_matrix, fix_method)
    assert isinstance(fixed_cov, pd.DataFrame), "Expected DataFrame output"
    return fixed_cov


def ledoit_wolf_shrinkage(
    prices: Union[pd.DataFrame, pd.Series],
    returns_data: bool = False,
    frequency: int = 252,
    log_returns: bool = False,
) -> pd.DataFrame:
    """
    Calculate the Ledoit-Wolf shrinkage covariance matrix.
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError(
            "sklearn is required for Ledoit-Wolf shrinkage. "
            "Install with: pip install scikit-learn"
        )

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices = prices.copy()
        prices.index = prices.index.tz_localize(None)

    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    # Drop rows where ANY stock has NaN (only use common trading days)
    # This ensures we only use dates where all stocks have data
    returns_clean = returns.dropna()

    # Apply Ledoit-Wolf shrinkage
    lw = LedoitWolf()
    shrunk_cov = lw.fit(returns_clean).covariance_

    # Annualize
    shrunk_cov = shrunk_cov * frequency

    # Convert back to DataFrame with labels
    tickers = returns_clean.columns
    shrunk_cov_df = pd.DataFrame(shrunk_cov, index=tickers, columns=tickers)

    return shrunk_cov_df


def exp_cov(
    prices: Union[pd.DataFrame, pd.Series],
    returns_data: bool = False,
    frequency: int = 252,
    log_returns: bool = False,
    span: int = 180,
    fix_method: Union[str, FixMethod] = FixMethod.SPECTRAL,
) -> pd.DataFrame:
    """
    Calculate exponentially-weighted covariance matrix.
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    # Calculate exponentially weighted covariance
    cov_matrix = returns.ewm(span=span).cov().iloc[-len(returns.columns) :]

    # Annualize
    cov_matrix = cov_matrix * frequency

    # fix_nonpositive_semidefinite preserves DataFrame type when input is DataFrame
    fixed_cov = fix_nonpositive_semidefinite(cov_matrix, fix_method)
    assert isinstance(fixed_cov, pd.DataFrame), "Expected DataFrame output"
    return fixed_cov
