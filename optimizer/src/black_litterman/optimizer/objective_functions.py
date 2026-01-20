from typing import Union

import numpy as np
import pandas as pd


def portfolio_variance(
    w: Union[np.ndarray, pd.Series], cov_matrix: Union[np.ndarray, pd.DataFrame]
) -> float:
    """
    Calculate the total portfolio variance (squared volatility).
    """
    # Ensure numpy arrays for computation (avoid ExtensionArray)
    w_array = np.asarray(w.values) if isinstance(w, pd.Series) else np.asarray(w)
    cov_array = (
        np.asarray(cov_matrix.values)
        if isinstance(cov_matrix, pd.DataFrame)
        else np.asarray(cov_matrix)
    )

    return float(w_array @ cov_array @ w_array)


def portfolio_return(
    w: Union[np.ndarray, pd.Series], expected_returns: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate the expected return of a portfolio.
    """
    # Ensure numpy arrays for computation (avoid ExtensionArray)
    w_array = np.asarray(w.values) if isinstance(w, pd.Series) else np.asarray(w)
    ret_array = (
        np.asarray(expected_returns.values)
        if isinstance(expected_returns, pd.Series)
        else np.asarray(expected_returns)
    )

    return float(w_array @ ret_array)


def sharpe_ratio(
    w: Union[np.ndarray, pd.Series],
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate the Sharpe ratio of a portfolio.
    """
    mu = portfolio_return(w, expected_returns)
    sigma = np.sqrt(portfolio_variance(w, cov_matrix))

    if sigma == 0:
        raise ZeroDivisionError("Portfolio volatility is zero, cannot calculate Sharpe ratio")

    return float((mu - risk_free_rate) / sigma)
