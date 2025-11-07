"""
Portfolio Performance Metrics and Objective Functions
======================================================

This module provides fundamental portfolio performance metrics used throughout
PyPortfolioOpt, particularly for evaluating Black-Litterman portfolio allocations.

These functions form the foundation of modern portfolio theory calculations,
implementing the core formulas used in mean-variance optimization and risk-adjusted
return analysis.
"""

from typing import Union

import numpy as np
import pandas as pd


def portfolio_variance(
    w: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame]
) -> float:
    """
    Calculate the total portfolio variance (squared volatility).

    Portfolio variance measures the total risk of a portfolio, accounting for both
    individual asset volatilities and their correlations. It is the square of the
    portfolio standard deviation (volatility).

    This function implements the fundamental mean-variance optimization formula
    from Modern Portfolio Theory (Markowitz, 1952).
    """
    # Ensure numpy arrays for computation (avoid ExtensionArray)
    w_array = np.asarray(w.values) if isinstance(w, pd.Series) else np.asarray(w)
    cov_array = np.asarray(cov_matrix.values) if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)

    return float(w_array @ cov_array @ w_array)


def portfolio_return(
    w: Union[np.ndarray, pd.Series],
    expected_returns: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate the expected return of a portfolio.

    Portfolio return is the weighted average of individual asset expected returns.
    This is a fundamental metric in Modern Portfolio Theory and is used in
    mean-variance optimization, Sharpe ratio calculations, and Black-Litterman
    portfolio analysis.
    """
    # Ensure numpy arrays for computation (avoid ExtensionArray)
    w_array = np.asarray(w.values) if isinstance(w, pd.Series) else np.asarray(w)
    ret_array = np.asarray(expected_returns.values) if isinstance(expected_returns, pd.Series) else np.asarray(expected_returns)

    return float(w_array @ ret_array)


def sharpe_ratio(
    w: Union[np.ndarray, pd.Series],
    expected_returns: Union[np.ndarray, pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate the Sharpe ratio of a portfolio.

    The Sharpe ratio measures risk-adjusted performance by comparing the portfolio's
    excess return (above the risk-free rate) to its volatility. Higher Sharpe ratios
    indicate better risk-adjusted performance.

    This is one of the most widely used metrics in finance for comparing investment
    strategies and portfolio allocations. In Black-Litterman optimization, the Sharpe
    ratio helps evaluate whether incorporating investor views improves risk-adjusted
    returns compared to the market-implied prior.
    """
    mu = portfolio_return(w, expected_returns)
    sigma = np.sqrt(portfolio_variance(w, cov_matrix))

    if sigma == 0:
        raise ZeroDivisionError("Portfolio volatility is zero, cannot calculate Sharpe ratio")

    return float((mu - risk_free_rate) / sigma)
