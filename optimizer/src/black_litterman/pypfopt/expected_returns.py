"""
Expected Returns Estimation for Portfolio Optimization
========================================================

This module provides functions for estimating expected returns of assets, which is a
critical input for Black-Litterman portfolio optimization and mean-variance analysis.

Expected returns represent the anticipated future performance of each asset and are
used to calculate optimal portfolio weights. These estimates can come from historical
data (this module) or from forward-looking models like CAPM or Black-Litterman.
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd


def returns_from_prices(
    prices: pd.DataFrame,
    log_returns: bool = False
) -> pd.DataFrame:
    """
    Calculate returns from a price series.

    Converts a time series of asset prices into a time series of returns.
    Returns can be computed as simple percentage changes (default) or as
    logarithmic returns for improved mathematical properties.

    This is typically the first step in portfolio optimization workflows,
    as most risk and return calculations require returns rather than prices.
    """
    if log_returns:
        # Calculate log returns: break into steps for type checker
        pct_change = prices.pct_change(fill_method=None)
        log_rets = np.log(1 + pct_change)
        # np.log preserves DataFrame type, but type checker needs help
        returns = pd.DataFrame(log_rets, index=prices.index, columns=prices.columns).dropna(how="all")
    else:
        returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns
