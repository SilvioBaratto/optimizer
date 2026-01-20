import warnings
from typing import Union

import numpy as np
import pandas as pd


def returns_from_prices(prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """
    Calculate returns from a price series.
    """
    if log_returns:
        # Calculate log returns: break into steps for type checker
        pct_change = prices.pct_change(fill_method=None)
        log_rets = np.log(1 + pct_change)
        # np.log preserves DataFrame type, but type checker needs help
        returns = pd.DataFrame(log_rets, index=prices.index, columns=prices.columns).dropna(
            how="all"
        )
    else:
        returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns
