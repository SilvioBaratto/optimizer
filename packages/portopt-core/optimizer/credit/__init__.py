"""Structural credit models.

The Merton (1974) model treats a firm's equity as a call option on its assets with
the face value of its debt as the strike, so the observable equity value and equity
volatility can be inverted into an asset value and volatility. From those come the
distance to default, the default probability, the market value of the debt, its
credit spread and the firm's leverage.

``solve_merton`` takes the equity value and volatility, ``merton_from_market``
takes a market capitalisation and a price history instead, and
``estimate_equity_volatility`` is the annualised volatility estimator the latter
uses.
"""

from optimizer.credit._config import MertonConfig
from optimizer.credit._merton import (
    MertonResult,
    estimate_equity_volatility,
    merton_from_market,
    solve_merton,
)

__all__ = [
    "MertonConfig",
    "MertonResult",
    "estimate_equity_volatility",
    "merton_from_market",
    "solve_merton",
]
