"""Option pricing and hedging: Black-Scholes, a binomial tree and delta hedging.

``black_scholes_price`` and ``black_scholes_greeks`` price European options with a
continuous dividend yield and accept numpy arrays that broadcast against each other;
``implied_volatility`` inverts the price. ``binomial_price`` is a
Cox-Ross-Rubinstein tree that also prices American exercise. ``simulate_delta_hedge``
measures the profit and loss of selling an option and hedging it with the
Black-Scholes delta at a finite number of dates, with proportional trading costs and
a hedge volatility that may differ from the true one.

All times are in years and all rates are continuously compounded.
"""

from optimizer.derivatives._binomial import binomial_price
from optimizer.derivatives._black_scholes import (
    Greeks,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from optimizer.derivatives._config import DeltaHedgeConfig, OptionType
from optimizer.derivatives._hedging import DeltaHedgeResult, simulate_delta_hedge

__all__ = [
    "DeltaHedgeConfig",
    "DeltaHedgeResult",
    "Greeks",
    "OptionType",
    "binomial_price",
    "black_scholes_greeks",
    "black_scholes_price",
    "implied_volatility",
    "simulate_delta_hedge",
]
