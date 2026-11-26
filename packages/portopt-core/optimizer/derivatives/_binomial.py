"""Cox-Ross-Rubinstein binomial tree for European and American options.

Cox, Ross and Rubinstein (1979), "Option Pricing: A Simplified Approach", let the
spot move up by ``u = e^{sigma sqrt(dt)}`` or down by ``d = 1/u`` each step with the
risk-neutral up probability ``p = (e^{(r - q) dt} - d) / (u - d)``. Working back from
the terminal payoffs, each node is the discounted expectation of its two children;
an American option takes the larger of that and its exercise value at every node.

The tree recombines, so a level holds ``k + 1`` nodes and the whole lattice of
prices is one array of ``2n + 1`` entries; the backward induction is a vectorised
update per level, with no Python loop over nodes.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from optimizer.derivatives._config import OptionType, exercise_value
from optimizer.derivatives._validation import (
    FloatArray,
    require_integer,
    scalar_contract,
)
from optimizer.exceptions import ConfigurationError

# exp(709) is the largest finite float64; the margin keeps the discounted sums of
# the backward induction finite as well.
_MAX_LOG_PRICE: Final = 700.0


def _crr_parameters(
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    steps: int,
) -> tuple[float, float, float]:
    """Return the log up-move, the one-step discount and the up probability."""
    dt = maturity / steps
    log_up = volatility * math.sqrt(dt)
    up = math.exp(log_up)
    down = 1.0 / up
    if up <= down:
        msg = (
            f"volatility {volatility} is too small for {steps} steps: "
            "the up and down moves coincide"
        )
        raise ConfigurationError(msg)
    probability = (math.exp((rate - dividend_yield) * dt) - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        msg = (
            f"the risk-neutral probability {probability} is outside [0, 1]: "
            "use more steps or a larger volatility"
        )
        raise ConfigurationError(msg)
    return log_up, math.exp(-rate * dt), probability


def _lattice(spot: float, log_up: float, steps: int) -> FloatArray:
    """Return all node prices; entry ``m + steps`` holds ``S u^m``, ``|m| <= steps``."""
    if math.log(spot) + log_up * steps > _MAX_LOG_PRICE:
        msg = "the node prices overflow float64: lower the volatility or the steps"
        raise ConfigurationError(msg)
    return spot * np.exp(log_up * np.arange(-steps, steps + 1, dtype=np.float64))


def _backward_induction(
    exercise: FloatArray,
    steps: int,
    discount: float,
    probability: float,
    american: bool,
) -> float:
    """Roll the terminal payoffs back to the root of the tree.

    ``exercise`` holds the exercise value of every lattice entry, so the nodes of
    level ``k`` are the slice ``[steps - k, steps + k]`` taken every second entry.
    """
    values = exercise[::2]
    for level in range(steps - 1, -1, -1):
        values = discount * (
            probability * values[1:] + (1.0 - probability) * values[:-1]
        )
        if american:
            nodes = exercise[steps - level : steps + level + 1 : 2]
            values = np.maximum(values, nodes)
    return float(values[0])


def binomial_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str,
    steps: int = 200,
    dividend_yield: float = 0.0,
    american: bool = False,
) -> float:
    """Return the Cox-Ross-Rubinstein binomial price of a call or put.

    Args:
        spot: Underlying price, strictly positive.
        strike: Strike price, strictly positive.
        maturity: Time to expiry in years; at zero the intrinsic value is returned.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility, strictly positive.
        option_type: ``OptionType.CALL``, ``OptionType.PUT`` or its string value.
        steps: Number of time steps; the European price converges to the
            Black-Scholes price as it grows.
        dividend_yield: Continuous dividend yield of the underlying.
        american: Allow exercise at every node instead of only at expiry.

    Returns:
        The option price.

    Raises:
        ConfigurationError: If an input is invalid or not a scalar, ``steps`` is
            not a positive integer, the risk-neutral probability falls outside
            ``[0, 1]`` (a high rate against a low volatility needs more steps), or
            the node prices overflow.
    """
    kind = OptionType.parse(option_type)
    n_steps = require_integer("steps", steps, 1)
    contract = scalar_contract(spot, strike, maturity, rate, volatility, dividend_yield)
    if contract.maturity == 0.0:
        return float(exercise_value(kind, contract.spot, contract.strike))
    log_up, discount, probability = _crr_parameters(
        contract.maturity,
        contract.rate,
        contract.volatility,
        contract.dividend_yield,
        n_steps,
    )
    nodes = _lattice(contract.spot, log_up, n_steps)
    exercise = exercise_value(kind, nodes, contract.strike)
    return _backward_induction(exercise, n_steps, discount, probability, american)
