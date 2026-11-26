"""Black-Scholes prices, Greeks and implied volatility for European options.

Black and Scholes (1973) and Merton (1973) price a European option on an
underlying that pays a continuous dividend yield ``q``. With
``d1 = (ln(S/K) + (r - q + sigma^2 / 2) T) / (sigma sqrt(T))`` and
``d2 = d1 - sigma sqrt(T)``:

* call = ``S e^{-qT} N(d1) - K e^{-rT} N(d2)``;
* put = ``K e^{-rT} N(-d2) - S e^{-qT} N(-d1)``.

The price and the Greeks accept scalars or numpy arrays, broadcast against each
other, so one call prices a whole strike by volatility grid. Hull, "Options,
Futures, and Other Derivatives", gives the formulas used for the Greeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from optimizer.derivatives._config import OptionType, exercise_value
from optimizer.derivatives._validation import (
    Contract,
    FloatArray,
    FloatOrArray,
    ScalarContract,
    as_output,
    broadcast_contract,
    scalar_contract,
)
from optimizer.exceptions import ConvergenceError, DataError

_INV_SQRT_2PI: Final = 1.0 / math.sqrt(2.0 * math.pi)
# Bracket of the implied-volatility root: 0.0001% to 500% a year covers every
# market quote while keeping the price strictly increasing in volatility.
_IV_LOWER: Final = 1e-6
_IV_UPPER: Final = 5.0
_IV_XTOL: Final = 1e-12
_IV_MAX_ITERATIONS: Final = 200


@dataclass(frozen=True)
class Greeks:
    """Sensitivities of a European option price.

    Each field is a float for scalar inputs and an array for array inputs.

    Attributes:
        delta: Change in price per unit change in the spot.
        gamma: Change in delta per unit change in the spot.
        vega: Change in price per 1.00 (100 percentage points) change in volatility.
        theta: Change in price per year of calendar time passing, so it is
            negative for most options.
        rho: Change in price per 1.00 (100 percentage points) change in the rate.
    """

    delta: FloatOrArray
    gamma: FloatOrArray
    vega: FloatOrArray
    theta: FloatOrArray
    rho: FloatOrArray


def _d1_d2(contract: Contract) -> tuple[FloatArray, FloatArray]:
    """Return the Black-Scholes ``d1`` and ``d2`` for a positive maturity."""
    spread = contract.volatility * np.sqrt(contract.maturity)
    drift = contract.rate - contract.dividend_yield + 0.5 * contract.volatility**2
    d1 = (np.log(contract.spot / contract.strike) + drift * contract.maturity) / spread
    return d1, d1 - spread


def _model_price(contract: Contract, kind: OptionType) -> FloatArray:
    """Return the closed-form price, valid only where the maturity is positive."""
    d1, d2 = _d1_d2(contract)
    sign = kind.sign
    asset_leg = contract.spot * np.exp(-contract.dividend_yield * contract.maturity)
    cash_leg = contract.strike * np.exp(-contract.rate * contract.maturity)
    price = sign * (asset_leg * ndtr(sign * d1) - cash_leg * ndtr(sign * d2))
    return np.asarray(price, dtype=np.float64)


def black_scholes_price(
    spot: FloatOrArray,
    strike: FloatOrArray,
    maturity: FloatOrArray,
    rate: FloatOrArray,
    volatility: FloatOrArray,
    option_type: OptionType | str,
    dividend_yield: FloatOrArray = 0.0,
) -> FloatOrArray:
    """Return the Black-Scholes price of a European call or put.

    Args:
        spot: Underlying price, strictly positive.
        strike: Strike price, strictly positive.
        maturity: Time to expiry in years; at zero the intrinsic value is returned.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility, strictly positive.
        option_type: ``OptionType.CALL``, ``OptionType.PUT`` or its string value.
        dividend_yield: Continuous dividend yield of the underlying.

    Returns:
        The price as a float when every input is a scalar, otherwise an array with
        the broadcast shape of the inputs.

    Raises:
        ConfigurationError: If an input is non-finite or outside its range, or the
            option type is unknown.
        DataError: If the array inputs cannot be broadcast together.
    """
    kind = OptionType.parse(option_type)
    contract = broadcast_contract(
        spot, strike, maturity, rate, volatility, dividend_yield
    )
    live = contract.maturity > 0.0
    # Where the option has expired the formula divides by zero, so it is evaluated
    # at a dummy maturity there and discarded.
    priced = _model_price(
        contract._replace(maturity=np.where(live, contract.maturity, 1.0)), kind
    )
    intrinsic = exercise_value(kind, contract.spot, contract.strike)
    return as_output(np.where(live, priced, intrinsic))


def black_scholes_greeks(
    spot: FloatOrArray,
    strike: FloatOrArray,
    maturity: FloatOrArray,
    rate: FloatOrArray,
    volatility: FloatOrArray,
    option_type: OptionType | str,
    dividend_yield: FloatOrArray = 0.0,
) -> Greeks:
    """Return the Black-Scholes Greeks of a European call or put.

    Args:
        spot: Underlying price, strictly positive.
        strike: Strike price, strictly positive.
        maturity: Time to expiry in years, strictly positive: the Greeks are not
            defined at expiry.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility, strictly positive.
        option_type: ``OptionType.CALL``, ``OptionType.PUT`` or its string value.
        dividend_yield: Continuous dividend yield of the underlying.

    Returns:
        The Greeks; vega is per 1.00 change in volatility, theta is per year and
        rho is per 1.00 change in the rate. Fields are floats for scalar inputs and
        arrays otherwise.

    Raises:
        ConfigurationError: If an input is non-finite or outside its range
            (including a maturity of zero), or the option type is unknown.
        DataError: If the array inputs cannot be broadcast together.
    """
    kind = OptionType.parse(option_type)
    contract = broadcast_contract(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        positive_maturity=True,
    )
    return _greeks(contract, kind)


def _greeks(contract: Contract, kind: OptionType) -> Greeks:
    """Return the Greeks of a validated contract with a positive maturity."""
    d1, d2 = _d1_d2(contract)
    sign = kind.sign
    root_t = np.sqrt(contract.maturity)
    asset_disc = np.exp(-contract.dividend_yield * contract.maturity)
    asset_leg = contract.spot * asset_disc
    cash_leg = contract.strike * np.exp(-contract.rate * contract.maturity)
    density = _INV_SQRT_2PI * np.exp(-0.5 * d1**2)
    decay = asset_leg * density * contract.volatility / (2.0 * root_t)
    carry = sign * contract.dividend_yield * asset_leg * ndtr(sign * d1)
    funding = sign * contract.rate * cash_leg * ndtr(sign * d2)
    gamma = asset_disc * density / (contract.spot * contract.volatility * root_t)
    return Greeks(
        delta=as_output(sign * asset_disc * ndtr(sign * d1)),
        gamma=as_output(gamma),
        vega=as_output(asset_leg * density * root_t),
        theta=as_output(-decay - funding + carry),
        rho=as_output(sign * cash_leg * contract.maturity * ndtr(sign * d2)),
    )


def _no_arbitrage_bounds(
    contract: ScalarContract, kind: OptionType
) -> tuple[float, float]:
    """Return the lowest and highest price a European option can have."""
    asset_leg = contract.spot * math.exp(-contract.dividend_yield * contract.maturity)
    cash_leg = contract.strike * math.exp(-contract.rate * contract.maturity)
    if kind is OptionType.CALL:
        return max(asset_leg - cash_leg, 0.0), asset_leg
    return max(cash_leg - asset_leg, 0.0), cash_leg


def _require_price_in_bounds(price: float, lower: float, upper: float) -> None:
    """Raise unless the price is finite and inside the no-arbitrage bounds."""
    if not math.isfinite(price):
        msg = f"price must be finite, got {price}"
        raise DataError(msg)
    if price < lower:
        msg = f"price {price} is below the no-arbitrage lower bound {lower}"
        raise DataError(msg)
    if price > upper:
        msg = f"price {price} is above the no-arbitrage upper bound {upper}"
        raise DataError(msg)


def _price_gap(
    volatility: float, contract: ScalarContract, kind: OptionType, price: float
) -> float:
    """Return the model price at a volatility minus the observed price."""
    model = black_scholes_price(
        contract.spot,
        contract.strike,
        contract.maturity,
        contract.rate,
        volatility,
        kind,
        contract.dividend_yield,
    )
    return float(model) - price


def _solve_volatility(
    contract: ScalarContract, kind: OptionType, price: float
) -> float:
    """Return the volatility in the search interval whose model price is ``price``."""
    args = (contract, kind, price)
    if _price_gap(_IV_LOWER, *args) * _price_gap(_IV_UPPER, *args) > 0.0:
        msg = (
            f"no volatility in [{_IV_LOWER}, {_IV_UPPER}] reproduces the price "
            f"{price}: the root is not bracketed"
        )
        raise ConvergenceError(msg)
    try:
        # scipy ships no stubs, and with full_output=False it returns the bare root.
        root = cast(
            "float",
            brentq(
                _price_gap,
                _IV_LOWER,
                _IV_UPPER,
                args=args,
                xtol=_IV_XTOL,
                maxiter=_IV_MAX_ITERATIONS,
                full_output=False,
            ),
        )
    except RuntimeError as exc:
        msg = f"the implied-volatility solver did not converge for price {price}"
        raise ConvergenceError(msg) from exc
    return float(root)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType | str,
    dividend_yield: float = 0.0,
) -> float:
    """Return the volatility at which Black-Scholes reproduces a market price.

    Solves ``black_scholes_price(..., volatility) = price`` with Brent's method on
    the interval from 0.0001% to 500% a year, where the price rises strictly with
    volatility.

    Args:
        price: Observed option price.
        spot: Underlying price, strictly positive.
        strike: Strike price, strictly positive.
        maturity: Time to expiry in years, strictly positive.
        rate: Continuously compounded risk-free rate.
        option_type: ``OptionType.CALL``, ``OptionType.PUT`` or its string value.
        dividend_yield: Continuous dividend yield of the underlying.

    Returns:
        The implied annualised volatility.

    Raises:
        ConfigurationError: If the contract is invalid, an input is not a scalar,
            or the option type is unknown.
        DataError: If the price is not finite or lies outside the no-arbitrage
            bounds: below the discounted intrinsic value, or above ``S e^{-qT}``
            for a call and ``K e^{-rT}`` for a put.
        ConvergenceError: If no volatility in the search interval reproduces the
            price, or the solver does not converge.
    """
    kind = OptionType.parse(option_type)
    # The volatility argument is a placeholder: it is the unknown being solved for.
    contract = scalar_contract(
        spot, strike, maturity, rate, 1.0, dividend_yield, positive_maturity=True
    )
    _require_price_in_bounds(price, *_no_arbitrage_bounds(contract, kind))
    return _solve_volatility(contract, kind, price)
