"""Input checks shared by the pricing, tree and hedging modules.

Every public function of the package accepts untrusted numbers, so the checks
live in one place and report the offending argument by name. Array inputs are
validated element-wise and broadcast together, which is what lets the
Black-Scholes functions price a whole strike or volatility grid in one call.
"""

from __future__ import annotations

import numbers
from typing import NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from optimizer.exceptions import ConfigurationError, DataError

FloatArray: TypeAlias = npt.NDArray[np.float64]
FloatOrArray: TypeAlias = float | FloatArray


class Contract(NamedTuple):
    """Broadcast, validated option inputs; every field has the same shape."""

    spot: FloatArray
    strike: FloatArray
    maturity: FloatArray
    rate: FloatArray
    volatility: FloatArray
    dividend_yield: FloatArray


class ScalarContract(NamedTuple):
    """Validated option inputs for the routines that price one contract."""

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float


def as_float_array(name: str, value: FloatOrArray) -> FloatArray:
    """Convert an input to a float array.

    Args:
        name: Argument name used in the error message.
        value: Scalar or array-like number.

    Returns:
        The input as a ``float64`` array (zero-dimensional for a scalar).

    Raises:
        ConfigurationError: If the value is not numeric.
    """
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric, got {value!r}"
        raise ConfigurationError(msg) from exc


def require_integer(name: str, value: object, minimum: int) -> int:
    """Return a setting as an ``int``, raising unless it is a whole number.

    ``bool`` is rejected although it subclasses ``int``, and so is every float,
    including NaN and integral values such as ``3.0``: a count or a seed written as
    a float is a mistake worth surfacing before it reaches numpy.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting; Python and numpy integers are accepted.
        minimum: Smallest accepted value.

    Returns:
        The value as a Python ``int``.

    Raises:
        ConfigurationError: If the value is not an integer or is below ``minimum``.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        msg = f"{name} must be an integer of at least {minimum}, got {value!r}"
        raise ConfigurationError(msg)
    return int(value)


def require_real(name: str, value: object) -> float:
    """Return a setting as a ``float``, raising unless it is a real number.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting; ``bool`` and non-numeric values are rejected.

    Returns:
        The value as a Python ``float`` (it may still be NaN or infinite).

    Raises:
        ConfigurationError: If the value is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"{name} must be a real number, got {value!r}"
        raise ConfigurationError(msg)
    return float(value)


def require_finite(name: str, values: FloatArray) -> None:
    """Raise unless every element is finite.

    Args:
        name: Argument name used in the error message.
        values: Array to check.

    Raises:
        ConfigurationError: If any element is NaN or infinite.
    """
    if not np.all(np.isfinite(values)):
        msg = f"{name} must be finite"
        raise ConfigurationError(msg)


def require_positive(name: str, values: FloatArray) -> None:
    """Raise unless every element is finite and strictly positive.

    Args:
        name: Argument name used in the error message.
        values: Array to check.

    Raises:
        ConfigurationError: If any element is NaN, infinite or not above zero.
    """
    if not np.all(np.isfinite(values) & (values > 0.0)):
        msg = f"{name} must be positive and finite"
        raise ConfigurationError(msg)


def require_non_negative(name: str, values: FloatArray) -> None:
    """Raise unless every element is finite and at least zero.

    Args:
        name: Argument name used in the error message.
        values: Array to check.

    Raises:
        ConfigurationError: If any element is NaN, infinite or negative.
    """
    if not np.all(np.isfinite(values) & (values >= 0.0)):
        msg = f"{name} must be non-negative and finite"
        raise ConfigurationError(msg)


def broadcast_contract(
    spot: FloatOrArray,
    strike: FloatOrArray,
    maturity: FloatOrArray,
    rate: FloatOrArray,
    volatility: FloatOrArray,
    dividend_yield: FloatOrArray = 0.0,
    *,
    positive_maturity: bool = False,
) -> Contract:
    """Validate the option inputs and broadcast them to a common shape.

    Args:
        spot: Underlying price, strictly positive.
        strike: Strike price, strictly positive.
        maturity: Time to expiry in years; zero is allowed unless
            ``positive_maturity`` is set.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility, strictly positive.
        dividend_yield: Continuous dividend yield.
        positive_maturity: Require a maturity above zero, for the routines that
            have no meaning at expiry (Greeks, implied volatility, trees).

    Returns:
        The six inputs as float arrays of one broadcast shape.

    Raises:
        ConfigurationError: If an input is non-numeric, non-finite or outside
            its range.
        DataError: If the inputs cannot be broadcast together.
    """
    spot_a = as_float_array("spot", spot)
    strike_a = as_float_array("strike", strike)
    maturity_a = as_float_array("maturity", maturity)
    rate_a = as_float_array("rate", rate)
    vol_a = as_float_array("volatility", volatility)
    yield_a = as_float_array("dividend_yield", dividend_yield)
    require_positive("spot", spot_a)
    require_positive("strike", strike_a)
    if positive_maturity:
        require_positive("maturity", maturity_a)
    else:
        require_non_negative("maturity", maturity_a)
    require_finite("rate", rate_a)
    require_positive("volatility", vol_a)
    require_finite("dividend_yield", yield_a)
    try:
        shaped = np.broadcast_arrays(
            spot_a, strike_a, maturity_a, rate_a, vol_a, yield_a
        )
    except ValueError as exc:
        msg = "the option inputs cannot be broadcast together"
        raise DataError(msg) from exc
    return Contract(*shaped)


def scalar_contract(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    positive_maturity: bool = False,
) -> ScalarContract:
    """Validate the inputs of a routine that prices exactly one contract.

    Args:
        spot: Underlying price, strictly positive.
        strike: Strike price, strictly positive.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility, strictly positive.
        dividend_yield: Continuous dividend yield.
        positive_maturity: Require a maturity above zero.

    Returns:
        The six inputs as Python floats.

    Raises:
        ConfigurationError: If an input is invalid or is not a scalar.
        DataError: If the inputs cannot be broadcast together.
    """
    contract = broadcast_contract(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        positive_maturity=positive_maturity,
    )
    if contract.spot.ndim != 0:
        msg = "the inputs must be scalars, not arrays"
        raise ConfigurationError(msg)
    return ScalarContract(*(float(field) for field in contract))


def as_output(values: FloatArray) -> FloatOrArray:
    """Return a Python float for a zero-dimensional result, else the array.

    Args:
        values: Result of an array computation.

    Returns:
        ``float`` when ``values`` has no dimensions, otherwise ``values``.
    """
    return float(values) if values.ndim == 0 else values
