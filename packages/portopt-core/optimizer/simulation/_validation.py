"""Input checks shared by the simulators.

Nothing here is exported from the package. The error type follows one rule, stated
in ``_gbm``: ``ConfigurationError`` for settings (including the arrays that act as
settings, such as the start prices and drifts of a market, and an overflow that the
settings cause), ``DataError`` for data arrays (prices, returns, covariance). Every
check that coerces an input with numpy turns numpy's ``TypeError`` or
``ValueError`` into the documented exception, so a ragged or non-numeric argument
fails the same way as a wrong value.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt

from optimizer.exceptions import ConfigurationError, DataError, OptimizerError


def _is_real(value: object) -> bool:
    """Return whether ``value`` is a real number; a bool is a flag, not a number."""
    return isinstance(value, Real) and not isinstance(value, bool)


def require_finite(name: str, value: float) -> None:
    """Raise unless ``value`` is a finite real number.

    Args:
        name: Setting name used in the error message.
        value: Value to check.

    Raises:
        ConfigurationError: If ``value`` is not a real number, or is NaN or infinite.
    """
    if not (_is_real(value) and math.isfinite(value)):
        msg = f"{name} must be a finite number, got {value!r}"
        raise ConfigurationError(msg)


def require_positive(name: str, value: float) -> None:
    """Raise unless ``value`` is a finite real number above zero.

    Args:
        name: Setting name used in the error message.
        value: Value to check.

    Raises:
        ConfigurationError: If ``value`` is not a finite real number above zero.
    """
    if not (_is_real(value) and math.isfinite(value) and value > 0.0):
        msg = f"{name} must be finite and positive, got {value!r}"
        raise ConfigurationError(msg)


def require_non_negative(name: str, value: float) -> None:
    """Raise unless ``value`` is a finite real number of at least zero.

    Args:
        name: Setting name used in the error message.
        value: Value to check.

    Raises:
        ConfigurationError: If ``value`` is not a finite real number of at least zero.
    """
    if not (_is_real(value) and math.isfinite(value) and value >= 0.0):
        msg = f"{name} must be finite and non-negative, got {value!r}"
        raise ConfigurationError(msg)


def require_open_unit_interval(name: str, value: float) -> None:
    """Raise unless ``value`` is a real number strictly between 0 and 1.

    Args:
        name: Setting name used in the error message.
        value: Value to check.

    Raises:
        ConfigurationError: If ``value`` is not a real number inside ``(0, 1)``.
    """
    if not (_is_real(value) and 0.0 < value < 1.0):
        msg = f"{name} must be strictly between 0 and 1, got {value!r}"
        raise ConfigurationError(msg)


def require_at_least_one(name: str, value: int) -> None:
    """Raise unless ``value`` is an integer of at least one.

    Args:
        name: Setting name used in the error message.
        value: Count to check; a float, a bool or NaN is not a count.

    Raises:
        ConfigurationError: If ``value`` is not an integer or is below one.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = f"{name} must be an integer, got {value!r}"
        raise ConfigurationError(msg)
    if value < 1:
        msg = f"{name} must be at least 1, got {value!r}"
        raise ConfigurationError(msg)


def require_seed(random_state: int | None) -> None:
    """Raise if ``random_state`` is not a seed numpy accepts.

    Args:
        random_state: Seed, or ``None`` for fresh entropy.

    Raises:
        ConfigurationError: If the seed is neither ``None`` nor a non-negative
            integer.
    """
    if random_state is None:
        return
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        msg = f"random_state must be an integer or None, got {random_state!r}"
        raise ConfigurationError(msg)
    if random_state < 0:
        msg = f"random_state must be non-negative or None, got {random_state!r}"
        raise ConfigurationError(msg)


def as_float_array(
    name: str, values: npt.ArrayLike, error: type[OptimizerError]
) -> npt.NDArray[np.float64]:
    """Return ``values`` as a float array, whatever numpy makes of a bad input.

    Args:
        name: Argument name used in the error message.
        values: Anything numpy can read as a rectangular array of numbers.
        error: Exception class to raise for an input numpy cannot convert.

    Returns:
        The values as a ``float64`` array of the input's shape.

    Raises:
        OptimizerError: ``error``, if the input is ragged or not numeric.
    """
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a rectangular array of numbers: {exc}"
        raise error(msg) from exc


def as_vector(
    name: str,
    values: npt.ArrayLike,
    *,
    error: type[OptimizerError],
    min_size: int = 1,
) -> npt.NDArray[np.float64]:
    """Return ``values`` as a finite one-dimensional float array.

    Args:
        name: Argument name used in the error message.
        values: Anything numpy can read as a one-dimensional array of numbers.
        error: Exception class to raise for an invalid input.
        min_size: Fewest values the array may hold.

    Returns:
        The values as a ``float64`` vector.

    Raises:
        OptimizerError: ``error``, if the input is not numeric, not
            one-dimensional, shorter than ``min_size`` or not finite.
    """
    vector = as_float_array(name, values, error)
    if vector.ndim != 1 or vector.size < min_size:
        msg = (
            f"{name} must be a one-dimensional array of at least {min_size} "
            f"values, got shape {vector.shape}"
        )
        raise error(msg)
    if not np.isfinite(vector).all():
        msg = f"{name} must be finite: drop or fill missing values first"
        raise error(msg)
    return vector


def as_returns(
    returns: npt.ArrayLike, min_observations: int
) -> npt.NDArray[np.float64]:
    """Return ``returns`` as a validated one-dimensional float array.

    Args:
        returns: Simple per-period returns.
        min_observations: Fewest values the series may have.

    Returns:
        The returns as a float array.

    Raises:
        DataError: If the series is not numeric, not one-dimensional, shorter than
            ``min_observations``, has a non-finite value or a return below -100%,
            which no unlevered position can suffer.
    """
    sample = as_vector("returns", returns, error=DataError, min_size=min_observations)
    if (sample < -1.0).any():
        msg = "returns must be at least -1 (a loss of more than 100% is impossible)"
        raise DataError(msg)
    return sample


def require_no_overflow(values: npt.NDArray[np.float64], remedy: str) -> None:
    """Raise if a simulated array left the float range.

    Args:
        values: Simulated prices or wealth.
        remedy: What the caller can change, appended to the message.

    Raises:
        ConfigurationError: If any value is infinite or NaN. The settings (drift,
            volatility, horizon, scale) decide whether a path overflows.
    """
    if not np.isfinite(values).all():
        msg = f"simulated values overflow the float range: {remedy}"
        raise ConfigurationError(msg)
