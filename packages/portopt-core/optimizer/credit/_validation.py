"""Input checks shared by the Merton model's settings and public functions.

Every number that reaches the package is untrusted. A plain ``<`` comparison lets
NaN through and lets a float stand in for a count, so each check names the argument
and rejects what the comparison would miss.
"""

from __future__ import annotations

import math
import numbers

from optimizer.exceptions import ConfigurationError


def require_real(name: str, value: object) -> float:
    """Return a setting as a ``float``, raising unless it is a real number.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting; Python and numpy numbers are accepted, while a
            ``bool``, a string or ``None`` is rejected.

    Returns:
        The value as a Python ``float``; it may still be NaN or infinite.

    Raises:
        ConfigurationError: If the value is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"{name} must be a real number, got {value!r}"
        raise ConfigurationError(msg)
    return float(value)


def require_finite(name: str, value: object) -> float:
    """Return a setting as a ``float``, raising unless it is finite.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting.

    Returns:
        The value as a Python ``float``.

    Raises:
        ConfigurationError: If the value is not a real number, or is NaN or
            infinite.
    """
    number = require_real(name, value)
    if not math.isfinite(number):
        msg = f"{name} must be finite, got {value!r}"
        raise ConfigurationError(msg)
    return number


def require_positive_finite(name: str, value: object) -> float:
    """Return a setting as a ``float``, raising unless it is finite and above zero.

    Args:
        name: Argument name used in the error message.
        value: Candidate setting.

    Returns:
        The value as a Python ``float``.

    Raises:
        ConfigurationError: If the value is not a real number, or is NaN, infinite,
            zero or negative.
    """
    number = require_real(name, value)
    if not (math.isfinite(number) and number > 0.0):
        msg = f"{name} must be a finite number above zero, got {value!r}"
        raise ConfigurationError(msg)
    return number


def require_integer_at_least(name: str, value: object, minimum: int) -> int:
    """Return a setting as an ``int``, raising unless it is a whole number.

    ``bool`` is rejected although it subclasses ``int``, and so is every float,
    including NaN and integral values such as ``3.0``: a count written as a float is
    a mistake worth surfacing before it reaches numpy or scipy.

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
