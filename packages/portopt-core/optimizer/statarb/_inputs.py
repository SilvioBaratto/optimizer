"""Input validation shared by the pair-trading tools.

Research tools only: nothing in ``optimizer.statarb`` models transaction costs,
borrow costs or short-sale limits, and the fund mandate is long-only and
paper-only, so none of it is wired to execution.
"""

from __future__ import annotations

import math
import numbers
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import ConfigurationError, DataError

# A residual sum of squares below this share of the target's own sum of squares is
# a fit to machine precision, where a t-statistic or a variance is arithmetic noise.
EXACT_FIT_RATIO: Final = 1e-24


def require_integer(name: str, value: object, minimum: int) -> int:
    """Return a setting as an ``int``, raising unless it is a whole number.

    ``bool`` is rejected although it subclasses ``int``, and so is every float,
    including NaN and integral values such as ``3.0``: a count written as a float
    is a mistake worth surfacing before it reaches ``range`` or a slice.

    Args:
        name: Setting name used in the error message.
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


def require_finite(name: str, value: object) -> float:
    """Return a setting as a ``float``, raising unless it is a finite real number.

    NaN would slip through a ``<`` comparison, so it is rejected here, together
    with infinities, ``bool`` and non-numeric values.

    Args:
        name: Setting name used in the error message.
        value: Candidate setting.

    Returns:
        The value as a Python ``float``.

    Raises:
        ConfigurationError: If the value is not a finite real number.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        msg = f"{name} must be a finite number, got {value!r}"
        raise ConfigurationError(msg)
    return float(value)


def as_float_array(values: object, name: str) -> npt.NDArray[np.float64]:
    """Return ``values`` as a float64 array.

    Args:
        values: A Series, DataFrame, array or nested sequence of numbers.
        name: Name of the input, used in the error message.

    Returns:
        The values as a float64 array of the same shape.

    Raises:
        DataError: If the values cannot be read as numbers, for example text.
    """
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{name} must contain only numbers"
        raise DataError(msg) from error


def aligned_pair(
    y: pd.Series, x: pd.Series, min_observations: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return two validated, index-aligned series as float arrays.

    Args:
        y: Dependent series.
        x: Independent series, on exactly the same index as ``y``.
        min_observations: Fewest observations accepted.

    Returns:
        The values of ``y`` and ``x`` as one-dimensional float arrays.

    Raises:
        DataError: If an input is not a Series, the indexes differ, there are
            fewer than ``min_observations`` rows, a value is not a number, or a
            value is not finite.
    """
    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        msg = "y and x must be pandas Series"
        raise DataError(msg)
    if not y.index.equals(x.index):
        msg = "y and x must have the same index"
        raise DataError(msg)
    if len(y) < min_observations:
        msg = f"at least {min_observations} observations are needed, got {len(y)}"
        raise DataError(msg)
    y_values = as_float_array(y, "y")
    x_values = as_float_array(x, "x")
    if not (np.isfinite(y_values).all() and np.isfinite(x_values).all()):
        msg = "y and x must be finite: drop or fill missing values first"
        raise DataError(msg)
    return y_values, x_values
