"""Input checks shared by the figure builders.

Every chart input is external data, so each builder validates it before plotly sees
it and reports a problem as a :class:`~optimizer.exceptions.DataError` that names
the offending argument.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from optimizer.exceptions import ConfigurationError, DataError


def require_integer(name: str, value: object, minimum: int) -> int:
    """Return a setting as an ``int``, raising unless it is a whole number.

    ``bool`` is rejected although it subclasses ``int``, and so is every float,
    including NaN and integral values such as ``3.0``: a size or a count written as a
    float is a mistake worth surfacing before it reaches plotly or pandas.

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


def as_finite_vector(
    values: npt.ArrayLike, name: str, *, min_size: int = 1
) -> npt.NDArray[np.float64]:
    """Return ``values`` as a validated one-dimensional float array.

    Args:
        values: Array-like of numbers.
        name: Argument name used in error messages.
        min_size: Fewest elements the vector may have.

    Returns:
        A new one-dimensional ``float64`` array.

    Raises:
        DataError: If ``values`` is not numeric, not one-dimensional, shorter than
            ``min_size`` or holds a missing or infinite value.
    """
    try:
        array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be numeric"
        raise DataError(msg) from error
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional, got shape {array.shape}"
        raise DataError(msg)
    if array.size < min_size:
        msg = f"{name} needs at least {min_size} values, got {array.size}"
        raise DataError(msg)
    if not np.isfinite(array).all():
        msg = f"{name} must be finite: drop or fill missing values first"
        raise DataError(msg)
    return array


def as_finite_scalar(value: float, name: str) -> float:
    """Return ``value`` as a finite float.

    Args:
        value: A number. Anything ``float`` cannot convert is reported as a
            ``DataError`` rather than a ``TypeError``, whatever the caller passed.
        name: Argument name used in error messages.

    Returns:
        The value as a ``float``.

    Raises:
        DataError: If ``value`` is not a number or is not finite.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be a number, got {value!r}"
        raise DataError(msg) from error
    if not math.isfinite(number):
        msg = f"{name} must be finite, got {number}"
        raise DataError(msg)
    return number


def require_non_negative(values: npt.NDArray[np.float64], name: str) -> None:
    """Reject a vector with a negative entry.

    Args:
        values: Validated numeric vector.
        name: Argument name used in error messages.

    Raises:
        DataError: If any entry is below zero.
    """
    if (values < 0.0).any():
        msg = f"{name} must be non-negative"
        raise DataError(msg)


def as_labels(labels: Sequence[object] | None, size: int) -> list[str] | None:
    """Return one text label per point, or ``None`` when no labels were given.

    Args:
        labels: Optional sequence with one entry per point.
        size: Number of points the labels must match.

    Returns:
        The labels as strings, or ``None``.

    Raises:
        DataError: If the number of labels differs from ``size``.
    """
    if labels is None:
        return None
    texts = [str(label) for label in labels]
    if len(texts) != size:
        msg = f"labels needs one entry per point: {size} points, {len(texts)} labels"
        raise DataError(msg)
    return texts
