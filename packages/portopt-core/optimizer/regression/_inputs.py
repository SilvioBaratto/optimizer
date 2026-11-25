"""Validation of the return series the CAPM and factor tools take.

Nothing here repairs data: a missing, infinite or misaligned value raises
``DataError`` so that the caller decides how to clean it, instead of the regression
silently dropping rows and changing the sample behind a statistic.
"""

from __future__ import annotations

import math
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import ConfigurationError, DataError

_Returns = TypeVar("_Returns", pd.Series, pd.DataFrame)


def require_finite(values: object, name: str) -> None:
    """Raise if ``values`` holds a missing or infinite number.

    Args:
        values: Array-like of floats.
        name: Name used in the error message.

    Raises:
        DataError: If any element is NaN or infinite.
    """
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        msg = f"{name} must be finite: drop or fill missing values first"
        raise DataError(msg)


def to_float_array(values: object, name: str) -> npt.NDArray[np.float64]:
    """Return ``values`` as a finite float64 array.

    Args:
        values: Array-like of numbers, or a pandas object.
        name: Name used in the error message.

    Returns:
        A float64 array of the same shape.

    Raises:
        DataError: If the values are not numeric or are not all finite.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be numeric"
        raise DataError(msg) from error
    require_finite(array, name)
    return array


def as_series(values: pd.Series, name: str) -> pd.Series:
    """Return a finite float64 copy of a return series.

    Args:
        values: Per-period returns indexed by date.
        name: Name used in the error message.

    Returns:
        The same data as float64.

    Raises:
        DataError: If ``values`` is not a Series, or is not numeric and finite.
    """
    if not isinstance(values, pd.Series):
        msg = f"{name} must be a pandas Series, got {type(values).__name__}"
        raise DataError(msg)
    checked = to_float_array(values, name)
    return pd.Series(checked, index=values.index, name=values.name)


def as_frame(values: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a finite float64 copy of a table of return series.

    Args:
        values: One column of per-period returns for each asset or factor.
        name: Name used in the error message.

    Returns:
        The same data as float64.

    Raises:
        DataError: If ``values`` is not a DataFrame, has no columns, repeats a
            column name, or is not numeric and finite.
    """
    if not isinstance(values, pd.DataFrame):
        msg = f"{name} must be a pandas DataFrame, got {type(values).__name__}"
        raise DataError(msg)
    if values.shape[1] == 0:
        msg = f"{name} needs at least one column"
        raise DataError(msg)
    if not values.columns.is_unique:
        msg = f"the column names of {name} must be unique"
        raise DataError(msg)
    checked = to_float_array(values, name)
    return pd.DataFrame(checked, index=values.index, columns=values.columns)


def require_same_index(
    first: pd.Series | pd.DataFrame,
    second: pd.Series | pd.DataFrame,
    first_name: str,
    second_name: str,
) -> None:
    """Raise unless two pandas objects share exactly the same index.

    Args:
        first: First object.
        second: Second object.
        first_name: Name of the first object for the error message.
        second_name: Name of the second object for the error message.

    Raises:
        DataError: If the indexes differ in length, labels or order.
    """
    if not first.index.equals(second.index):
        msg = (
            f"{first_name} and {second_name} must share the same index: "
            "align them before the call"
        )
        raise DataError(msg)


def subtract_risk_free(
    returns: _Returns, risk_free_rate: float | pd.Series, name: str = "returns"
) -> _Returns:
    """Return per-period excess returns over a scalar or per-date risk-free rate.

    Args:
        returns: Per-period returns of one asset (Series) or of several (DataFrame,
            one column each).
        risk_free_rate: Per-period rate as a scalar, or a series on the same index
            as ``returns``.
        name: Name of ``returns`` for the error message.

    Returns:
        ``returns`` minus the rate, with the same shape and labels.

    Raises:
        ConfigurationError: If a scalar rate is not a finite number.
        DataError: If a rate series is not finite or does not share the index.
    """
    if isinstance(risk_free_rate, pd.Series):
        rate = as_series(risk_free_rate, "risk_free_rate")
        require_same_index(returns, rate, name, "risk_free_rate")
        return returns.sub(rate, axis=0)
    if not isinstance(risk_free_rate, (int, float, np.integer, np.floating)):
        msg = f"risk_free_rate must be a number or a Series, got {risk_free_rate!r}"
        raise ConfigurationError(msg)
    scalar = float(risk_free_rate)
    if not math.isfinite(scalar):
        msg = f"risk_free_rate must be finite, got {risk_free_rate!r}"
        raise ConfigurationError(msg)
    return returns - scalar


def annualized_mean(
    values: pd.Series | npt.NDArray[np.float64], periods_per_year: int
) -> float:
    """Return the arithmetic mean of per-period returns scaled to a year.

    Args:
        values: Per-period returns.
        periods_per_year: Number of periods in a year.

    Returns:
        The mean times ``periods_per_year``.
    """
    return float(np.mean(values)) * periods_per_year
