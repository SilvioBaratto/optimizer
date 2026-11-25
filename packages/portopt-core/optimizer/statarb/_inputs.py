"""Input validation shared by the pair-trading tools.

Research tools only: nothing in ``optimizer.statarb`` models transaction costs,
borrow costs or short-sale limits, and the fund mandate is long-only and
paper-only, so none of it is wired to execution.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import DataError

# A residual sum of squares below this share of the target's own sum of squares is
# a fit to machine precision, where a t-statistic or a variance is arithmetic noise.
EXACT_FIT_RATIO: Final = 1e-24


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
            fewer than ``min_observations`` rows, or a value is not finite.
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
    y_values = y.to_numpy(dtype=np.float64)
    x_values = x.to_numpy(dtype=np.float64)
    if not (np.isfinite(y_values).all() and np.isfinite(x_values).all()):
        msg = "y and x must be finite: drop or fill missing values first"
        raise DataError(msg)
    return y_values, x_values
