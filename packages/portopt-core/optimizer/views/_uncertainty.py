"""Empirical omega calibration from forecast error track record."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import DataError

logger = logging.getLogger(__name__)

_MIN_HISTORY = 5


def calibrate_omega_from_track_record(
    view_history: pd.DataFrame,
    return_history: pd.DataFrame,
) -> npt.NDArray[np.float64]:
    """Calibrate the Black-Litterman omega matrix from a forecast error track record.

    Implements the empirical method described in the theory:

        Ω_{kk} = Var(Q_{k,t} − r_{k,t})

    where ``Q_{k,t}`` is the analyst's forecast for view ``k`` at time ``t``
    and ``r_{k,t}`` is the realised return for that view.  The diagonal
    entries are non-negative by construction (they are sample variances).

    Args:
        view_history: Historical forecasted Q values, one column per view,
            indexed by date.
        return_history: Realised returns aligned to each view, same shape
            and columns as ``view_history``.

    Returns:
        Diagonal Ω matrix of shape (n_views, n_views) where
        ``Ω_{kk} = Var(Q_k − r_k)``.

    Raises:
        DataError: If the two DataFrames differ in shape or column names,
            or if any view has fewer than 5 aligned observations after
            dropping its NaN entries.
    """
    if view_history.shape != return_history.shape:
        raise DataError(
            "view_history and return_history must have the same shape; "
            f"got {view_history.shape} vs {return_history.shape}"
        )
    if list(view_history.columns) != list(return_history.columns):
        raise DataError(
            "view_history and return_history must have the same column names"
        )

    errors: pd.DataFrame = view_history - return_history

    # A NaN in either input propagates into that view's error, so dropping per column
    # aligns forecast and realisation for that view only. Dropping whole rows would
    # shrink every view's sample to the shortest track record.
    n_views = errors.shape[1]
    variances = np.empty(n_views, dtype=np.float64)
    for k in range(n_views):
        view_errors = errors.iloc[:, k].dropna()
        if len(view_errors) < _MIN_HISTORY:
            raise DataError(
                f"calibrate_omega_from_track_record requires at least "
                f"{_MIN_HISTORY} aligned observations per view after dropping "
                f"NaN entries; view {errors.columns[k]!r} has {len(view_errors)}"
            )
        variances[k] = view_errors.var(ddof=1)

    omega: npt.NDArray[np.float64] = np.diag(variances)
    return omega
