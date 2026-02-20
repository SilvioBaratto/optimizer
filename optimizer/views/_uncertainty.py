"""Empirical omega calibration from forecast error track record."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

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

    Parameters
    ----------
    view_history : pd.DataFrame, shape (n_dates, n_views)
        Historical forecasted Q values, one column per view.
    return_history : pd.DataFrame, shape (n_dates, n_views)
        Realised returns aligned to each view, same shape as
        ``view_history``.

    Returns
    -------
    ndarray, shape (n_views, n_views)
        Diagonal Ω matrix where ``Ω_{kk} = Var(Q_k − r_k)``.

    Raises
    ------
    ValueError
        If the two DataFrames have different shapes or column sets,
        or if fewer than 5 aligned observations are available per view.
    """
    if view_history.shape != return_history.shape:
        raise ValueError(
            "view_history and return_history must have the same shape; "
            f"got {view_history.shape} vs {return_history.shape}"
        )
    if list(view_history.columns) != list(return_history.columns):
        raise ValueError(
            "view_history and return_history must have the same column names"
        )

    errors: pd.DataFrame = view_history - return_history

    # Drop rows where any view has a NaN (aligned drop across both inputs)
    errors = errors.dropna()

    n_obs = len(errors)
    if n_obs < _MIN_HISTORY:
        raise ValueError(
            f"calibrate_omega_from_track_record requires at least {_MIN_HISTORY} "
            f"aligned observations after dropping NaN rows, got {n_obs}"
        )

    n_views = errors.shape[1]
    omega = np.zeros((n_views, n_views), dtype=np.float64)
    variances: npt.NDArray[np.float64] = errors.var(axis=0, ddof=1).to_numpy(
        dtype=np.float64
    )
    np.fill_diagonal(omega, variances)
    return omega
