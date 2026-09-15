"""Shared Decimal/object -> float64 coercion for price panels and axis=0 transformers.

The DB ``price_history`` OHLCV columns are SQL ``Numeric(20, 6)`` and read back
as Python ``Decimal`` (pandas ``object`` dtype).  A return frame derived from
them (or built directly from DB values) is therefore object-dtype, which either
crashes numpy/skfolio math outright (``float`` - ``Decimal`` raises
``TypeError`` — e.g. in :class:`~optimizer.preprocessing.OutlierTreater`'s
z-scores) or silently propagates ``Decimal`` cells.  Every time-series
transformer casts its input here, at its own boundary, so it is independently
safe on DB-sourced data.  The cast preserves ``NaN`` gaps and is a no-op for
already-float frames.
"""

from __future__ import annotations

import pandas as pd

from optimizer.exceptions import DataError

__all__ = ["_coerce_numeric"]


def _coerce_numeric(X: pd.DataFrame, who: str) -> pd.DataFrame:
    """Return *X* cast to ``float64`` when it carries object-dtype columns.

    Parameters
    ----------
    X : pd.DataFrame
        Return frame that may contain ``Decimal`` (object-dtype) columns.
    who : str
        Caller name, used in the error message.

    Returns
    -------
    pd.DataFrame
        ``X`` unchanged when already float, else a float64 copy.

    Raises
    ------
    DataError
        If an object-dtype column cannot be cast to float (non-numeric data).
    """
    if all(pd.api.types.is_float_dtype(dt) for dt in X.dtypes):
        return X
    try:
        return X.astype("float64")
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"{who}: a column could not be cast to float64 "
            "(expected DB Numeric/Decimal or float, got non-numeric data)"
        ) from exc
