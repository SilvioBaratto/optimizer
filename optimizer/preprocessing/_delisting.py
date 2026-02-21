"""Survivorship-bias guard: apply delisting returns."""

from __future__ import annotations

import pandas as pd

from optimizer.exceptions import DataError


def apply_delisting_returns(
    returns: pd.DataFrame,
    delisting_returns: dict[str, float],
) -> pd.DataFrame:
    """Replace each ticker's last valid return with its delisting return.

    This prevents survivorship bias by incorporating the terminal return
    that investors would have experienced when a stock was delisted.

    Parameters
    ----------
    returns : pd.DataFrame
        Dates x tickers return matrix.
    delisting_returns : dict[str, float]
        Mapping of ticker to delisting return value.  Each ticker's
        last valid (non-NaN) return is replaced with this value.

    Returns
    -------
    pd.DataFrame
        A copy of *returns* with delisting returns applied.

    Raises
    ------
    DataError
        If a ticker in *delisting_returns* is not in *returns* columns.
    """
    result = returns.copy()

    for ticker, delist_ret in delisting_returns.items():
        if ticker not in result.columns:
            raise DataError(
                f"Ticker {ticker!r} not found in returns columns"
            )

        col = result[ticker]
        if col.isna().all():
            continue

        last_valid = col.last_valid_index()
        result.at[last_valid, ticker] = delist_ret

    return result
