"""Survivorship-bias guard: apply delisting returns."""

from __future__ import annotations

import logging
import math

import pandas as pd

from optimizer.exceptions import DataError

logger = logging.getLogger(__name__)


def apply_delisting_returns(
    returns: pd.DataFrame,
    delisting_returns: dict[str, float],
) -> pd.DataFrame:
    """Replace each ticker's last valid return with its delisting return.

    This prevents survivorship bias by incorporating the terminal return an
    investor would have realised when a stock was delisted.

    DB contract (``instruments.delisting_return``)
    ----------------------------------------------
    The value is a **simple (linear) return** already signed as a return, and
    is written verbatim onto the terminal period — it is *not* negated,
    scaled, or defaulted here:

    - ``-0.30`` — the CRSP-style default for a performance/liquidity delisting
      (a 30% terminal loss).
    - ``-1.0`` — bankruptcy / total loss.
    - Any other realised terminal return.

    The DB column is nullable and (on a fresh universe) currently all-``NULL``.
    ``NULL`` means "not delisted / terminal return unknown" and **must be
    resolved by the caller** (e.g. to the ``-0.30`` default) before calling —
    this DB-agnostic function cannot query the column or invent a policy, so a
    ``None`` / non-finite value is rejected rather than silently written as
    ``NaN`` (which would corrupt a real observation and quietly *reintroduce*
    survivorship bias).

    Because the module receives no ``delisted_at`` date, the terminal return is
    written at each ticker's last valid (non-``NaN``) return, overwriting it.
    The remainder of the window (rows past the last valid return) is then filled
    with ``0.0`` — the delisted asset is held as cash post-death.  This keeps the
    column free of trailing ``NaN`` so a downstream ``SelectComplete`` (which
    drops trailing-``NaN`` columns) retains the asset *with* its realised loss,
    instead of dropping it and silently discarding the terminal return.  Leading
    ``NaN`` (a late listing) is left untouched, so genuinely short-history assets
    are still dropped rather than fabricated.

    Parameters
    ----------
    returns : pd.DataFrame
        Dates x tickers return matrix.
    delisting_returns : dict[str, float]
        Mapping of ticker to its (finite) delisting return.  Each ticker's
        last valid (non-NaN) return is replaced with this value.  Tickers whose
        column is entirely ``NaN`` are skipped.

    Returns
    -------
    pd.DataFrame
        A copy of *returns* with delisting returns applied.

    Raises
    ------
    DataError
        If a ticker in *delisting_returns* is not in *returns* columns, or its
        delisting return is ``None`` / non-finite (unresolved ``NULL``).
    """
    result = returns.copy()

    for ticker, delist_ret in delisting_returns.items():
        if ticker not in result.columns:
            raise DataError(f"Ticker {ticker!r} not found in returns columns")

        if delist_ret is None or not math.isfinite(delist_ret):
            raise DataError(
                f"Delisting return for {ticker!r} is {delist_ret!r}; resolve a "
                "NULL/unknown delisting_return to a finite value (e.g. the "
                "CRSP-style -0.30 default or -1.0 for bankruptcy) before calling."
            )

        col = result[ticker]
        if col.isna().all():
            continue

        last_valid = col.last_valid_index()
        result.at[last_valid, ticker] = delist_ret

        # Hold the asset as cash (0.0) for the remainder of the window rather
        # than leaving trailing NaN after the terminal return.  A downstream
        # SelectComplete drops trailing-NaN columns, which would discard the
        # delisted asset (and its realised loss) entirely; an explicit 0.0
        # keeps the loss in the sample without fabricating post-death returns.
        pos = result.index.get_loc(last_valid)
        if pos + 1 < len(result):
            result.iloc[pos + 1 :, result.columns.get_loc(ticker)] = 0.0

    return result


def delisting_protection_mask(
    returns: pd.DataFrame,
    delisting_returns: dict[str, float],
) -> pd.DataFrame:
    """Boolean mask marking each ticker's delisting (terminal-return) cell.

    Companion to :func:`apply_delisting_returns`: it flags exactly the cells
    that function overwrites with a delisting return — each ticker's last valid
    (non-``NaN``) observation — so a downstream ``OutlierTreater`` can exempt
    those cells from outlier removal/winsorisation.  A genuine delisting return
    (e.g. the CRSP ``-0.30`` or ``-1.0`` bankruptcy total loss) is a real
    economic event, not a data error: on daily equity vol its z-score is huge
    (``|z|`` ≈ 11 for ``-0.30``), so an unguarded ``OutlierTreater`` would NaN
    it as a "data error" (and, being the terminal row, ``SelectComplete`` would
    then drop the whole asset) or winsorise it to ``μ ± kσ`` — either way
    silently defeating the survivorship correction.

    Built from the *pre-fill* ``returns`` (the same input passed to
    :func:`apply_delisting_returns`), so the marked cell is the original last
    valid observation — the one that receives the terminal return — not the
    ``0.0`` post-death padding written after it.

    Parameters
    ----------
    returns : pd.DataFrame
        Dates x tickers return matrix, *before* delisting is applied.
    delisting_returns : dict[str, float]
        The same mapping passed to :func:`apply_delisting_returns`.  Tickers
        absent from *returns* or whose column is entirely ``NaN`` contribute no
        mark (mirroring that function's skip logic); no validation is repeated
        here — :func:`apply_delisting_returns` is the validator.

    Returns
    -------
    pd.DataFrame
        Boolean matrix aligned to *returns* (same index and columns); ``True``
        only at each ticker's terminal-return cell, ``False`` everywhere else.
    """
    mask = pd.DataFrame(False, index=returns.index, columns=returns.columns)

    for ticker in delisting_returns:
        if ticker not in returns.columns:
            continue
        col = returns[ticker]
        if col.isna().all():
            continue
        mask.at[col.last_valid_index(), ticker] = True

    return mask
