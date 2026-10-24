"""Survivorship-bias guard: apply delisting returns."""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Mapping
from typing import TypeAlias

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError

logger = logging.getLogger(__name__)

DelistingDate: TypeAlias = dt.date | dt.datetime | pd.Timestamp | str


def _delisting_timestamp(
    ticker: str, raw: DelistingDate | None, index: pd.DatetimeIndex
) -> pd.Timestamp:
    """Parse one delisting date, refusing any value that cannot be placed.

    A ``NaT`` would sort past every row and book the terminal return on the
    window end, the defect ``delisted_at`` exists to prevent, so null and
    unparseable dates raise instead of degrading to that inference.
    """
    try:
        when = pd.Timestamp(raw) if raw is not None else pd.NaT
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"Delisting date {raw!r} for {ticker!r} cannot be parsed as a date"
        ) from exc
    if not isinstance(when, pd.Timestamp) or pd.isna(when):
        raise DataError(
            f"Delisting date for {ticker!r} is {raw!r}; resolve a NULL delisted_at "
            "or drop the ticker from delisted_at before calling."
        )
    if (when.tzinfo is None) != (index.tz is None):
        raise DataError(
            f"Delisting date {when} for {ticker!r} and the returns index differ "
            "in timezone awareness; localise both the same way."
        )
    return when


def _terminal_position(
    returns: pd.DataFrame,
    ticker: str,
    delisted_at: Mapping[str, DelistingDate] | None,
) -> int | None:
    """Row position that receives *ticker*'s terminal return.

    Args:
        returns: Dates x tickers return matrix.
        ticker: Column to locate; must be in *returns*.
        delisted_at: Optional ticker -> delisting date. When it names *ticker*,
            the position is the last row dated on or before that date.
            Otherwise the position is the column's last non-``NaN`` row.

    Returns:
        Integer row position, or ``None`` when the column is entirely ``NaN``.

    Raises:
        DataError: If a delisting date is given but *returns* lacks a sorted
            ``DatetimeIndex``, the date is null, unparseable or differs from
            the index in timezone awareness, or it precedes the first row.
    """
    valid = np.flatnonzero(returns[ticker].notna().to_numpy())
    if valid.size == 0:
        return None

    if delisted_at is None or ticker not in delisted_at:
        return int(valid[-1])

    index = returns.index
    if not isinstance(index, pd.DatetimeIndex) or not index.is_monotonic_increasing:
        raise DataError(
            f"delisted_at for {ticker!r} needs returns indexed by a sorted "
            "DatetimeIndex"
        )
    when = _delisting_timestamp(ticker, delisted_at[ticker], index)
    pos = int(index.searchsorted(when, side="right")) - 1
    if pos < 0:
        raise DataError(
            f"Delisting date {when.date()} for {ticker!r} precedes the first "
            f"row of returns ({index[0].date()})"
        )
    return pos


def apply_delisting_returns(
    returns: pd.DataFrame,
    delisting_returns: dict[str, float],
    delisted_at: Mapping[str, DelistingDate] | None = None,
) -> pd.DataFrame:
    """Replace each ticker's terminal-period return with its delisting return.

    This prevents survivorship bias by incorporating the terminal return an
    investor would have realised when a stock was delisted.

    DB contract (``instruments.delisting_return``):
    The value is a **simple (linear) return** already signed as a return, and
    is written verbatim onto the terminal period — it is *not* negated,
    scaled, or defaulted here:

    - ``-0.30`` — the CRSP-style default for a performance/liquidity delisting
      (a 30% terminal loss).
    - ``-1.0`` — bankruptcy / total loss.
    - Any other realised terminal return.

    The DB column is nullable and ``NULL`` is ambiguous on its own; read it
    together with ``instruments.delisted_at``:

    - ``delisted_at`` NULL — still in the universe, no terminal return exists.
      Leave the ticker out of *delisting_returns*.
    - ``delisted_at`` set, ``delisting_return`` set — a delisting. The universe
      builder stores the ``-0.30`` default when the source stopped returning the
      symbol, so a genuinely unknown terminal return is never left ``NULL``.
      Pass the value.
    - ``delisted_at`` set, ``delisting_return`` NULL — the line left the
      universe (dedup, floor, venue allowlist) but is **still trading**. It has
      no terminal return. Leave it out of *delisting_returns* so its history
      stays untouched; do **not** resolve it to the ``-0.30`` default, which
      would impose a loss nobody realised.

    This DB-agnostic function cannot query the columns or invent a policy, so a
    ``None`` / non-finite value is rejected rather than silently written as
    ``NaN`` (which would corrupt a real observation and quietly *reintroduce*
    survivorship bias). Filtering out the still-trading rows is the caller's job.

    Where the terminal return lands depends on ``delisted_at``.  Without a date
    it is written at each ticker's last valid (non-``NaN``) return, which is the
    delisting period only if the return matrix keeps the post-delisting tail as
    ``NaN``.  The default ``to_returns`` / ``prices_to_returns`` forward-fill
    prices (``fill_nan=True``), turning that tail into exact ``0.0`` returns, so
    the last valid return is then the end of the window — possibly years after
    the real delisting.  For such matrices pass ``delisted_at`` so the terminal
    return is booked on the real date.

    The remainder of the window (rows past the terminal return) is then filled
    with ``0.0`` — the delisted asset is held as cash post-death.  This keeps the
    column free of trailing ``NaN`` so a downstream ``SelectComplete`` (which
    drops trailing-``NaN`` columns) retains the asset *with* its realised loss,
    instead of dropping it and silently discarding the terminal return.  Leading
    ``NaN`` (a late listing) is left untouched, so genuinely short-history assets
    are still dropped rather than fabricated.

    Args:
        returns: Dates x tickers return matrix.
        delisting_returns: Mapping of ticker to its delisting return, a finite
            fraction of at least ``-1.0``.  Tickers whose column is entirely
            ``NaN`` are skipped.
        delisted_at: Optional mapping of ticker to its delisting date.  For a
            listed ticker the terminal return replaces the return on the last
            row dated on or before that date, and every later row becomes
            ``0.0`` regardless of its prior value.  Requires *returns* to have
            a sorted ``DatetimeIndex``.  Tickers not listed here use their last
            valid return.

    Returns:
        A copy of *returns* with delisting returns applied.

    Raises:
        DataError: If a ticker in *delisting_returns* is not in *returns*
            columns, or its delisting return is ``None`` / non-finite
            (unresolved ``NULL``) or below ``-1.0`` (impossible for a linear
            return), or a delisting date is null, unparseable, mismatched with
            the index in timezone awareness or earlier than the first row, or
            the index is not a sorted ``DatetimeIndex``.
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

        if delist_ret < -1.0:
            raise DataError(
                f"Delisting return for {ticker!r} is {delist_ret!r}, below the "
                "-1.0 total loss a linear return can reach; pass a fraction "
                "(-0.30), not a percentage (-30)."
            )

        pos = _terminal_position(result, ticker, delisted_at)
        if pos is None:
            continue

        inferred = delisted_at is None or ticker not in delisted_at
        if inferred and pos == len(result) - 1:
            logger.warning(
                "Delisting return for %r lands on the final row of the "
                "window; if the returns were forward-filled (the default "
                "of to_returns), pass delisted_at to book it on the real "
                "delisting date.",
                ticker,
            )

        loc = result.columns.get_loc(ticker)
        result.iloc[pos, loc] = delist_ret

        # Hold the asset as cash (0.0) for the remainder of the window rather
        # than leaving trailing NaN after the terminal return.  A downstream
        # SelectComplete drops trailing-NaN columns, which would discard the
        # delisted asset (and its realised loss) entirely; an explicit 0.0
        # keeps the loss in the sample without fabricating post-death returns.
        if pos + 1 < len(result):
            result.iloc[pos + 1 :, loc] = 0.0

    return result


def delisting_protection_mask(
    returns: pd.DataFrame,
    delisting_returns: dict[str, float],
    delisted_at: Mapping[str, DelistingDate] | None = None,
) -> pd.DataFrame:
    """Return a boolean mask marking each ticker's delisting (terminal-return) cell.

    Companion to `apply_delisting_returns`: it flags exactly the cells
    that function overwrites with a delisting return — each ticker's terminal
    row (its last valid observation, or the row at ``delisted_at``) — so a
    downstream ``OutlierTreater`` can exempt
    those cells from outlier removal/winsorisation.  A genuine delisting return
    (e.g. the CRSP ``-0.30`` or ``-1.0`` bankruptcy total loss) is a real
    economic event, not a data error: on daily equity vol its z-score is huge
    (``|z|`` ≈ 11 for ``-0.30``), so an unguarded ``OutlierTreater`` would NaN
    it as a "data error" (and, being the terminal row, ``SelectComplete`` would
    then drop the whole asset) or winsorise it to ``μ ± kσ`` — either way
    silently defeating the survivorship correction.

    Built from the *pre-fill* ``returns`` (the same input passed to
    `apply_delisting_returns`), so the marked cell is the original last
    valid observation — the one that receives the terminal return — not the
    ``0.0`` post-death padding written after it.

    Args:
        returns: Dates x tickers return matrix, *before* delisting is applied.
        delisting_returns: The same mapping passed to
            `apply_delisting_returns`. Tickers absent from *returns* or
            whose column is entirely ``NaN`` contribute no mark (mirroring that
            function's skip logic); no validation is repeated here —
            `apply_delisting_returns` is the validator.
        delisted_at: The same optional ticker -> delisting date mapping passed
            to `apply_delisting_returns`; it must match so the mark lands on
            the cell that receives the terminal return.

    Returns:
        Boolean matrix aligned to *returns* (same index and columns); ``True``
        only at each ticker's terminal-return cell, ``False`` everywhere else.
    """
    mask = pd.DataFrame(False, index=returns.index, columns=returns.columns)

    for ticker in delisting_returns:
        if ticker not in returns.columns:
            continue
        pos = _terminal_position(returns, ticker, delisted_at)
        if pos is None:
            continue
        mask.iloc[pos, mask.columns.get_loc(ticker)] = True

    return mask
