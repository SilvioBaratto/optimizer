"""Shared constants and pure utility functions for data assembly.

Extracted from ``data_assembly.py`` — every other ``research/data/``
sub-module depends on this foundation module.
"""

from __future__ import annotations

import logging
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

from app.models.universe.universe import Instrument  # noqa: E402

from research.data._currency import currency_dedup_rank  # noqa: E402

logger = logging.getLogger(__name__)

# Number of trading days per year (equity convention).
_TRADING_DAYS: int = 252

# Maximum fraction of input rows that may be dropped by `_pivot_with_dedup`
# before the pivot is treated as a structural failure.  Production runs
# observe ~0.18% (10,072 / ~5.5M); 5% catches accidental full-exchange
# duplication while leaving normal cross-listing dedup silent.
_DEDUP_DROP_THRESHOLD_PCT: float = 0.05

# Country -> Region mapping used by downstream cycles (Cycle 4 checklist,
# Cycle 5 reporting).  Unmapped countries default to ``"Other"`` at call
# sites; ``"Other"`` is intentionally NOT a key in this dict.
REGION_MAP: dict[str, str] = {
    # Americas
    "United States": "Americas",
    "Canada": "Americas",
    "Mexico": "Americas",
    "Brazil": "Americas",
    "Argentina": "Americas",
    "Chile": "Americas",
    "Colombia": "Americas",
    # Europe
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Switzerland": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Belgium": "Europe",
    "Austria": "Europe",
    "Ireland": "Europe",
    "Portugal": "Europe",
    "Luxembourg": "Europe",
    "Monaco": "Europe",
    # Asia-Pacific
    "Japan": "Asia-Pacific",
    "China": "Asia-Pacific",
    "Hong Kong": "Asia-Pacific",
    "South Korea": "Asia-Pacific",
    "Taiwan": "Asia-Pacific",
    "Singapore": "Asia-Pacific",
    "Australia": "Asia-Pacific",
    "New Zealand": "Asia-Pacific",
    "India": "Asia-Pacific",
    "Indonesia": "Asia-Pacific",
    # Middle East & Africa
    "United Arab Emirates": "Middle East & Africa",
    "Saudi Arabia": "Middle East & Africa",
    "Israel": "Middle East & Africa",
    "South Africa": "Middle East & Africa",
    "Qatar": "Middle East & Africa",
    "Turkey": "Middle East & Africa",
}


# Line items to extract from the FinancialStatement EAV table.
# Mapping: DB line_item -> (statement_type, target_column_name)
_STMT_LINE_ITEMS: dict[str, tuple[str, str]] = {
    "Net Income": ("income_statement", "net_income"),
    "Gross Profit": ("income_statement", "gross_profit"),
    "Operating Income": ("income_statement", "operating_income"),
    "Total Assets": ("balance_sheet", "total_assets"),
    "Stockholders Equity": ("balance_sheet", "total_equity"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float(val: Any) -> float | None:
    """Coerce a DB value (Decimal / int / None) to float."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    return float(val)


def _build_ticker_map(
    session: Session, include_delisted: bool = True
) -> dict[str, str]:
    """Return {instrument_id_hex: yfinance_ticker} for instruments.

    Parameters
    ----------
    include_delisted : bool, default=True
        When ``False``, exclude instruments with a non-null ``delisted_at``.
    """
    stmt = (
        select(Instrument.id, Instrument.yfinance_ticker)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
    )
    if not include_delisted:
        stmt = stmt.where(Instrument.delisted_at.is_(None))
    rows = session.execute(stmt).all()
    return {str(r[0]): r[1] for r in rows}


def _build_ticker_rank_map(
    session: Session, include_delisted: bool = True
) -> dict[str, tuple[str, int]]:
    """Return ``{instrument_id_hex: (yfinance_ticker, currency_rank)}``.

    The currency rank is derived from :func:`~research._currency.currency_dedup_rank`
    and is used to deterministically resolve cross-listed instruments that share
    the same ``yfinance_ticker`` (e.g. LSE vs Frankfurt listings).  Lower rank
    means higher priority (USD=0 beats GBX=3 beats unknown=99).
    """
    stmt = (
        select(Instrument.id, Instrument.yfinance_ticker, Instrument.currency_code)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
    )
    if not include_delisted:
        stmt = stmt.where(Instrument.delisted_at.is_(None))
    rows = session.execute(stmt).all()
    return {
        str(r[0]): (str(r[1]), currency_dedup_rank(r[2] if r[2] is not None else None))
        for r in rows
    }


def _build_currency_map_from_instruments(session: Session) -> dict[str, str]:
    """Return {yfinance_ticker: currency_code} from the Instrument table.

    Lightweight query for price/volume normalisation when TickerProfile
    is not loaded.
    """
    rows = session.execute(
        select(Instrument.yfinance_ticker, Instrument.currency_code)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
        .where(Instrument.currency_code.isnot(None))
    ).all()
    return {str(t): str(c) for t, c in rows}


def _pivot_with_dedup(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    name: str = "",
) -> pd.DataFrame:
    """Pivot *df* after deterministically deduplicating (index, columns) pairs.

    When two rows share the same ``(index, columns)`` pair the one with the
    lower ``_ccy_rank`` value is kept (primary-currency listing wins).  A
    :mod:`warnings` warning is emitted once per call when rows are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *index*, *columns*, *values*, and ``_ccy_rank`` columns.
    name : str
        Function label used in the warning message.
    """
    n_before = len(df)
    df = df.sort_values("_ccy_rank", kind="stable")
    df = df.drop_duplicates(subset=[index, columns], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped > 0 and n_before > 0:
        dropped_pct = n_dropped / n_before
        label = f"{name}: " if name else ""
        if dropped_pct > _DEDUP_DROP_THRESHOLD_PCT:
            raise ValueError(
                f"{label}dedup dropped {n_dropped}/{n_before} "
                f"({dropped_pct:.1%}) duplicate rows; "
                "expected ≤ 5%; check upstream normalization"
            )
        logger.info(
            "dedup_dropped_duplicates",
            extra={
                # ``name`` is a reserved LogRecord attribute; expose under
                # ``dedup_name`` so structured-log handlers can index it.
                "dedup_name": name,
                "n_dropped": n_dropped,
                "n_before": n_before,
                "dropped_pct": dropped_pct,
            },
        )
    pivoted = df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc="first",
    )
    return pivoted


def _apply_delisting_returns(
    prices: pd.DataFrame,
    delistings: list[tuple[str, pd.Timestamp, float]],
) -> pd.DataFrame:
    """Append a synthetic delisting-date price row for each delisted instrument.

    For each ``(yf_ticker, delisted_at, delisting_return)`` tuple, finds the
    last known close price on or before ``delisted_at`` and adds a synthetic
    price row at ``delisted_at`` equal to ``last_price * (1 + delisting_return)``.

    When ``prices_to_returns`` is subsequently applied, this synthetic row
    produces the correct delisting return as the final observation for that
    instrument.

    Parameters
    ----------
    prices : pd.DataFrame
        dates × tickers close-price DataFrame.
    delistings : list[tuple[str, pd.Timestamp, float]]
        ``(yf_ticker, delisted_at, delisting_return)`` for each delisted
        instrument.  Instruments not in ``prices.columns`` are silently
        skipped.

    Returns
    -------
    pd.DataFrame
        Copy of ``prices`` with synthetic delisting rows appended and sorted.
    """
    if not delistings:
        return prices

    out = prices.copy()

    for yf_ticker, delisted_ts, r in delistings:
        if yf_ticker not in out.columns:
            continue

        col = out[yf_ticker].dropna()
        if col.empty:
            continue

        # Last known price on or before the delisting date.
        before = col[col.index <= delisted_ts]
        if before.empty:
            continue

        last_price = float(before.iloc[-1])
        synthetic_price = last_price * (1.0 + r)

        # Add a new index row for the delisting date if not already present.
        if delisted_ts not in out.index:
            new_row = pd.Series(dict.fromkeys(out.columns, np.nan), name=delisted_ts)
            out = pd.concat([out, new_row.to_frame().T])
            out = out.sort_index()

        # Only write synthetic price if the cell is currently NaN.
        if pd.isna(out.loc[delisted_ts, yf_ticker]):
            out.loc[delisted_ts, yf_ticker] = synthetic_price

    return out
