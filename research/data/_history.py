"""Historical fundamental panel and delisting return assembly.

Extracted from ``data_assembly.py``.
"""

from __future__ import annotations

import logging
import sys
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

from ._helpers import _STMT_LINE_ITEMS, _build_ticker_map, _to_float  # noqa: E402

logger = logging.getLogger(__name__)

_HISTORY_COLUMNS: list[str] = [
    "net_income",
    "gross_profit",
    "operating_income",
    "total_assets",
    "total_equity",
    "period_type",
    "asset_growth",
]


def _empty_history_df() -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=str)],
        names=["period_date", "ticker"],
    )
    return pd.DataFrame(columns=_HISTORY_COLUMNS, index=idx)


def assemble_delisting_returns(session: Session) -> dict[str, float]:
    """Build a ``{yfinance_ticker: delisting_return}`` mapping for delisted instruments.

    Used by ``run_full_pipeline()`` to apply the returns-space
    survivorship-bias correction after ``prices_to_returns()``.

    Returns
    -------
    dict[str, float]
        Empty when no delisted instruments exist.  Defaults to ``-0.30``
        when ``delisting_return`` is ``NULL`` in the database.
    """
    from app.models.universe.universe import Instrument

    rows = session.execute(
        select(
            Instrument.yfinance_ticker,
            Instrument.delisting_return,
        )
        .where(Instrument.delisted_at.isnot(None))
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
    ).all()
    return {
        str(yf_ticker): float(dr) if dr is not None else -0.30 for yf_ticker, dr in rows
    }


def assemble_fundamental_history(
    session: Session,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Build a ``(period_date, ticker)`` panel from financial_statements EAV.

    Queries all historical financial statement rows for key line items,
    pivots them into a MultiIndex panel suitable for point-in-time slicing.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    tickers : list[str] or None
        Restrict to these tickers.  ``None`` fetches all.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(period_date: pd.Timestamp, ticker: str)``.
        Columns: ``net_income``, ``gross_profit``, ``operating_income``,
        ``total_assets``, ``total_equity``, ``period_type``
        (``'annual'`` | ``'quarterly'``), ``asset_growth`` (float | NaN).

    Notes
    -----
    Financial statement monetary values (net_income, gross_profit,
    operating_income, total_assets, total_equity) are stored by yfinance
    in the company's **reporting currency** — GBP for UK-listed
    companies, USD for US companies, etc.  This is distinct from the
    listing quote currency: an LSE stock quoted in GBX (pence) still
    has balance-sheet data reported in GBP.

    Consequently, ``asset_growth`` — computed as the year-over-year ratio
    of ``total_assets`` values for the same ticker — is dimensionless
    and inherently currency-safe: the currency cancels in the division.
    No normalisation is applied or needed for this column.

    Callers combining ``total_assets`` (in reporting-currency units)
    with market data (market_cap, current_price) for cross-sectional
    ratios **must** ensure market data has been normalised from minor
    units to major units (e.g. via ``normalize_fundamentals()``) before
    computing ratios.
    """
    from app.models.market_data.yfinance_data import FinancialStatement

    ticker_map = _build_ticker_map(session)
    if tickers is not None:
        inv = {v: k for k, v in ticker_map.items()}
        allowed_ids = {inv[t] for t in tickers if t in inv}
        ticker_map = {k: v for k, v in ticker_map.items() if k in allowed_ids}

    line_item_names = list(_STMT_LINE_ITEMS.keys())

    rows = session.execute(
        select(
            FinancialStatement.instrument_id,
            FinancialStatement.period_type,
            FinancialStatement.period_date,
            FinancialStatement.line_item,
            FinancialStatement.value,
        )
        .where(FinancialStatement.line_item.in_(line_item_names))
        .where(FinancialStatement.value.isnot(None))
        .order_by(FinancialStatement.instrument_id, FinancialStatement.period_date)
    ).all()

    if not rows:
        return _empty_history_df()

    records: list[dict[str, Any]] = []
    for instrument_id, period_type, period_date, line_item, value in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        _, target_col = _STMT_LINE_ITEMS[line_item]
        records.append(
            {
                "ticker": ticker,
                "period_date": pd.Timestamp(period_date),
                "period_type": period_type,
                "target_col": target_col,
                "value": _to_float(value),
            }
        )

    if not records:
        return _empty_history_df()

    raw = pd.DataFrame(records)

    # Pivot: one row per (period_date, ticker, period_type), columns = target_col
    pivoted = raw.pivot_table(
        index=["period_date", "ticker", "period_type"],
        columns="target_col",
        values="value",
        aggfunc="first",
    )
    pivoted = pivoted.reset_index()

    # Compute asset_growth per ticker from annual Total Assets (YoY)
    pivoted["asset_growth"] = np.nan
    annual_mask = pivoted["period_type"] == "annual"
    if annual_mask.any() and "total_assets" in pivoted.columns:
        annual = pivoted.loc[annual_mask].sort_values(["ticker", "period_date"])
        growth = annual.groupby("ticker")["total_assets"].pct_change()
        pivoted.loc[annual.index, "asset_growth"] = growth.values

    # Build MultiIndex (period_date, ticker)
    pivoted = pivoted.set_index(["period_date", "ticker"]).sort_index()

    # Ensure all expected columns exist
    for col in [
        "net_income",
        "gross_profit",
        "operating_income",
        "total_assets",
        "total_equity",
    ]:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    logger.info(
        "Assembled fundamental history panel: %d rows, %d tickers.",
        len(pivoted),
        pivoted.index.get_level_values("ticker").nunique(),
    )
    return pivoted
