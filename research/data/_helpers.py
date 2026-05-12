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

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

logger = logging.getLogger(__name__)

__all__ = [
    "REGION_MAP",
    "_DEDUP_DROP_THRESHOLD_PCT",
    "_STMT_LINE_ITEMS",
    "_TRADING_DAYS",
    "_build_ticker_map",
    "_pivot_with_dedup",
    "_to_float",
]

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
    from app.models.universe.universe import Instrument

    stmt = (
        select(Instrument.id, Instrument.yfinance_ticker)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
    )
    if not include_delisted:
        stmt = stmt.where(Instrument.delisted_at.is_(None))
    rows = session.execute(stmt).all()
    return {str(r[0]): r[1] for r in rows}


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
