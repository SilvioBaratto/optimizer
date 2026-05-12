"""Equity assembly facade — re-exports from split sub-modules.

All public names previously defined here are preserved so that
``research.data.__init__`` and ``research.data_assembly`` continue to
import from this module without modification.
"""

from __future__ import annotations

from ._equity_fundamentals import (
    _FUNDAMENTAL_COLUMNS,
    _compute_asset_growth_from_statements,
    _dedup_fundamentals_df,
    _enrich_from_financial_statements,
    assemble_analyst_data,
    assemble_financial_statements,
    assemble_fundamentals,
    assemble_insider_data,
)
from ._equity_prices import (
    _apply_delisting_returns,
    _build_currency_map_from_instruments,
    _build_ticker_rank_map,
    assemble_prices,
    assemble_volumes,
)

__all__ = [
    "_FUNDAMENTAL_COLUMNS",
    "_apply_delisting_returns",
    "_build_currency_map_from_instruments",
    "_build_ticker_rank_map",
    "_compute_asset_growth_from_statements",
    "_dedup_fundamentals_df",
    "_enrich_from_financial_statements",
    "assemble_analyst_data",
    "assemble_financial_statements",
    "assemble_fundamentals",
    "assemble_insider_data",
    "assemble_prices",
    "assemble_volumes",
]
