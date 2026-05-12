"""Assemble optimizer-ready DataFrames from database ORM rows.

This module is the glue layer between the API data layer (PostgreSQL)
and the optimizer library.  It queries the database tables and pivots /
reshapes ORM rows into the exact DataFrame shapes that
``run_full_pipeline_with_selection()`` expects.

All functions are implemented in domain-focused sub-modules under
``research.data.*``.  This file is a backward-compatible re-export facade.
"""

__all__ = [
    "CURRENCY_DEDUP_PRIORITY",
    "FRED_SERIES_IDS",
    "MINOR_CURRENCY_DIVISORS",
    "REGION_MAP",
    "_DEDUP_DROP_THRESHOLD_PCT",
    "_FRED_REGIME_MAP",
    "_REQUIRED_REGIME_COLUMNS",
    "_STMT_LINE_ITEMS",
    "_TRADING_DAYS",
    "DataAssembly",
    "_apply_delisting_returns",
    "_build_currency_map_from_instruments",
    "_build_ticker_map",
    "_build_ticker_rank_map",
    "_compute_assembly_hash",
    "_dedup_fundamentals_df",
    "_pivot_with_dedup",
    "_to_float",
    "assemble_all",
    "assemble_analyst_data",
    "assemble_bond_observations",
    "assemble_delisting_returns",
    "assemble_financial_statements",
    "assemble_fred_series",
    "assemble_fundamental_history",
    "assemble_fundamentals",
    "assemble_fx_rates",
    "assemble_insider_data",
    "assemble_macro_data",
    "assemble_macro_timeseries",
    "assemble_prices",
    "assemble_regime_data",
    "assemble_sentiment",
    "assemble_te_observations",
    "assemble_volumes",
    "build_currency_map",
    "currency_dedup_rank",
    "normalize_fundamentals",
    "normalize_prices",
]

from research.data._container import DataAssembly, _compute_assembly_hash
from research.data._currency import (
    CURRENCY_DEDUP_PRIORITY,
    MINOR_CURRENCY_DIVISORS,
    build_currency_map,
    currency_dedup_rank,
    normalize_fundamentals,
    normalize_prices,
)
from research.data._equity import (
    _apply_delisting_returns,
    _build_currency_map_from_instruments,
    _build_ticker_rank_map,
    _dedup_fundamentals_df,
    assemble_analyst_data,
    assemble_financial_statements,
    assemble_fundamentals,
    assemble_insider_data,
    assemble_prices,
    assemble_volumes,
)
from research.data._helpers import (
    _DEDUP_DROP_THRESHOLD_PCT,
    _STMT_LINE_ITEMS,
    _TRADING_DAYS,
    REGION_MAP,
    _build_ticker_map,
    _pivot_with_dedup,
    _to_float,
)
from research.data._history import (
    assemble_delisting_returns,
    assemble_fundamental_history,
)
from research.data._macro import (
    FRED_SERIES_IDS,
    assemble_bond_observations,
    assemble_fred_series,
    assemble_macro_data,
    assemble_macro_timeseries,
    assemble_te_observations,
)
from research.data._orchestrator import assemble_all, assemble_fx_rates
from research.data._regime import (
    _FRED_REGIME_MAP,
    _REQUIRED_REGIME_COLUMNS,
    assemble_regime_data,
)
from research.data._sentiment import assemble_sentiment
