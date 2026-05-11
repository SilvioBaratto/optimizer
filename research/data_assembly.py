"""Assemble optimizer-ready DataFrames from database ORM rows.

This module is the glue layer between the API data layer (PostgreSQL)
and the optimizer library.  It queries the database tables and pivots /
reshapes ORM rows into the exact DataFrame shapes that
``run_full_pipeline_with_selection()`` expects.

All functions are implemented in domain-focused sub-modules under
``research.data.*``.  This file is a backward-compatible re-export facade.
"""

from research.data._container import DataAssembly, _compute_assembly_hash  # noqa: F401
from research.data._currency import (  # noqa: F401
    CURRENCY_DEDUP_PRIORITY,
    MINOR_CURRENCY_DIVISORS,
    build_currency_map,
    currency_dedup_rank,
    normalize_fundamentals,
    normalize_prices,
)
from research.data._equity import (  # noqa: F401
    _dedup_fundamentals_df,
    assemble_analyst_data,
    assemble_financial_statements,
    assemble_fundamentals,
    assemble_insider_data,
    assemble_prices,
    assemble_volumes,
)
from research.data._helpers import (  # noqa: F401
    _DEDUP_DROP_THRESHOLD_PCT,
    _STMT_LINE_ITEMS,
    _TRADING_DAYS,
    REGION_MAP,
    _apply_delisting_returns,
    _build_currency_map_from_instruments,
    _build_ticker_map,
    _build_ticker_rank_map,
    _pivot_with_dedup,
    _to_float,
)
from research.data._history import (  # noqa: F401
    assemble_delisting_returns,
    assemble_fundamental_history,
)
from research.data._macro import (  # noqa: F401
    FRED_SERIES_IDS,
    assemble_bond_observations,
    assemble_fred_series,
    assemble_macro_data,
    assemble_macro_timeseries,
    assemble_te_observations,
)
from research.data._orchestrator import assemble_all, assemble_fx_rates  # noqa: F401
from research.data._regime import (  # noqa: F401
    _FRED_REGIME_MAP,
    _REQUIRED_REGIME_COLUMNS,
    assemble_regime_data,
)
from research.data._sentiment import assemble_sentiment  # noqa: F401
