"""Data assembly layer — DB → DataFrames glue for the optimizer pipeline.

Re-exports from sub-modules.  All assembly functions accept a synchronous
SQLAlchemy ``Session`` and return pandas DataFrames ready for the optimizer
library.
"""

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
    _apply_delisting_returns,
    _build_currency_map_from_instruments,
    _build_ticker_map,
    _build_ticker_rank_map,
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
