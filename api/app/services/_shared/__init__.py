"""Shared Services.

Cross-cutting service helpers and utilities.
"""

# Import order matters: _benchmark_bootstrap → reference_index_seeder →
# yfinance_data_service → needs ProgressCallback from _progress and
# has_sufficient_history from trading_calendar.  Load those before
# _benchmark_bootstrap so the symbols exist when the transitive import runs.
# ruff: noqa: I001
from app.services._shared._json_safe import safe_float
from app.services._shared._price_fetcher import fetch_close_prices
from app.services._shared._progress import (
    ProgressCallback,
    _noop,
    make_cancellable_progress,
    make_progress,
)
from app.services._shared._sector_resolver import (
    UNCLASSIFIED,
    resolve_sector_map,
)
from app.services._shared.notifications import notify_failure
from app.services._shared.trading_calendar import (
    EXCHANGE_NAME_TO_MIC,
    get_expected_trading_sessions,
    has_sufficient_history,
    parse_period_years,
)
from app.services._shared._benchmark_bootstrap import (
    SessionFactory,
    bootstrap_benchmarks,
)

__all__ = [
    "EXCHANGE_NAME_TO_MIC",
    "UNCLASSIFIED",
    "ProgressCallback",
    "SessionFactory",
    "_noop",
    "bootstrap_benchmarks",
    "fetch_close_prices",
    "get_expected_trading_sessions",
    "has_sufficient_history",
    "make_cancellable_progress",
    "make_progress",
    "notify_failure",
    "parse_period_years",
    "resolve_sector_map",
    "safe_float",
]
