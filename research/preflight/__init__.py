"""Preflight DB health checks for the research pipeline.

Re-exports from ``_checks.py`` — the canonical location for all preflight
check functions and constants.
"""

from research.preflight._checks import (
    _KNOWN_MAJOR_CURRENCIES,
    _MIN_INSTRUMENTS,
    _MIN_PRICE_TICKERS,
    _PRICE_COVERAGE_WINDOW_DAYS,
    _REQUIRED_FRED_SERIES,
    Severity,
    _check_country_coverage,
    _check_fred_freshness,
    _check_fx_coverage,
    _check_price_coverage,
    _check_price_staleness,
    _check_universe_coverage,
    run_db_preflight,
)

__all__ = [
    "_KNOWN_MAJOR_CURRENCIES",
    "_MIN_INSTRUMENTS",
    "_MIN_PRICE_TICKERS",
    "_PRICE_COVERAGE_WINDOW_DAYS",
    "_REQUIRED_FRED_SERIES",
    "Severity",
    "_check_country_coverage",
    "_check_fred_freshness",
    "_check_fx_coverage",
    "_check_price_coverage",
    "_check_price_staleness",
    "_check_universe_coverage",
    "run_db_preflight",
]
