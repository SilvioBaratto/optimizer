"""Research module: factor history building and validation utilities."""

from research._factors import (
    _slice_fundamentals_at,
    build_factor_scores_history,
    validate_factors,
)
from research.data import (  # noqa: F401
    _STMT_LINE_ITEMS,
    _TRADING_DAYS,
    CURRENCY_DEDUP_PRIORITY,
    MINOR_CURRENCY_DIVISORS,
    REGION_MAP,
    _apply_delisting_returns,
    _build_ticker_map,
    _build_ticker_rank_map,
    _pivot_with_dedup,
    _to_float,
    build_currency_map,
    currency_dedup_rank,
    normalize_fundamentals,
    normalize_prices,
)
from research.persistence import (  # noqa: F401
    _diff_from_default,
    _flatten_metrics,
    persist_research_run,
)

# Compat shims — re-export preflight and persistence APIs from their new
# canonical locations so that `from research.preflight import ...` and
# `from research.persistence import ...` resolve through the package.
# Original `_preflight.py` and `_persistence.py` files remain in place
# for backward compat with `from research._preflight import X` importers.
from research.preflight import (  # noqa: F401
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
from research.returns import (  # noqa: F401
    DEFAULT_TAX_RATE,
    apply_fx_to_prices,
    build_return_preprocessing_pipeline,
    compute_after_tax_returns,
)

__all__ = [
    "_slice_fundamentals_at",
    "build_factor_scores_history",
    "validate_factors",
]
