"""Tests for research/data/__init__.py public API contract.

Issue #663: The package __init__ must expose only public (non-underscore)
symbols, keep __all__ ≤30 lines, and cause no circular imports.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Import sanity — no circular imports
# ---------------------------------------------------------------------------


def test_import_research_data_no_circular_import() -> None:
    """Importing research.data must not raise ImportError or circular-import."""
    mod = importlib.import_module("research.data")
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. Required public symbols are accessible at the package level
# ---------------------------------------------------------------------------


def test_data_assembly_importable() -> None:
    from research.data import DataAssembly

    assert inspect.isclass(DataAssembly)


def test_assemble_all_importable() -> None:
    from research.data import assemble_all

    assert callable(assemble_all)


def test_region_map_importable() -> None:
    from research.data import REGION_MAP

    assert isinstance(REGION_MAP, dict)
    assert len(REGION_MAP) > 0


def test_fred_series_ids_importable() -> None:
    from research.data import FRED_SERIES_IDS

    assert isinstance(FRED_SERIES_IDS, (list, dict, tuple))


def test_currency_dedup_priority_importable() -> None:
    from research.data import CURRENCY_DEDUP_PRIORITY

    assert isinstance(CURRENCY_DEDUP_PRIORITY, (list, tuple, dict))


def test_minor_currency_divisors_importable() -> None:
    from research.data import MINOR_CURRENCY_DIVISORS

    assert isinstance(MINOR_CURRENCY_DIVISORS, dict)


def test_build_currency_map_importable() -> None:
    from research.data import build_currency_map

    assert callable(build_currency_map)


def test_currency_dedup_rank_importable() -> None:
    from research.data import currency_dedup_rank

    assert callable(currency_dedup_rank)


def test_normalize_fundamentals_importable() -> None:
    from research.data import normalize_fundamentals

    assert callable(normalize_fundamentals)


def test_normalize_prices_importable() -> None:
    from research.data import normalize_prices

    assert callable(normalize_prices)


# ---------------------------------------------------------------------------
# 3. __all__ must contain no underscore-prefixed symbols
# ---------------------------------------------------------------------------


def test_all_contains_no_private_symbols() -> None:
    import research.data as data_pkg

    private_in_all = [name for name in data_pkg.__all__ if name.startswith("_")]
    assert private_in_all == [], (
        f"__all__ must not contain private symbols, found: {private_in_all}"
    )


# ---------------------------------------------------------------------------
# 4. __all__ size constraint: ≤30 entries
# ---------------------------------------------------------------------------


def test_all_has_at_most_30_entries() -> None:
    import research.data as data_pkg

    count = len(data_pkg.__all__)
    assert count <= 30, (
        f"research/data/__init__.py __all__ has {count} entries; must be ≤30"
    )


# ---------------------------------------------------------------------------
# 5. No underscore symbols leak into the module's public namespace
#    (i.e. they must NOT be importable via `from research.data import _X`)
# ---------------------------------------------------------------------------


def test_private_helpers_not_in_data_namespace() -> None:
    """Private symbols removed from data/__init__ must not be re-exported."""
    import research.data as data_pkg

    private_symbols = [
        "_STMT_LINE_ITEMS",
        "_TRADING_DAYS",
        "_build_ticker_map",
        "_pivot_with_dedup",
        "_to_float",
        "_apply_delisting_returns",
        "_build_ticker_rank_map",
        "_DEDUP_DROP_THRESHOLD_PCT",
        "_FRED_REGIME_MAP",
        "_REQUIRED_REGIME_COLUMNS",
        "_compute_assembly_hash",
        "_build_currency_map_from_instruments",
    ]
    leaked = [sym for sym in private_symbols if hasattr(data_pkg, sym)]
    assert leaked == [], (
        f"Private symbols must not be accessible from research.data: {leaked}"
    )


# ---------------------------------------------------------------------------
# 6. Line count of the __init__.py source file is ≤30
# ---------------------------------------------------------------------------


def test_init_file_line_count_le_30() -> None:
    import research.data as data_pkg

    init_path = data_pkg.__file__
    assert init_path is not None
    with Path(init_path).open() as fh:
        lines = fh.readlines()
    line_count = len(lines)
    assert line_count <= 30, (
        f"research/data/__init__.py has {line_count} lines; must be ≤30"
    )
