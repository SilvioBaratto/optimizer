"""Gate tests for research/optimization/__init__.py API contract.

Convention: The optimization sub-package intentionally exposes private
symbols (underscore-prefixed) in ``__all__``.  Unlike ``research.data``,
which forms a stable public data-access API (zero private exports), the
``research.optimization`` sub-package is a research toolkit where
test-patching of internal constants and helpers is a first-class design
goal.  Private exports are therefore *permitted and required* here.

These tests enforce:
- every symbol in ``__all__`` is importable from ``research.optimization``
- ``__all__`` is sorted alphabetically
- every public symbol from ``_config.py``, ``_retighten.py``, and
  ``_rebalance.py`` is re-exported (no accidental omissions)
- no symbol appears twice in ``__all__``
- the module has a docstring that documents the private-export convention
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Constants — expected symbols per source module
# ---------------------------------------------------------------------------

_CONFIG_SYMBOLS: frozenset[str] = frozenset(
    {
        "_MAX_REGION_WEIGHT",
        "_REGION_MAP",
        "_SECTOR_FLOORS",
        "_SOLVER_PARAMS",
        "_make_builder",
        "_make_opt_config",
        "_resolve_min_weights",
        "_select_optimizer",
        "build_research_optimizer",
    }
)

_RETIGHTEN_SYMBOLS: frozenset[str] = frozenset(
    {
        "_SHRINK_FACTOR",
        "_TOP4_THRESHOLD",
        "_TOP_N",
        "_TRADING_DAYS_PER_YEAR",
        "_annualized_sharpe",
        "_retighten_error_message",
        "_solve_with_retighten",
        "_split_into_subperiods",
        "_top4_weight",
    }
)

_REBALANCE_SYMBOLS: frozenset[str] = frozenset(
    {
        "_HOCKEY_STICK_NEG_THRESHOLD",
        "_HOCKEY_STICK_POS_THRESHOLD",
        "_decide_rebalance",
        "_hockey_stick_warn",
    }
)

_ALL_REQUIRED: frozenset[str] = (
    _CONFIG_SYMBOLS | _RETIGHTEN_SYMBOLS | _REBALANCE_SYMBOLS
)


# ---------------------------------------------------------------------------
# 1. Import sanity — no circular imports
# ---------------------------------------------------------------------------


def test_import_research_optimization_no_circular_import() -> None:
    """Importing research.optimization must not raise ImportError."""
    mod = importlib.import_module("research.optimization")
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. Every symbol in __all__ is importable from research.optimization
# ---------------------------------------------------------------------------


def test_every_all_symbol_is_importable() -> None:
    """Each name in __all__ must be accessible as an attribute of the module."""
    import research.optimization as m

    missing = [name for name in m.__all__ if not hasattr(m, name)]
    assert missing == [], f"Symbols in __all__ not accessible on module: {missing}"


def test_import_star_works() -> None:
    """``from research.optimization import *`` must succeed."""
    mod = importlib.import_module("research.optimization")
    ns: dict[str, object] = {}
    exec("from research.optimization import *", ns)  # noqa: S102
    for name in mod.__all__:
        assert name in ns, f"{name!r} not exported by import *"


# ---------------------------------------------------------------------------
# 3. __all__ is sorted alphabetically
# ---------------------------------------------------------------------------


def test_all_is_sorted_alphabetically() -> None:
    """__all__ must be in ascending lexicographic order."""
    import research.optimization as m

    expected = sorted(m.__all__)
    actual = list(m.__all__)
    assert actual == expected, (
        f"__all__ is not sorted. Expected:\n  {expected}\nGot:\n  {actual}"
    )


# ---------------------------------------------------------------------------
# 4. Every required public symbol from sub-modules is re-exported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", sorted(_CONFIG_SYMBOLS))
def test_config_symbol_in_all(symbol: str) -> None:
    """Every symbol from _config.py must appear in __all__."""
    import research.optimization as m

    assert symbol in m.__all__, (
        f"{symbol!r} from _config.py missing from research.optimization.__all__"
    )


@pytest.mark.parametrize("symbol", sorted(_RETIGHTEN_SYMBOLS))
def test_retighten_symbol_in_all(symbol: str) -> None:
    """Every symbol from _retighten.py must appear in __all__."""
    import research.optimization as m

    assert symbol in m.__all__, (
        f"{symbol!r} from _retighten.py missing from research.optimization.__all__"
    )


@pytest.mark.parametrize("symbol", sorted(_REBALANCE_SYMBOLS))
def test_rebalance_symbol_in_all(symbol: str) -> None:
    """Every symbol from _rebalance.py must appear in __all__."""
    import research.optimization as m

    assert symbol in m.__all__, (
        f"{symbol!r} from _rebalance.py missing from research.optimization.__all__"
    )


def test_no_required_symbol_missing() -> None:
    """__all__ must contain every required symbol (bulk check)."""
    import research.optimization as m

    missing = _ALL_REQUIRED - set(m.__all__)
    assert not missing, f"Required symbols missing from __all__: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 5. No symbol appears twice in __all__
# ---------------------------------------------------------------------------


def test_no_duplicates_in_all() -> None:
    """__all__ must not contain duplicate entries."""
    import research.optimization as m

    duplicates = [name for name in m.__all__ if list(m.__all__).count(name) > 1]
    assert duplicates == [], f"Duplicate entries in __all__: {duplicates}"


# ---------------------------------------------------------------------------
# 6. The module has a docstring documenting the private-export convention
# ---------------------------------------------------------------------------


def test_module_has_docstring() -> None:
    """research.optimization must have a non-empty module docstring."""
    import research.optimization as m

    assert m.__doc__ is not None and m.__doc__.strip(), (
        "research/optimization/__init__.py must have a module docstring"
    )


def test_module_docstring_mentions_private_export_convention() -> None:
    """The module docstring must explain why private symbols are in __all__."""
    import research.optimization as m

    doc = (m.__doc__ or "").lower()
    # The docstring should mention test-patching or private exports
    mentions_convention = any(
        keyword in doc
        for keyword in ("patch", "private", "test-patch", "underscore", "internal")
    )
    assert mentions_convention, (
        "Module docstring must document the intentional private-export convention "
        "(mention 'patch', 'private', 'test-patch', 'underscore', or 'internal'). "
        f"Current docstring: {m.__doc__!r}"
    )


# ---------------------------------------------------------------------------
# 7. build_research_optimizer is callable (smoke test for sole public symbol)
# ---------------------------------------------------------------------------


def test_build_research_optimizer_is_callable() -> None:
    """build_research_optimizer must be a callable."""
    from research.optimization import build_research_optimizer

    assert callable(build_research_optimizer)


# ---------------------------------------------------------------------------
# 9. Private constants have expected types
# ---------------------------------------------------------------------------


def test_region_map_is_dict() -> None:
    from research.optimization import _REGION_MAP

    assert isinstance(_REGION_MAP, dict)
    assert len(_REGION_MAP) > 0


def test_sector_floors_is_dict() -> None:
    from research.optimization import _SECTOR_FLOORS

    assert isinstance(_SECTOR_FLOORS, dict)
    assert len(_SECTOR_FLOORS) == 0


def test_max_region_weight_is_float() -> None:
    from research.optimization import _MAX_REGION_WEIGHT

    assert isinstance(_MAX_REGION_WEIGHT, float)
    assert 0.0 < _MAX_REGION_WEIGHT <= 1.0


def test_shrink_factor_is_float() -> None:
    from research.optimization import _SHRINK_FACTOR

    assert isinstance(_SHRINK_FACTOR, float)
    assert 0.0 < _SHRINK_FACTOR < 1.0


def test_top4_threshold_is_float() -> None:
    from research.optimization import _TOP4_THRESHOLD

    assert isinstance(_TOP4_THRESHOLD, float)
    assert 0.0 < _TOP4_THRESHOLD <= 1.0


def test_top_n_is_int() -> None:
    from research.optimization import _TOP_N

    assert isinstance(_TOP_N, int)
    assert _TOP_N > 0


def test_trading_days_per_year_is_int() -> None:
    from research.optimization import _TRADING_DAYS_PER_YEAR

    assert isinstance(_TRADING_DAYS_PER_YEAR, int)
    assert _TRADING_DAYS_PER_YEAR == 252
