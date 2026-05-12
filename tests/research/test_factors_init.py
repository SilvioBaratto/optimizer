"""Gate tests for research/factors/__init__.py public API contract.

Issue #676: Audit factors/__init__.py re-exports.
The package __init__ must expose the three named re-exports, keep __all__
consistent with the imports, and introduce no upstream coupling to
research.data or api layers.
"""

from __future__ import annotations

import importlib
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Import sanity — no circular imports
# ---------------------------------------------------------------------------


def test_import_research_factors_no_circular_import() -> None:
    """Importing research.factors must not raise ImportError or circular-import."""
    mod = importlib.import_module("research.factors")
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. Required symbols are accessible at the package level
# ---------------------------------------------------------------------------


def test_build_factor_scores_history_importable() -> None:
    from research.factors import build_factor_scores_history

    assert callable(build_factor_scores_history)


def test_validate_factors_importable() -> None:
    from research.factors import validate_factors

    assert callable(validate_factors)


def test_slice_fundamentals_at_importable() -> None:
    """_slice_fundamentals_at is intentionally re-exported for test consumers."""
    from research.factors import _slice_fundamentals_at

    assert callable(_slice_fundamentals_at)


# ---------------------------------------------------------------------------
# 3. __all__ must contain exactly the three declared re-exports
# ---------------------------------------------------------------------------

_EXPECTED_ALL = {
    "_slice_fundamentals_at",
    "build_factor_scores_history",
    "validate_factors",
}


def test_all_contains_required_public_functions() -> None:
    import research.factors as factors_pkg

    for name in ("build_factor_scores_history", "validate_factors"):
        assert name in factors_pkg.__all__, (
            f"'{name}' must be present in research.factors.__all__"
        )


def test_all_contains_slice_fundamentals_at() -> None:
    """_slice_fundamentals_at is intentionally re-exported in __all__."""
    import research.factors as factors_pkg

    assert "_slice_fundamentals_at" in factors_pkg.__all__, (
        "_slice_fundamentals_at must be in __all__ — it is an intentional re-export "
        "for downstream test consumers and the backward-compat shim."
    )


def test_all_matches_declared_exports_exactly() -> None:
    """__all__ must contain exactly the three named re-exports — no phantom entries."""
    import research.factors as factors_pkg

    actual = set(factors_pkg.__all__)
    assert actual == _EXPECTED_ALL, (
        f"__all__ mismatch.\n"
        f"  Extra entries: {actual - _EXPECTED_ALL}\n"
        f"  Missing entries: {_EXPECTED_ALL - actual}"
    )


# ---------------------------------------------------------------------------
# 4. Every symbol in __all__ must be importable from research.factors
# ---------------------------------------------------------------------------


def test_all_symbols_importable_from_package() -> None:
    import research.factors as factors_pkg

    for name in factors_pkg.__all__:
        assert hasattr(factors_pkg, name), (
            f"'{name}' listed in __all__ but not accessible from research.factors"
        )


# ---------------------------------------------------------------------------
# 5. No upstream coupling to research.data or api layers
# ---------------------------------------------------------------------------


def test_no_upstream_import_from_research_data() -> None:
    """research/factors/__init__.py must not import from research.data."""
    import research.factors as factors_pkg

    init_path = factors_pkg.__file__
    assert init_path is not None
    source = Path(init_path).read_text()

    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("from research.data"), (
            f"Upstream coupling detected in research/factors/__init__.py: {line!r}"
        )
        assert not stripped.startswith("from api"), (
            f"Cross-layer coupling detected in research/factors/__init__.py: {line!r}"
        )


# ---------------------------------------------------------------------------
# 6. Module has a docstring
# ---------------------------------------------------------------------------


def test_module_has_docstring() -> None:
    import research.factors as factors_pkg

    assert factors_pkg.__doc__ is not None and factors_pkg.__doc__.strip(), (
        "research/factors/__init__.py must have a non-empty module docstring"
    )


# ---------------------------------------------------------------------------
# 7. __all__ has no duplicate entries
# ---------------------------------------------------------------------------


def test_all_has_no_duplicates() -> None:
    import research.factors as factors_pkg

    all_list = factors_pkg.__all__
    assert len(all_list) == len(set(all_list)), (
        f"__all__ contains duplicate entries: "
        f"{[n for n in all_list if all_list.count(n) > 1]}"
    )
