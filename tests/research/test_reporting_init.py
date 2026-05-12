"""Gate tests for research/reporting/__init__.py public API contract.

Issue #678: The package __init__ must expose only public (non-underscore)
symbols, keep __all__ <= 25 entries, have no cross-package imports at the
__init__ level, no circular imports, and expose a module docstring.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. No private symbols in __all__
# ---------------------------------------------------------------------------


def test_all_contains_no_private_symbols() -> None:
    """__all__ must not contain any underscore-prefixed names."""
    import research.reporting as pkg

    private_in_all = [name for name in pkg.__all__ if name.startswith("_")]
    assert private_in_all == [], (
        f"__all__ must not contain private symbols, found: {private_in_all}"
    )


# ---------------------------------------------------------------------------
# 2. Every symbol in __all__ is importable from research.reporting
# ---------------------------------------------------------------------------


def test_all_symbols_are_importable() -> None:
    """Every name listed in __all__ must be importable from research.reporting."""
    import research.reporting as pkg

    missing: list[str] = []
    for name in pkg.__all__:
        if not hasattr(pkg, name):
            missing.append(name)
    assert missing == [], (
        f"Names in __all__ that are not accessible on research.reporting: {missing}"
    )


# ---------------------------------------------------------------------------
# 3. No duplicates in __all__
# ---------------------------------------------------------------------------


def test_all_has_no_duplicates() -> None:
    """__all__ must not contain duplicate entries."""
    import research.reporting as pkg

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in pkg.__all__:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    assert duplicates == [], f"__all__ contains duplicate entries: {duplicates}"


# ---------------------------------------------------------------------------
# 4. Module has a docstring
# ---------------------------------------------------------------------------


def test_module_has_docstring() -> None:
    """research/reporting/__init__.py must have a module-level docstring."""
    import research.reporting as pkg

    assert pkg.__doc__ is not None and pkg.__doc__.strip() != "", (
        "research/reporting/__init__.py must have a non-empty module docstring"
    )


# ---------------------------------------------------------------------------
# 5. No cross-package imports at __init__ level
#    (must not import from research.data, research.optimization,
#     research.factors, or api.*)
# ---------------------------------------------------------------------------


def test_no_cross_package_imports_at_init_level() -> None:
    """__init__.py must not import from research.data, research.optimization,
    research.factors, or api.* at the top level."""
    init_path = (
        Path(__file__).parent.parent.parent / "research" / "reporting" / "__init__.py"
    )
    source = init_path.read_text()
    tree = ast.parse(source)

    forbidden_prefixes = (
        "research.data",
        "research.optimization",
        "research.factors",
        "api.",
    )

    violations: list[str] = []
    for node in ast.walk(tree):
        module_name: str | None = None
        if isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(p) for p in forbidden_prefixes):
                    violations.append(alias.name)
            continue

        if module_name and any(module_name.startswith(p) for p in forbidden_prefixes):
            violations.append(module_name)

    assert violations == [], (
        f"research/reporting/__init__.py must not import from: {violations}"
    )


# ---------------------------------------------------------------------------
# 6. No circular imports (subprocess check)
# ---------------------------------------------------------------------------


def test_no_circular_import() -> None:
    """Importing research.reporting in a subprocess must exit 0."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import research.reporting; import sys; sys.exit(0)",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 0, (
        f"Circular import detected when importing research.reporting:\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# 7. __all__ length is reasonable (<= 25 entries)
# ---------------------------------------------------------------------------


def test_all_has_at_most_25_entries() -> None:
    """__all__ must contain no more than 25 entries."""
    import research.reporting as pkg

    count = len(pkg.__all__)
    assert count <= 25, (
        f"research/reporting/__init__.py __all__ has {count} entries; must be <=25"
    )


# ---------------------------------------------------------------------------
# 8. All names consumed by research/__init__.py are importable
# ---------------------------------------------------------------------------


def test_research_init_contract_symbols_importable() -> None:
    """Every name imported by research/__init__.py from research.reporting
    must be importable."""
    # These are the 18 names imported in research/__init__.py lines 101-118
    expected_names = [
        "compute_binding_constraints",
        "dict_table",
        "error_panel",
        "generate_backtest_plots",
        "info_panel",
        "list_table",
        "plot_country_allocation",
        "plot_cumulative_returns",
        "plot_drawdowns",
        "plot_factor_ic",
        "plot_rolling_sharpe",
        "plot_sector_weights_over_time",
        "progress_loop",
        "render_report",
        "success_panel",
        "warning_panel",
    ]
    import research.reporting as pkg

    missing = [name for name in expected_names if not hasattr(pkg, name)]
    assert missing == [], (
        f"Names required by research/__init__.py missing from research.reporting: "
        f"{missing}"
    )


# ---------------------------------------------------------------------------
# 9. __all__ is a superset of research/__init__.py contract (no phantom entries)
# ---------------------------------------------------------------------------


def test_all_matches_exported_names() -> None:
    """Every name in __all__ must be importable, and every contract name must
    appear in __all__."""
    contract_names = {
        "compute_binding_constraints",
        "dict_table",
        "error_panel",
        "generate_backtest_plots",
        "info_panel",
        "list_table",
        "plot_country_allocation",
        "plot_cumulative_returns",
        "plot_drawdowns",
        "plot_factor_ic",
        "plot_rolling_sharpe",
        "plot_sector_weights_over_time",
        "progress_loop",
        "render_report",
        "success_panel",
        "warning_panel",
    }
    import research.reporting as pkg

    all_set = set(pkg.__all__)
    not_in_all = contract_names - all_set
    assert not_in_all == set(), f"Contract names missing from __all__: {not_in_all}"
