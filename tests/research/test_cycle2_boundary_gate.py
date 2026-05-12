"""Cycle 2 boundary gate — circular imports, lint, mypy, size.

Issue #679: Cross-cutting acceptance gate for the three Cycle 2 sub-packages:
``research/optimization/``, ``research/factors/``, ``research/reporting/``.

Covers:
- A. Circular import checks via subprocess (isolated from in-process side effects)
- B. ``__init__.py`` size limits (per-package thresholds)
- C. Private-symbol convention enforcement
- D. ``__all__`` completeness — no phantom entries
- E. Lint check via ruff
- F. Import boundary — no cross-package pollution (AST-level)
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent

_INIT_FILES = {
    "research.optimization": _REPO_ROOT / "research" / "optimization" / "__init__.py",
    "research.factors": _REPO_ROOT / "research" / "factors" / "__init__.py",
    "research.reporting": _REPO_ROOT / "research" / "reporting" / "__init__.py",
}

# Per-package __init__.py line-count thresholds (actual counts + headroom).
# optimization: 71 lines → ≤80; factors: 13 lines → ≤25; reporting: 40 lines → ≤55
_SIZE_LIMITS: dict[str, int] = {
    "research.optimization": 80,
    "research.factors": 25,
    "research.reporting": 55,
}

# Forbidden import prefixes for all three __init__.py files.
_FORBIDDEN_IMPORT_PREFIXES = (
    "research.data",
    "research.pipeline",
    "research.cli",
    "api.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_import_subprocess(import_stmt: str) -> subprocess.CompletedProcess[str]:
    """Run a Python import in an isolated subprocess."""
    return subprocess.run(
        [sys.executable, "-c", import_stmt],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


def _check_no_circular_import(pkg: str) -> None:
    """Assert that importing *pkg* in a fresh subprocess exits 0."""
    result = _run_import_subprocess(f"import {pkg}; import sys; sys.exit(0)")
    assert result.returncode == 0, (
        f"Circular import or ImportError detected when importing {pkg!r}:\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )


def _ast_imports(init_path: Path) -> list[str]:
    """Return all imported module names found in *init_path* via AST."""
    source = init_path.read_text()
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


# ---------------------------------------------------------------------------
# A. Circular import checks (subprocess — mandatory isolation)
# ---------------------------------------------------------------------------


class TestCircularImports:
    def test_import_research_optimization_no_circular(self) -> None:
        """research.optimization must import cleanly in a fresh process."""
        _check_no_circular_import("research.optimization")

    def test_import_research_factors_no_circular(self) -> None:
        """research.factors must import cleanly in a fresh process."""
        _check_no_circular_import("research.factors")

    def test_import_research_reporting_no_circular(self) -> None:
        """research.reporting must import cleanly in a fresh process."""
        _check_no_circular_import("research.reporting")

    def test_import_all_three_together_no_circular(self) -> None:
        """All three packages imported together must not cause cross-package cycles."""
        stmt = (
            "import research.optimization; "
            "import research.factors; "
            "import research.reporting; "
            "import sys; sys.exit(0)"
        )
        result = _run_import_subprocess(stmt)
        assert result.returncode == 0, (
            "Cross-package circular import detected:\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# B. __init__.py size limits (per-package thresholds)
# ---------------------------------------------------------------------------


class TestInitFileSizeLimits:
    @pytest.mark.parametrize("pkg", list(_SIZE_LIMITS))
    def test_init_file_within_size_limit(self, pkg: str) -> None:
        """Each __init__.py must not exceed its per-package line-count threshold."""
        init_path = _INIT_FILES[pkg]
        lines = init_path.read_text().splitlines()
        limit = _SIZE_LIMITS[pkg]
        assert len(lines) <= limit, (
            f"{init_path.relative_to(_REPO_ROOT)} has {len(lines)} lines; "
            f"limit for {pkg!r} is ≤{limit}"
        )


# ---------------------------------------------------------------------------
# C. Private-symbol convention enforcement
# ---------------------------------------------------------------------------


class TestPrivateSymbolConvention:
    def test_optimization_all_is_non_empty(self) -> None:
        """research.optimization.__all__ must be non-empty."""
        import research.optimization as m

        assert len(m.__all__) > 0, "research.optimization.__all__ must not be empty"

    def test_optimization_build_research_optimizer_in_all(self) -> None:
        """build_research_optimizer must be present in research.optimization.__all__."""
        import research.optimization as m

        assert "build_research_optimizer" in m.__all__

    def test_factors_build_factor_scores_history_in_all(self) -> None:
        """build_factor_scores_history must appear in research.factors.__all__."""
        import research.factors as m

        assert "build_factor_scores_history" in m.__all__

    def test_factors_validate_factors_in_all(self) -> None:
        """validate_factors must appear in research.factors.__all__."""
        import research.factors as m

        assert "validate_factors" in m.__all__

    def test_reporting_all_has_zero_private_symbols(self) -> None:
        """research.reporting.__all__ must contain NO underscore-prefixed names."""
        import research.reporting as m

        private = [name for name in m.__all__ if name.startswith("_")]
        assert private == [], (
            f"research.reporting.__all__ must expose only public symbols; "
            f"found private entries: {private}"
        )


# ---------------------------------------------------------------------------
# D. __all__ completeness — no phantom entries
# ---------------------------------------------------------------------------


class TestAllCompleteness:
    @pytest.mark.parametrize("pkg", list(_INIT_FILES))
    def test_every_all_entry_is_accessible(self, pkg: str) -> None:
        """Every name in __all__ must be reachable via getattr(module, name)."""
        import importlib

        m = importlib.import_module(pkg)
        phantom = [name for name in m.__all__ if not hasattr(m, name)]
        assert phantom == [], (
            f"{pkg}.__all__ contains entries not accessible on the module: {phantom}"
        )


# ---------------------------------------------------------------------------
# E. Lint check (ruff)
# ---------------------------------------------------------------------------


class TestLint:
    def test_ruff_passes_on_all_three_inits(self) -> None:
        """ruff check must exit 0 for all three __init__.py files."""
        paths = [str(p) for p in _INIT_FILES.values()]
        result = subprocess.run(
            ["ruff", "check", *paths],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"ruff check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# G. Mypy baseline — no new type errors introduced in Cycle 2 packages
# ---------------------------------------------------------------------------

# Baseline error count in the three Cycle 2 packages as of the Cycle 2 gate.
# These errors pre-exist in research/reporting/_plots.py (matplotlib stubs).
# The test asserts the count does not increase (no regression).
_MYPY_BASELINE_ERROR_COUNT = 4
_CYCLE2_PACKAGE_PATHS = [
    "research/optimization",
    "research/factors",
    "research/reporting",
]


class TestMypyBaseline:
    def test_mypy_error_count_does_not_exceed_baseline(self) -> None:
        """mypy must not introduce new type errors beyond the pre-existing baseline."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                *_CYCLE2_PACKAGE_PATHS,
                "--ignore-missing-imports",
                "--no-error-summary",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        # Count only lines from the Cycle 2 packages that contain "error:"
        pkg_prefixes = tuple(
            p.replace("research/", "research/") for p in _CYCLE2_PACKAGE_PATHS
        )
        error_lines = [
            line
            for line in result.stdout.splitlines()
            if any(line.startswith(p) for p in pkg_prefixes) and "error:" in line
        ]
        count = len(error_lines)
        assert count <= _MYPY_BASELINE_ERROR_COUNT, (
            f"mypy error count in Cycle 2 packages is {count}, "
            f"exceeds baseline of {_MYPY_BASELINE_ERROR_COUNT}.\n"
            f"New errors:\n" + "\n".join(error_lines)
        )


# ---------------------------------------------------------------------------
# F. Import boundary — no cross-package pollution (AST-level)
# ---------------------------------------------------------------------------


class TestImportBoundary:
    @pytest.mark.parametrize("pkg", list(_INIT_FILES))
    def test_no_forbidden_imports_in_init(self, pkg: str) -> None:
        """__init__.py must not import from forbidden namespaces."""
        init_path = _INIT_FILES[pkg]
        imported = _ast_imports(init_path)
        violations = [
            mod
            for mod in imported
            if any(mod.startswith(prefix) for prefix in _FORBIDDEN_IMPORT_PREFIXES)
        ]
        assert violations == [], (
            f"{init_path.relative_to(_REPO_ROOT)} imports from forbidden namespaces: "
            f"{violations}. Allowed imports: within own package or optimizer.*"
        )
