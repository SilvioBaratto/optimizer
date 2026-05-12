"""Tests for ``research/reporting/`` module boundaries — issue #677.

Covers five acceptance criteria:
1. Importing ``_display.py`` does not produce visible stdout/stderr output
   (capture-safe: Console must NOT be initialised at module level).
2. ``_plots.py`` has zero runtime imports from ``optimizer.*`` or ``research.*``
   (only ``TYPE_CHECKING`` references permitted).
3. The Jinja2 template ``report.md.j2`` exists at the path ``_report.py`` computes.
4. Every name listed in ``research.reporting.__all__`` is importable.
5. ``import research.reporting`` in a subprocess exits 0 (no circular imports).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Test 1 — _display.py must not produce output at import time
# ---------------------------------------------------------------------------


class TestDisplayCapturesSafe:
    def test_importing_display_produces_no_stdout(self) -> None:
        """Importing research.reporting._display must not write to stdout."""
        # Force a fresh import by running in subprocess to avoid cached modules
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import research.reporting._display; import sys; sys.exit(0)",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.stdout == "", (
            f"Importing research.reporting._display produced stdout:\n{result.stdout!r}"
        )

    def test_importing_display_produces_no_stderr(self) -> None:
        """Importing research.reporting._display must not write to stderr."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import research.reporting._display; import sys; sys.exit(0)",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.stderr == "", (
            f"Importing research.reporting._display produced stderr:\n{result.stderr!r}"
        )

    def test_console_is_not_module_level_attribute(self) -> None:
        """The module-level ``console`` name must NOT exist on _display after lazy-init fix."""
        import research.reporting._display as display_mod

        # After the fix, ``console`` should be gone; ``_get_console`` should exist
        assert not hasattr(display_mod, "console"), (
            "research.reporting._display still exposes a module-level 'console'. "
            "Apply the lazy-init fix: remove 'console = Console()' and add '_get_console()'."
        )

    def test_get_console_accessor_exists(self) -> None:
        """After the fix, ``_get_console`` must be callable on _display."""
        import research.reporting._display as display_mod

        assert callable(getattr(display_mod, "_get_console", None)), (
            "research.reporting._display must expose a callable '_get_console()' accessor."
        )

    def test_get_console_returns_console_instance(self) -> None:
        """_get_console() must return a rich.console.Console instance."""
        from rich.console import Console

        from research.reporting._display import _get_console

        assert isinstance(_get_console(), Console)

    def test_get_console_is_idempotent(self) -> None:
        """Calling _get_console() twice must return the same singleton instance."""
        from research.reporting._display import _get_console

        first = _get_console()
        second = _get_console()
        assert first is second


# ---------------------------------------------------------------------------
# Test 2 — _plots.py has zero runtime optimizer/research imports
# ---------------------------------------------------------------------------


class TestPlotsZeroRuntimeOptimizerImports:
    """Verify _plots.py only references optimizer.* / research.* under TYPE_CHECKING."""

    _PLOTS_PATH = (
        Path(__file__).parent.parent.parent / "research" / "reporting" / "_plots.py"
    )

    def _collect_runtime_imports(self, source: str) -> list[str]:
        """Return module names imported at runtime (excluding TYPE_CHECKING blocks)."""
        tree = ast.parse(source)

        class RuntimeImportCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.imports: list[str] = []
                self._in_type_checking: bool = False

            def visit_If(self, node: ast.If) -> None:  # type: ignore[override]
                test = node.test
                is_tc = (
                    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                ) or (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                if is_tc:
                    saved = self._in_type_checking
                    self._in_type_checking = True
                    for child in node.body:
                        self.visit(child)
                    self._in_type_checking = saved
                else:
                    self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:  # type: ignore[override]
                if not self._in_type_checking:
                    for alias in node.names:
                        self.imports.append(alias.name)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # type: ignore[override]
                if not self._in_type_checking:
                    mod = node.module or ""
                    self.imports.append(mod)

        collector = RuntimeImportCollector()
        collector.visit(tree)
        return collector.imports

    def test_no_runtime_optimizer_imports_in_plots(self) -> None:
        source = self._PLOTS_PATH.read_text()
        runtime_imports = self._collect_runtime_imports(source)
        offenders = [m for m in runtime_imports if m.startswith("optimizer")]
        assert offenders == [], (
            f"_plots.py has runtime imports from optimizer.*: {offenders}"
        )

    def test_no_runtime_research_imports_in_plots(self) -> None:
        source = self._PLOTS_PATH.read_text()
        runtime_imports = self._collect_runtime_imports(source)
        offenders = [m for m in runtime_imports if m.startswith("research")]
        assert offenders == [], (
            f"_plots.py has runtime imports from research.*: {offenders}"
        )


# ---------------------------------------------------------------------------
# Test 3 — _report.py Jinja2 template path exists
# ---------------------------------------------------------------------------


class TestReportTemplatePath:
    def test_template_dir_exists(self) -> None:
        """The template directory resolved by _report.py must exist on disk."""
        from research.reporting import _report

        template_dir = _report._TEMPLATE_DIR
        assert template_dir.is_dir(), (
            f"Template directory does not exist: {template_dir}"
        )

    def test_report_template_file_exists(self) -> None:
        """The file ``report.md.j2`` must exist inside the resolved template dir."""
        from research.reporting import _report

        template_path = _report._TEMPLATE_DIR / _report._TEMPLATE_NAME
        assert template_path.is_file(), f"Jinja2 template not found: {template_path}"


# ---------------------------------------------------------------------------
# Test 4 — all public symbols importable from research.reporting
# ---------------------------------------------------------------------------


class TestReportingPublicAPI:
    def test_all_symbols_in_dunder_all_are_importable(self) -> None:
        """Every name in research.reporting.__all__ must be importable."""
        import research.reporting as pkg

        missing = []
        for name in pkg.__all__:
            obj = getattr(pkg, name, None)
            if obj is None:
                missing.append(name)
        assert missing == [], (
            f"The following names are in __all__ but not importable: {missing}"
        )


# ---------------------------------------------------------------------------
# Test 5 — no circular imports
# ---------------------------------------------------------------------------


class TestNoCircularImports:
    def test_import_research_reporting_exits_zero(self) -> None:
        """``import research.reporting`` in a subprocess must exit 0."""
        result = subprocess.run(
            [sys.executable, "-c", "import research.reporting"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 0, (
            f"Circular import detected.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
