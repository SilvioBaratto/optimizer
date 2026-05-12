"""Import-graph boundary tests for research.pipeline and research.cli.

Cycle 3 acceptance criteria: pipeline/ and cli/ must never pull app.* modules
at import time. Circular imports must not exist across research subpackages.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

_RESEARCH_PACKAGES = [
    "research",
    "research.cli",
    "research.data",
    "research.factors",
    "research.optimization",
    "research.persistence",
    "research.pipeline",
    "research.preflight",
    "research.reporting",
    "research.returns",
    "research.strategies",
]

_SUBPROCESS_SNIPPET = """\
import {module}
import sys
leaks = [k for k in sys.modules if k.startswith("app.")]
assert not leaks, f"app.* leaked into sys.modules: {{leaks}}"
"""


def _run_isolation(module: str) -> tuple[int, str]:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SUBPROCESS_SNIPPET.format(module=module)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stderr


class TestPipelineNoAppAtImportTime:
    def test_when_pipeline_imported_then_no_app_modules_in_sys_modules(
        self,
    ) -> None:
        code, stderr = _run_isolation("research.pipeline")
        assert code == 0, f"app.* leaked on import of research.pipeline:\n{stderr}"

    def test_when_pipeline_load_imported_then_no_app_modules(self) -> None:
        code, stderr = _run_isolation("research.pipeline._load")
        assert code == 0, (
            f"app.* leaked on import of research.pipeline._load:\n{stderr}"
        )

    def test_when_pipeline_optimize_imported_then_no_app_modules(self) -> None:
        code, stderr = _run_isolation("research.pipeline._optimize")
        assert code == 0, (
            f"app.* leaked on import of research.pipeline._optimize:\n{stderr}"
        )

    def test_when_pipeline_regime_imported_then_no_app_modules(self) -> None:
        code, stderr = _run_isolation("research.pipeline._regime")
        assert code == 0, (
            f"app.* leaked on import of research.pipeline._regime:\n{stderr}"
        )


class TestCliNoAppAtImportTime:
    def test_when_cli_imported_then_no_app_modules_in_sys_modules(self) -> None:
        code, stderr = _run_isolation("research.cli")
        assert code == 0, f"app.* leaked on import of research.cli:\n{stderr}"

    def test_when_cli_parser_imported_then_no_app_modules(self) -> None:
        code, stderr = _run_isolation("research.cli._parser")
        assert code == 0, f"app.* leaked on import of research.cli._parser:\n{stderr}"


class TestNoCircularImports:
    def test_each_research_package_imports_without_error(self) -> None:
        errors: list[str] = []
        for pkg in _RESEARCH_PACKAGES:
            try:
                importlib.import_module(pkg)
            except ImportError as exc:
                errors.append(f"{pkg}: {exc}")
        assert not errors, "Circular/broken imports detected:\n" + "\n".join(errors)

    def test_pipeline_and_cli_import_in_sequence_without_error(self) -> None:
        importlib.import_module("research.pipeline")
        importlib.import_module("research.cli")
