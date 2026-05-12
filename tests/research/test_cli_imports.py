"""Regression test: research.cli must not import app.database at module level.

Issue #681: defer DatabaseManager import so `import research.cli` succeeds
outside the FastAPI app context (i.e., when `app/` is not on sys.path).
"""

from __future__ import annotations

import subprocess
import sys


class TestCliImportIsolation:
    def test_when_research_cli_imported_then_app_database_not_in_sys_modules(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import research.cli; import sys; "
                    "assert 'app.database' not in sys.modules, "
                    "'app.database was imported at module level'"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Import leaked app.database at module level.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_when_research_cli_main_imported_then_no_app_top_level_import(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import research.cli._main; import sys; "
                    "assert 'app.database' not in sys.modules, "
                    "'app.database was imported at module level via _main'"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"research.cli._main leaked app.database at module level.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
