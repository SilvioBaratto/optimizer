"""Makefile contract: no dead targets, and ``make all`` gates every workspace member.

Targets are exercised with ``make -n`` (print, never run), so nothing here starts
uv, Docker or a test suite.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MAKEFILE = _REPO_ROOT / "Makefile"

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None, reason="make not available"
)


def _phony_targets() -> list[str]:
    line = next(
        entry
        for entry in _MAKEFILE.read_text(encoding="utf-8").splitlines()
        if entry.startswith(".PHONY:")
    )
    return line.removeprefix(".PHONY:").split()


def _dry_run(target: str) -> str:
    make = shutil.which("make")
    assert make is not None
    result = subprocess.run(  # noqa: S603 - resolved make path + a fixed target name
        [make, "-n", "-C", str(_REPO_ROOT), target],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestMakefileTargets:
    """Every declared target is real and runs something that exists."""

    @pytest.mark.parametrize("target", _phony_targets())
    def test_phony_target_has_a_rule(self, target: str) -> None:
        """A name in .PHONY without a rule is a leftover from a deleted feature."""
        text = _MAKEFILE.read_text(encoding="utf-8")
        assert re.search(rf"^{re.escape(target)}:", text, re.MULTILINE)

    @pytest.mark.parametrize("target", _phony_targets())
    def test_target_only_runs_python_scripts_that_exist(self, target: str) -> None:
        """A recipe pointing at a deleted script fails with "can't open file"."""
        for script in re.findall(r"[\w./-]+\.py\b", _dry_run(target)):
            assert (_REPO_ROOT / script).is_file(), f"{target} runs missing {script}"


class TestMakeAll:
    """``make all`` is the local mirror of CI, not just the portopt-core slice."""

    @pytest.mark.parametrize(
        "command",
        [
            "--cov=optimizer",
            "--cov=portopt_db",
            "--cov=app",
            "--cov=fund",
            "mypy src/portopt_db",
            "pyright",
        ],
    )
    def test_all_runs_the_ci_gate_for_every_member(self, command: str) -> None:
        """Lint, types and tests for db, ingestion and fund are part of ``all``."""
        assert command in _dry_run("all")

    def test_contributing_lists_every_target_that_all_runs(self) -> None:
        """The contributor guide names each ``make`` step behind ``make all``.

        ``make all`` grew from lint + typecheck + test into a CI mirror; a guide that
        still says "lint + typecheck + test" hides a much longer run from whoever
        follows "ensure all checks pass: make all".
        """
        makefile = _MAKEFILE.read_text(encoding="utf-8")
        match = re.search(r"^all:\s*(.+)$", makefile, re.MULTILINE)
        assert match is not None
        guide = (_REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

        missing = [
            target
            for target in match.group(1).split()
            if not re.search(rf"^make {re.escape(target)}\s", guide, re.MULTILINE)
        ]

        assert not missing, f"CONTRIBUTING.md does not document: {missing}"
