"""Makefile contract: every target is real and runs scripts that exist.

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
