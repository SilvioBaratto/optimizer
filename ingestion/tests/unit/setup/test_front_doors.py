"""Front-door funnel contract (task T13).

``setup.sh`` / ``setup.ps1`` / ``setup.cmd`` / ``make setup`` are razor-thin funnels:
ensure ``uv``, install the ``portopt`` CLI from the local checkout, export
``OPTIMIZER_REPO`` (so the out-of-repo tool venv can still locate
``scripts/optimizer`` — the repo self-heal), then hand off to ``portopt setup`` with
the caller's args. These assert the funnel shape, not a live run (live smoke is C4).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(name: str) -> str:
    return (_REPO_ROOT / name).read_text(encoding="utf-8")


class TestSetupSh:
    """The POSIX / Git Bash front door funnels into ``portopt setup``."""

    def test_exists(self) -> None:
        """setup.sh is present at the repo root."""
        assert (_REPO_ROOT / "setup.sh").is_file()

    def test_is_lf_only(self) -> None:
        """A CRLF shebang breaks bash under Git Bash / Linux (.gitattributes forces LF)."""
        assert b"\r" not in (_REPO_ROOT / "setup.sh").read_bytes()

    def test_has_bash_shebang(self) -> None:
        """The first line is a bash shebang."""
        first = _read("setup.sh").splitlines()[0]
        assert first.startswith("#!")
        assert "bash" in first

    def test_exports_optimizer_repo(self) -> None:
        """It exports OPTIMIZER_REPO so the tool-venv code can find the checkout."""
        assert "OPTIMIZER_REPO" in _read("setup.sh")

    def test_installs_cli_from_local_checkout(self) -> None:
        """It installs the CLI from the local project, not PyPI."""
        text = _read("setup.sh")
        assert "uv tool install" in text
        assert "--from ." in text

    def test_ensures_uv(self) -> None:
        """It bootstraps uv when missing."""
        assert "astral.sh/uv/install.sh" in _read("setup.sh")

    def test_hands_off_to_portopt_setup_with_args(self) -> None:
        """It execs ``portopt setup`` forwarding the caller's args verbatim."""
        text = _read("setup.sh")
        assert "portopt setup" in text
        assert '"$@"' in text

    def test_bash_syntax_is_valid(self) -> None:
        """``bash -n`` parses the script without error."""
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not available")
        result = subprocess.run(  # noqa: S603 - resolved bash path + our own script
            [bash, "-n", str(_REPO_ROOT / "setup.sh")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestSetupPs1:
    """The Windows PowerShell front door funnels into ``portopt setup``."""

    def test_exists(self) -> None:
        """setup.ps1 is present at the repo root."""
        assert (_REPO_ROOT / "setup.ps1").is_file()

    def test_exports_optimizer_repo(self) -> None:
        """It sets OPTIMIZER_REPO for the tool-venv code to locate the checkout."""
        assert "OPTIMIZER_REPO" in _read("setup.ps1")

    def test_installs_cli_from_local_checkout(self) -> None:
        """It installs the CLI from the local project, not PyPI."""
        text = _read("setup.ps1")
        assert "uv tool install" in text
        assert "--from ." in text

    def test_ensures_uv(self) -> None:
        """It bootstraps uv when missing."""
        assert "astral.sh/uv/install.ps1" in _read("setup.ps1")

    def test_hands_off_to_portopt_setup_with_args(self) -> None:
        """It hands off to ``portopt setup`` forwarding @args."""
        text = _read("setup.ps1")
        assert "portopt setup" in text
        assert "@args" in text


class TestSetupCmd:
    """The Windows cmd.exe front door funnels into ``portopt setup``."""

    def test_exists(self) -> None:
        """setup.cmd is present at the repo root."""
        assert (_REPO_ROOT / "setup.cmd").is_file()

    def test_exports_optimizer_repo(self) -> None:
        """It sets OPTIMIZER_REPO for the tool-venv code to locate the checkout."""
        assert "OPTIMIZER_REPO" in _read("setup.cmd")

    def test_installs_cli_from_local_checkout(self) -> None:
        """It installs the CLI from the local project, not PyPI."""
        text = _read("setup.cmd")
        assert "uv tool install" in text
        assert "--from ." in text

    def test_hands_off_to_portopt_setup_with_args(self) -> None:
        """It hands off to ``portopt setup`` forwarding %* (all args)."""
        text = _read("setup.cmd")
        assert "portopt setup" in text
        assert "%*" in text


class TestMakefileSetup:
    """``make setup`` reaches the wizard via the thin front door."""

    def test_has_setup_target(self) -> None:
        """A ``setup`` target exists."""
        assert "\nsetup:" in _read("Makefile")

    def test_setup_invokes_the_front_door(self) -> None:
        """The target delegates to setup.sh rather than duplicating logic."""
        assert "./setup.sh" in _read("Makefile")

    def test_setup_is_phony(self) -> None:
        """``setup`` is declared .PHONY (it produces no file named setup)."""
        phony = next(
            line for line in _read("Makefile").splitlines() if line.startswith(".PHONY")
        )
        assert "setup" in phony.split()
