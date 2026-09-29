"""Clone-free installer contract.

``install.sh`` is the ``curl … | bash`` path: piping the script into bash binds the
shell's stdin to the pipe, so an interactive ``portopt setup`` would read EOF and
abort. The fix reconnects stdin to ``/dev/tty`` when a controlling terminal exists
and runs as-is when it does not (headless / ``--non-interactive``). ``install.ps1``
mirrors it (no ``/dev/tty`` on Windows — PowerShell children inherit the console
stdin directly — so parity there is arg-forwarding + matching structure).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(name: str) -> str:
    return (_REPO_ROOT / name).read_text(encoding="utf-8")


class TestInstallSh:
    """The POSIX clone-free installer reaches interactive prompts under curl|bash."""

    def test_exists(self) -> None:
        """install.sh is present at the repo root."""
        assert (_REPO_ROOT / "install.sh").is_file()

    def test_is_lf_only(self) -> None:
        """A CRLF shebang breaks bash (.gitattributes forces LF)."""
        assert b"\r" not in (_REPO_ROOT / "install.sh").read_bytes()

    def test_guards_the_dev_tty_read(self) -> None:
        """It reconnects stdin to /dev/tty only when a terminal is readable."""
        text = _read("install.sh")
        assert "/dev/tty" in text
        assert "-r /dev/tty" in text

    def test_has_a_headless_fallback(self) -> None:
        """A no-tty branch still runs setup (so --non-interactive/CI survives)."""
        text = _read("install.sh")
        # Two invocations: one redirected from /dev/tty, one plain fallback.
        assert text.count("portopt setup") >= 2

    def test_forwards_args_to_setup(self) -> None:
        """It forwards args so `curl … | bash -s -- --non-interactive …` works."""
        assert '"$@"' in _read("install.sh")

    def test_installs_the_cli_via_uv_tool(self) -> None:
        """The clone-free path installs the published `portopt` via uv tool."""
        assert "uv tool install portopt" in _read("install.sh")

    def test_ensures_uv(self) -> None:
        """It bootstraps uv when missing."""
        assert "astral.sh/uv/install.sh" in _read("install.sh")

    def test_bash_syntax_is_valid(self) -> None:
        """``bash -n`` parses the script without error."""
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not available")
        result = subprocess.run(  # noqa: S603 - resolved bash path + our own script
            [bash, "-n", str(_REPO_ROOT / "install.sh")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestInstallPs1:
    """The Windows clone-free installer mirrors install.sh behaviour."""

    def test_exists(self) -> None:
        """install.ps1 is present at the repo root."""
        assert (_REPO_ROOT / "install.ps1").is_file()

    def test_forwards_args_to_setup(self) -> None:
        """It forwards @args to `portopt setup` (parity with install.sh)."""
        text = _read("install.ps1")
        assert "portopt setup" in text
        assert "@args" in text

    def test_installs_the_cli_via_uv_tool(self) -> None:
        """It installs the published `portopt` via uv tool."""
        assert "uv tool install portopt" in _read("install.ps1")

    def test_ensures_uv(self) -> None:
        """It bootstraps uv when missing."""
        assert "astral.sh/uv/install.ps1" in _read("install.ps1")
