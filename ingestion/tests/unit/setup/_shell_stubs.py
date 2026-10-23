"""Harness that runs the repo-root shell front doors against recording stub tools.

``install.sh`` and ``setup.sh`` shell out to ``uv``, ``git`` and ``portopt``. Putting
recording stubs first on ``PATH`` lets a test execute the real script end to end
without installing, cloning or starting anything. Each stub appends one
tab-separated line (tool, cwd, ``OPTIMIZER_REPO``, argv) to a shared log.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

_RECORD = (
    'printf \'%s\\t%s\\t%s\\t%s\\n\' "{name}" "$PWD" "${{OPTIMIZER_REPO-}}" "$*"'
    ' >> "$STUB_LOG"'
)

# `git clone` materialises the minimum a checkout needs for the installer to carry on.
_GIT = f"""#!/usr/bin/env bash
{_RECORD.format(name="git")}
if [ "$1" = clone ]; then
  dest="${{@: -1}}"
  mkdir -p "$dest/.git" "$dest/ingestion"
  : > "$dest/docker-compose.yml"
fi
"""

# `uv tool install` drops the CLI shim into ~/.local/bin, exactly where real uv puts it.
_UV = f"""#!/usr/bin/env bash
{_RECORD.format(name="uv")}
if [ "$1" = tool ] && [ "$2" = install ]; then
  mkdir -p "$HOME/.local/bin"
  cp "$STUB_DIR/portopt-shim" "$HOME/.local/bin/portopt"
fi
"""

_PORTOPT = f"""#!/usr/bin/env bash
{_RECORD.format(name="portopt")}
"""

_CURL = f"""#!/usr/bin/env bash
{_RECORD.format(name="curl")}
exit 1
"""


@dataclass(frozen=True)
class StubRun:
    """Outcome of one front-door execution.

    Attributes:
        result: The completed script process.
        calls: One ``[tool, cwd, optimizer_repo, argv]`` entry per stub invocation,
            in call order.
        home: The throwaway ``$HOME`` the script ran under.
    """

    result: subprocess.CompletedProcess[str]
    calls: list[list[str]]
    home: Path

    def calls_to(self, tool: str) -> list[list[str]]:
        """Return the recorded invocations of one stubbed tool."""
        return [call for call in self.calls if call[0] == tool]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_front_door(
    script: str,
    tmp_path: Path,
    *args: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    path_extra: tuple[str, ...] = (),
) -> StubRun:
    """Execute ``script`` (relative to the repo root) headless against the stubs.

    The child gets a new session and a null stdin, so it has no controlling
    terminal - the shape of a CI job, cron entry or ``ssh -T`` install.

    Args:
        script: Repo-root-relative script name, e.g. ``"install.sh"``.
        tmp_path: Scratch directory that holds ``$HOME``, the stubs and the log.
        *args: Arguments forwarded to the script.
        cwd: Working directory for the script (defaults to an empty scratch dir).
        env: Extra environment variables for the script.
        path_extra: Directories appended after the stub dir on ``PATH``, standing in
            for tools the user already had installed.

    Returns:
        The process result plus every stub invocation it made.
    """
    bash = shutil.which("bash")
    assert bash is not None
    home = tmp_path / "home"
    stubs = tmp_path / "stubs"
    workdir = cwd if cwd is not None else tmp_path / "cwd"
    for directory in (home, stubs, workdir):
        directory.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "stub.log"
    log.touch()
    _write_executable(stubs / "portopt-shim", _PORTOPT)
    _write_executable(stubs / "uv", _UV)
    _write_executable(stubs / "git", _GIT)
    _write_executable(stubs / "curl", _CURL)

    child_env = {
        "HOME": str(home),
        "PATH": os.pathsep.join([str(stubs), *path_extra, "/usr/bin", "/bin"]),
        "STUB_LOG": str(log),
        "STUB_DIR": str(stubs),
        **(env or {}),
    }
    result = subprocess.run(  # noqa: S603 - resolved bash + our own repo script
        [bash, str(REPO_ROOT / script), *args],
        capture_output=True,
        text=True,
        cwd=workdir,
        env=child_env,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        timeout=60,
        check=False,
    )
    calls = [line.split("\t") for line in log.read_text().splitlines()]
    return StubRun(result=result, calls=calls, home=home)
