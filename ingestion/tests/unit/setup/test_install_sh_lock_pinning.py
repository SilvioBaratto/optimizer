"""`install.sh` builds the host CLI against the checkout's uv.lock.

The containers install from the lock; the CLI is installed with `uv tool install`
from the `ingestion` package alone, so its transitive dependencies would otherwise
float to whatever is newest on the day. Constraints exported from the lock pin them.
A host Python the pinned wheels do not cover must still get a working CLI, so a failed
pinned install is retried without them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]

# `uv export` writes the file named after -o, like the real tool; `uv tool install`
# fails when asked to apply constraints and STUB_PINNED_FAILS is set.
_UV = r"""#!/bin/sh
echo "uv $*" >> "$STUB_LOG"
if [ "$1" = export ]; then
  [ -n "$STUB_EXPORT_FAILS" ] && exit 1
  while [ $# -gt 0 ]; do
    if [ "$1" = -o ]; then echo "certifi==1.0" > "$2"; fi
    shift
  done
fi
if [ "$1" = tool ] && [ "$2" = install ]; then
  case " $* " in
    *" --constraints "*) [ -n "$STUB_PINNED_FAILS" ] && exit 1 ;;
  esac
fi
exit 0
"""


def _run_install(tmp_path: Path, **env: str) -> tuple[list[str], str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")
    (stubs / "uv").write_text(_UV, encoding="utf-8")
    (stubs / "portopt").write_text(
        '#!/bin/sh\necho "portopt $*" >> "$STUB_LOG"\n', encoding="utf-8"
    )
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    home = tmp_path / "home"
    (home / "optimizer").mkdir(parents=True)
    (home / "optimizer" / "docker-compose.yml").write_text("services: {}\n")
    done = subprocess.run(  # noqa: S603 - fixed bash path + our own install script
        [bash, str(_REPO_ROOT / "install.sh"), "--non-interactive"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env={
            "PATH": f"{stubs}:/usr/bin:/bin",
            "HOME": str(home),
            "STUB_LOG": str(log),
            **env,
        },
    )
    return log.read_text(encoding="utf-8").splitlines(), done.stderr


def _installs(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("uv tool install")]


def test_the_cli_is_installed_against_constraints_exported_from_the_lock(
    tmp_path: Path,
) -> None:
    """The lock is exported for the CLI's package only, without workspace members."""
    calls, _ = _run_install(tmp_path)

    (export,) = [c for c in calls if c.startswith("uv export")]
    for flag in ("--frozen", "--no-hashes", "--no-emit-workspace", "--package portopt"):
        assert flag in export
    (install,) = _installs(calls)
    assert " --constraints " in install
    assert install.rstrip().endswith("#subdirectory=ingestion")
    assert calls.index(export) < calls.index(install)


def test_a_failed_pinned_install_is_retried_without_the_constraints(
    tmp_path: Path,
) -> None:
    """A host Python the pinned wheels do not cover still gets a working CLI."""
    calls, stderr = _run_install(tmp_path, STUB_PINNED_FAILS="1")

    pinned, fresh = _installs(calls)
    assert " --constraints " in pinned
    assert " --constraints " not in fresh
    assert "without the lock constraints" in stderr
    assert any(c.startswith("portopt setup") for c in calls)


def test_an_unreadable_lock_installs_without_constraints_and_says_so(
    tmp_path: Path,
) -> None:
    """No constraints file means one plain install, never a half-pinned one."""
    calls, stderr = _run_install(tmp_path, STUB_EXPORT_FAILS="1")

    (install,) = _installs(calls)
    assert " --constraints " not in install
    assert "uv.lock" in stderr
