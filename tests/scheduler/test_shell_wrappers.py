"""Structural tests for the ``scheduler/*.sh`` wrappers.

The scripts no longer drive HTTP endpoints — the API layer is gone. They are
thin wrappers that hand off to the daemon's Typer CLI, either inside the
compose container or against a local venv.

What is locked here is the handoff contract, since a typo in a script is not
caught by any Python import: the CLI subcommand each script invokes must be one
the CLI actually defines, the scripts must fail loudly rather than silently
continue past an error, and ``refetch_all`` must not have quietly lost its
universe-first ordering.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_DIR = REPO_ROOT / "scheduler"

FETCH = SCHEDULER_DIR / "fetch.sh"
REFETCH = SCHEDULER_DIR / "refetch_all.sh"
SCRIPTS = [FETCH, REFETCH]

# Subcommands registered on the Typer app in ingestion/app/cli.py.
CLI_COMMANDS = {
    "daily",
    "refetch-all",
    "universe",
    "yfinance",
    "macro",
    "fred",
    "news",
    "summarize",
    "calibrate",
    "reference-indices",
}

_CLI_INVOCATION = re.compile(r"python -m app\.cli ([a-z-]+)")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
class TestWrapperContract:
    def test_script_exists_and_is_executable(self, script: Path) -> None:
        assert script.is_file(), f"{script} is missing"
        assert script.stat().st_mode & stat.S_IXUSR, f"{script} is not executable"

    def test_invokes_only_real_cli_subcommands(self, script: Path) -> None:
        """A typo here fails at 3am in cron, not in CI — so check it in CI."""
        invoked = set(_CLI_INVOCATION.findall(script.read_text()))

        assert invoked, f"{script.name} never invokes `python -m app.cli`"
        unknown = invoked - CLI_COMMANDS
        assert not unknown, f"{script.name} invokes unknown CLI command(s): {unknown}"

    def test_aborts_on_error(self, script: Path) -> None:
        """Without -e a failed fetch would still exit 0 and look like success."""
        assert "set -euo pipefail" in script.read_text()

    def test_does_not_drive_the_removed_http_api(self, script: Path) -> None:
        text = script.read_text()
        assert "/api/v1" not in text
        assert "curl" not in text


class TestScriptTargets:
    def test_fetch_runs_the_daily_pipeline(self) -> None:
        assert _CLI_INVOCATION.findall(FETCH.read_text()) == ["daily", "daily"]

    def test_refetch_all_runs_the_full_rebuild(self) -> None:
        assert _CLI_INVOCATION.findall(REFETCH.read_text()) == [
            "refetch-all",
            "refetch-all",
        ]

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_supports_docker_and_local_runners(self, script: Path) -> None:
        text = script.read_text()
        assert "docker compose exec" in text
        assert 'RUNNER="${RUNNER:-docker}"' in text


_STUB = """#!/bin/sh
{ pwd -P; printf '%s\\n' "$@"; } > "$RECORD_FILE"
"""


def _run_with_stub_tool(
    script: Path, tool: str, tmp_path: Path, env_extra: dict[str, str]
) -> tuple[str, list[str]]:
    """Run ``script`` from an unrelated cwd with ``tool`` replaced by a recorder.

    Returns the cwd and argv the stub saw, i.e. what the wrapper really handed
    to docker / uv.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / tool
    stub.write_text(_STUB)
    stub.chmod(0o755)
    record = tmp_path / "record.txt"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    env = {
        "PATH": f"{stub_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "HOME": str(tmp_path),
        "RECORD_FILE": str(record),
        **env_extra,
    }
    done = subprocess.run(
        ["/bin/bash", str(script)],
        cwd=elsewhere,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    cwd, *argv = record.read_text().splitlines()
    return cwd, argv


@pytest.mark.skipif(sys.platform == "win32", reason="bash wrappers")
@pytest.mark.parametrize(
    ("script", "command"),
    [(FETCH, "daily"), (REFETCH, "refetch-all")],
    ids=lambda v: v.name if isinstance(v, Path) else v,
)
class TestRunnerEnvironment:
    """The wrappers must work from any cwd, e.g. a cron entry or ``$HOME``."""

    def test_docker_runner_executes_from_the_repo_root(
        self, script: Path, command: str, tmp_path: Path
    ) -> None:
        """Compose finds docker-compose.yml only from the cwd or a parent."""
        cwd, argv = _run_with_stub_tool(script, "docker", tmp_path, {})

        assert Path(cwd) == REPO_ROOT.resolve()
        assert argv == [
            "compose",
            "exec",
            "-T",
            "scheduler",
            "python",
            "-m",
            "app.cli",
            command,
        ]

    def test_local_runner_uses_the_workspace_venv_from_the_ingestion_dir(
        self, script: Path, command: str, tmp_path: Path
    ) -> None:
        """Bare ``python`` is whatever the caller has on PATH, not the workspace."""
        cwd, argv = _run_with_stub_tool(script, "uv", tmp_path, {"RUNNER": "local"})

        assert Path(cwd) == (REPO_ROOT / "ingestion").resolve()
        assert argv == [
            "run",
            "--package",
            "portopt",
            "python",
            "-m",
            "app.cli",
            command,
        ]
