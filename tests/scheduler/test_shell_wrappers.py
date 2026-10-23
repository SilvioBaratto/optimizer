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

import ast
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

# Pipeline steps that used to be CLI commands and no longer exist; anything
# that still mentions them is describing behaviour the daemon does not have.
# ``ref-indices`` is the short spelling the old ``daily`` help text used.
REMOVED_STEPS = {"summarize", "calibrate", "reference-indices", "ref-indices"}

CLI_SOURCE = REPO_ROOT / "ingestion" / "app" / "cli.py"
INGESTION_README = REPO_ROOT / "ingestion" / "README.md"
VSCODE_LAUNCH = REPO_ROOT / ".vscode" / "launch.json"


def _registered_cli_commands(source: Path) -> set[str]:
    """Read the Typer command names off ``cli.py`` without importing it.

    Importing would load dotenv and open the DB engine; the decorators are all
    ``@app.command()`` / ``@app.command(name=...)`` so the AST is enough, and a
    hand-kept list goes stale the moment a command is added or removed.
    """
    commands: set[str] = set()
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            if not (
                isinstance(deco, ast.Call)
                and isinstance(deco.func, ast.Attribute)
                and deco.func.attr == "command"
            ):
                continue
            named = {
                kw.arg: kw.value.value
                for kw in deco.keywords
                if isinstance(kw.value, ast.Constant)
            }
            commands.add(named.get("name", node.name.replace("_", "-")))
    return commands


def _cli_help_texts(source: Path) -> dict[str, str]:
    """Module docstring plus every top-level function docstring in ``cli.py``.

    The command docstrings become ``--help`` output, so a stale step name there
    is shown to the operator who runs the command.
    """
    tree = ast.parse(source.read_text())
    texts = {"<module>": ast.get_docstring(tree) or ""}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            texts[node.name] = ast.get_docstring(node) or ""
    return texts


CLI_COMMANDS = _registered_cli_commands(CLI_SOURCE)

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


class TestStaleReferences:
    def test_command_guard_lists_only_commands_the_cli_defines(self) -> None:
        assert {"daily", "refetch-all"} <= CLI_COMMANDS
        assert CLI_COMMANDS.isdisjoint(REMOVED_STEPS)

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_does_not_describe_removed_pipeline_steps(self, script: Path) -> None:
        mentioned = {
            step
            for step in REMOVED_STEPS
            if re.search(rf"\b{re.escape(step)}\b", script.read_text())
        }

        assert not mentioned, (
            f"{script.name} still mentions removed step(s): {mentioned}"
        )

    def test_cli_help_text_does_not_mention_removed_steps(self) -> None:
        stale = {
            name: sorted(
                step
                for step in REMOVED_STEPS
                if re.search(rf"\b{re.escape(step)}\b", text)
            )
            for name, text in _cli_help_texts(CLI_SOURCE).items()
        }
        stale = {name: steps for name, steps in stale.items() if steps}

        assert not stale, f"cli.py help text still names removed step(s): {stale}"

    def test_readme_command_list_only_names_real_cli_commands(self) -> None:
        invoked = set(_CLI_INVOCATION.findall(INGESTION_README.read_text()))

        assert invoked, "ingestion/README.md no longer lists any CLI command"
        unknown = invoked - CLI_COMMANDS
        assert not unknown, f"README lists unknown CLI command(s): {unknown}"

    def test_vscode_launch_configs_only_target_real_cli_commands(self) -> None:
        """launch.json is JSONC, so match the module/args pair instead of parsing."""
        launched = set(
            re.findall(
                r'"module":\s*"app\.cli",\s*"args":\s*\["([^"]+)"',
                VSCODE_LAUNCH.read_text(),
            )
        )

        assert launched, ".vscode/launch.json no longer launches any app.cli command"
        unknown = launched - CLI_COMMANDS
        assert not unknown, f"launch.json targets unknown CLI command(s): {unknown}"


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
exit "${STUB_EXIT:-0}"
"""


def _run_with_stub_tool(
    script: Path, tool: str, tmp_path: Path, env_extra: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], str, list[str]]:
    """Run ``script`` from an unrelated cwd with ``tool`` replaced by a recorder.

    Returns the finished process plus the cwd and argv the stub saw, i.e. what
    the wrapper really handed to docker / uv. The stub exits with ``STUB_EXIT``.
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
    cwd, *argv = record.read_text().splitlines()
    return done, cwd, argv


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
        done, cwd, argv = _run_with_stub_tool(script, "docker", tmp_path, {})

        assert done.returncode == 0, done.stderr
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
        done, cwd, argv = _run_with_stub_tool(
            script, "uv", tmp_path, {"RUNNER": "local"}
        )

        assert done.returncode == 0, done.stderr
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

    @pytest.mark.parametrize(
        ("tool", "env_extra"),
        [("docker", {}), ("uv", {"RUNNER": "local"})],
        ids=["docker", "local"],
    )
    def test_exit_status_is_the_clis_own(
        self,
        script: Path,
        command: str,
        tool: str,
        env_extra: dict[str, str],
        tmp_path: Path,
    ) -> None:
        """The wrapper neither masks nor invents a failure.

        Whether ``daily`` / ``refetch-all`` ever exit non-zero is the CLI's
        contract (pinned in ingestion/tests/unit/test_cli.py: they always exit
        0). The wrapper only has to hand that status through unchanged, which
        is what makes "do not gate on this script" true advice.
        """
        done, _, _ = _run_with_stub_tool(
            script, tool, tmp_path, {**env_extra, "STUB_EXIT": "7"}
        )

        assert done.returncode == 7, done.stderr
