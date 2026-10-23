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
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
INGESTION_README = REPO_ROOT / "ingestion" / "README.md"
VSCODE_LAUNCH = REPO_ROOT / ".vscode" / "launch.json"


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


_CLI_INVOCATION = re.compile(r"python -m app\.cli ([a-z-]+)")
_SERVICE_DEFAULT = re.compile(r'SERVICE="\$\{SERVICE:-([^}]+)\}"')


def _command_functions(source: str) -> dict[str, ast.FunctionDef]:
    """Map each Typer command name in ``source`` to the function behind it.

    Reads the ``@app.command`` decorators statically rather than importing
    ``app.cli``: that module loads ``.env`` into ``os.environ`` as an import side
    effect, which must not leak into the library suite.
    """
    commands: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if _is_app_command(decorator):
                commands[_command_name(decorator, node.name)] = node
    return commands


def _is_app_command(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "command"
        and isinstance(target.value, ast.Name)
        and target.value.id == "app"
    )


def _command_name(decorator: ast.expr, function_name: str) -> str:
    """Return the name Typer registers: explicit ``name`` or the derived one."""
    if isinstance(decorator, ast.Call):
        explicit = decorator.args[0] if decorator.args else None
        for keyword in decorator.keywords:
            if keyword.arg == "name":
                explicit = keyword.value
        if isinstance(explicit, ast.Constant) and isinstance(explicit.value, str):
            return explicit.value
    # Typer lower-cases the function name and swaps underscores for hyphens.
    return function_name.lower().replace("_", "-")


def registered_cli_commands(source: str) -> set[str]:
    """Return the subcommand names the Typer app in ``source`` registers."""
    return set(_command_functions(source))


CLI_COMMANDS = registered_cli_commands(CLI_SOURCE.read_text(encoding="utf-8"))


def unknown_cli_commands(script_text: str) -> set[str]:
    """Return CLI subcommands a wrapper invokes that the CLI does not define."""
    return set(_CLI_INVOCATION.findall(script_text)) - CLI_COMMANDS


def refetch_all_step_order(cli_source: str) -> list[str]:
    """Return the scheduler step functions ``refetch-all`` calls, in call order."""
    function = _command_functions(cli_source)["refetch-all"]
    steps = [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id.startswith("run_")
    ]
    steps.sort(key=lambda call: (call.lineno, call.col_offset))
    return [call.func.id for call in steps if isinstance(call.func, ast.Name)]


def service_default(script_text: str) -> str | None:
    """Return the compose service a wrapper targets when ``SERVICE`` is unset."""
    match = _SERVICE_DEFAULT.search(script_text)
    return match.group(1) if match else None


def compose_services(compose_text: str) -> set[str]:
    """Return the service names declared in a compose file."""
    return set(yaml.safe_load(compose_text)["services"])


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

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
    def test_parses_under_bash(self, script: Path) -> None:
        """A syntax error otherwise only surfaces when cron fires the script."""
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_targets_a_service_compose_defines(self, script: Path) -> None:
        """Renaming the compose service must fail here, not at the next cron run."""
        target = service_default(script.read_text())

        assert target is not None, f"{script.name} has no SERVICE default"
        services = compose_services(COMPOSE_FILE.read_text(encoding="utf-8"))
        assert target in services, (
            f"{script.name} execs into {target!r}, absent from docker-compose.yml "
            f"services {sorted(services)}"
        )


class TestCliCommandSet:
    def test_parser_finds_commands_in_the_real_cli(self) -> None:
        """An empty set would make every wrapper look like it calls a typo."""
        assert CLI_COMMANDS, f"no @app.command found in {CLI_SOURCE}"

    def test_every_command_a_wrapper_invokes_is_registered(self) -> None:
        invoked = {
            command
            for script in SCRIPTS
            for command in _CLI_INVOCATION.findall(script.read_text())
        }

        assert invoked, "no wrapper invokes the CLI"
        assert invoked <= CLI_COMMANDS

    def test_wrapper_calling_an_unregistered_command_is_rejected(self) -> None:
        missing = "no-such-command"
        assert missing not in CLI_COMMANDS
        text = f"exec docker compose exec -T x python -m app.cli {missing}"

        assert unknown_cli_commands(text) == {missing}

    def test_explicit_name_overrides_the_function_name(self) -> None:
        source = (
            "@app.command(name='refetch-all')\n"
            "def refetch_all(): ...\n"
            "@app.command('market-summary')\n"
            "def market_summary_cmd(): ...\n"
        )

        assert registered_cli_commands(source) == {"refetch-all", "market-summary"}

    def test_unnamed_command_takes_a_hyphenated_function_name(self) -> None:
        source = (
            "@app.command()\ndef market_structure(): ...\n"
            "@app.command\ndef Daily(): ...\n"
        )

        assert registered_cli_commands(source) == {"market-structure", "daily"}

    def test_functions_not_registered_on_the_app_are_ignored(self) -> None:
        source = (
            "@other.command()\ndef not_ours(): ...\n"
            "@app.callback()\ndef main(): ...\n"
            "def helper(): ...\n"
        )

        assert registered_cli_commands(source) == set()


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

    def test_refetch_all_rebuilds_the_universe_before_fetching(self) -> None:
        """Every fetch step iterates ``instruments``, so a stale universe caps them."""
        order = refetch_all_step_order(CLI_SOURCE.read_text(encoding="utf-8"))

        assert order[:1] == ["run_universe_build"], f"refetch-all runs {order}"

    def test_step_order_helper_reports_a_swapped_command(self) -> None:
        source = (
            "@app.command(name='refetch-all')\n"
            "def refetch_all():\n"
            "    run_weekly_refetch()\n"
            "    run_universe_build()\n"
        )

        assert refetch_all_step_order(source) == [
            "run_weekly_refetch",
            "run_universe_build",
        ]

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_supports_docker_and_local_runners(self, script: Path) -> None:
        text = script.read_text()
        assert "docker compose exec" in text
        assert 'RUNNER="${RUNNER:-docker}"' in text

    def test_service_default_parser_reads_the_shipped_default(self) -> None:
        assert service_default('RUNNER="x"\nSERVICE="${SERVICE:-scheduler}"\n') == (
            "scheduler"
        )
        assert service_default("no default here") is None


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
