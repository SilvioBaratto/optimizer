"""Guard: the opt-in deployment suite stays wired to the live workspace layout.

The suite is excluded from every default run, so nothing executes it and it
rots silently when the workspace moves underneath it. These checks read the
suite's own constants and argv builders without ever opting in, so they
cannot build a venv or reach Docker.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

import tomllib

INGESTION_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = INGESTION_ROOT.parent
DEPLOYMENT_TESTS = INGESTION_ROOT / "tests" / "integration" / "deployment"


def _load(filename: str) -> ModuleType:
    path = DEPLOYMENT_TESTS / filename
    spec = importlib.util.spec_from_file_location(
        f"_deployment_suite_{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workspace_dependency_dirs() -> set[Path]:
    """Directories of the workspace members that ingestion depends on."""
    ingestion = tomllib.loads((INGESTION_ROOT / "pyproject.toml").read_text())
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    workspace_sources = {
        name
        for name, spec in ingestion["tool"]["uv"]["sources"].items()
        if spec.get("workspace")
    }
    member_dirs: dict[str, Path] = {}
    for member in root["tool"]["uv"]["workspace"]["members"]:
        member_dir = REPO_ROOT / member
        manifest = tomllib.loads((member_dir / "pyproject.toml").read_text())
        member_dirs[manifest["project"]["name"]] = member_dir

    bare_names = {
        dep.split("==")[0].split("[")[0].strip()
        for dep in ingestion["project"]["dependencies"]
    }
    return {member_dirs[name] for name in workspace_sources & bare_names}


def test_hygiene_guard_file_exists_in_requirements_only_venv_test():
    """The guard the venv test runs must exist where the test points."""
    module = _load("test_hygiene_suite_runs_in_requirements_only_venv.py")

    guard = module._INGESTION_ROOT / module._GUARD_FILE

    assert guard.is_file(), f"stale guard path: {guard}"


def test_fresh_venv_installs_workspace_dependencies_from_local_paths():
    """PyPI has no ``portopt-db``; pip only resolves it from its directory."""
    conftest = _load("conftest.py")
    expected = _workspace_dependency_dirs()
    assert expected, "ingestion declares no workspace dependency to install"

    targets = {Path(arg.split("[")[0]) for arg in conftest._pip_install_targets()}

    assert expected <= targets, (
        f"pip install targets {sorted(map(str, targets))} omit {sorted(map(str, expected - targets))}"
    )


def test_compose_down_uses_the_profile_that_compose_up_started():
    """A profile-less ``down`` touches no profile-gated service, so it leaks."""
    conftest = _load("conftest.py")

    up = conftest._compose_argv("docker", "up", "-d")
    down = conftest._compose_argv("docker", "down")

    assert up[:2] == ["docker", "compose"]
    assert "--profile" in down
    assert down[: down.index("down")] == up[: up.index("up")]


def test_scheduler_fixture_teardown_goes_through_the_profiled_compose_helper():
    """The fixture must not hand-roll a ``compose down`` argv of its own."""
    tree = ast.parse((DEPLOYMENT_TESTS / "conftest.py").read_text())
    fixture = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "running_scheduler_container"
    )

    literal_downs = [
        node
        for node in ast.walk(fixture)
        if isinstance(node, ast.List)
        and {"compose", "down"}
        <= {
            elt.value
            for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    ]

    assert not literal_downs, "teardown hand-rolls an argv without --profile"
