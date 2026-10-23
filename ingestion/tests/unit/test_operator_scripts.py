"""Guards for the operator scripts under ``ingestion/scripts``.

Nothing in the daemon imports these scripts, so a refactor of ``app`` can
break one without any other test noticing. These guards read the scripts'
imports instead of running them: a script is never executed, so no database,
network or API key is touched.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "ingestion" / "scripts"
_LAUNCH_JSON = _REPO_ROOT / ".vscode" / "launch.json"

# Third-party imports depend on which extras the environment installed, which is
# not what a refactor of this repository breaks.
_FIRST_PARTY_ROOTS = {"app", "portopt_db", "optimizer"}

# Dropped together with its tables in the ingestion strip, and the Trading 212
# client no longer exposes positions, so a script cannot work around its absence.
_REMOVED_PORTFOLIO_LAYER = (
    "app.repositories.portfolio",
    "app.services.portfolio",
)


def _script_paths() -> list[Path]:
    return sorted(p for p in _SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py")


def _imported_modules(script: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _first_party_import_targets(script: Path) -> list[tuple[str, list[str]]]:
    """Return ``(module, imported names)`` for each first-party import."""
    targets: list[tuple[str, list[str]]] = []
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            targets.extend(
                (alias.name, [])
                for alias in node.names
                if alias.name.split(".")[0] in _FIRST_PARTY_ROOTS
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
            and node.module.split(".")[0] in _FIRST_PARTY_ROOTS
        ):
            targets.append((node.module, [alias.name for alias in node.names]))
    return targets


def _unresolved_imports(script: Path) -> list[str]:
    unresolved: list[str] = []
    for module_name, names in _first_party_import_targets(script):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            unresolved.append(f"{module_name} ({exc})")
            continue
        for name in names:
            if hasattr(module, name):
                continue
            try:
                importlib.import_module(f"{module_name}.{name}")
            except ImportError:
                unresolved.append(f"{module_name}.{name}")
    return unresolved


def _launch_script_modules() -> list[str]:
    lines = _LAUNCH_JSON.read_text(encoding="utf-8").splitlines()
    launch = json.loads(
        "\n".join(ln for ln in lines if not ln.lstrip().startswith("//"))
    )
    return [
        cfg["module"]
        for cfg in launch["configurations"]
        if str(cfg.get("module", "")).startswith("scripts.")
    ]


def test_scripts_do_not_import_the_removed_portfolio_layer() -> None:
    """No operator script may depend on the portfolio / snapshot layer."""
    offenders = {
        script.name: sorted(
            module
            for module in _imported_modules(script)
            if module.startswith(_REMOVED_PORTFOLIO_LAYER)
        )
        for script in _script_paths()
    }
    offenders = {name: mods for name, mods in offenders.items() if mods}

    assert not offenders, f"scripts import the removed portfolio layer: {offenders}"


@pytest.mark.parametrize("script", _script_paths(), ids=lambda p: p.stem)
def test_script_first_party_imports_resolve(script: Path) -> None:
    """Every ``app`` / ``portopt_db`` / ``optimizer`` import a script makes exists."""
    assert _unresolved_imports(script) == []


@pytest.mark.skipif(not _LAUNCH_JSON.exists(), reason="no .vscode/launch.json")
def test_launch_configs_for_scripts_point_at_startable_scripts() -> None:
    """A VS Code launch config must not target a missing or unimportable script."""
    broken = {}
    for module in _launch_script_modules():
        script = _SCRIPTS_DIR / f"{module.removeprefix('scripts.')}.py"
        problems = (
            _unresolved_imports(script) if script.exists() else ["script is missing"]
        )
        if problems:
            broken[module] = problems

    assert not broken, f"launch configs target broken scripts: {broken}"
