"""Every third-party module the daemon imports is a declared dependency.

The host CLI (`portopt`) is installed with `uv tool install` from the `ingestion`
package alone, outside the workspace lock. A module the code imports but the
`pyproject.toml` does not declare is therefore only present while some other
package happens to pull it in transitively; a release of that package that drops
the dependency breaks the CLI at import time.
"""

from __future__ import annotations

import ast
import importlib.metadata as metadata
import re
import sys
from pathlib import Path

import tomllib

# unit -> hygiene -> tests -> ingestion
_INGESTION = Path(__file__).resolve().parents[3]
_REPO = _INGESTION.parent
_PORTOPT_DB = _REPO / "packages" / "portopt-db" / "pyproject.toml"

# Modules that are part of this workspace, not third-party distributions.
_FIRST_PARTY = {"app", "portopt_db"}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared(pyproject: Path) -> set[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    names = set()
    for spec in data["project"]["dependencies"]:
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec)
        assert match, spec
        names.add(_normalise(match.group(0)))
    return names


def _imported_top_level_modules() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted((_INGESTION / "app").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.setdefault(name.split(".")[0], path)
    return found


def test_every_third_party_import_is_a_declared_dependency() -> None:
    """An import served only by a transitive dependency is a latent breakage."""
    declared = _declared(_INGESTION / "pyproject.toml") | _declared(_PORTOPT_DB)
    owners = metadata.packages_distributions()
    undeclared: dict[str, str] = {}
    for module, path in _imported_top_level_modules().items():
        if module in sys.stdlib_module_names or module in _FIRST_PARTY:
            continue
        distributions = {_normalise(d) for d in owners.get(module, [])}
        if not distributions & declared:
            undeclared[module] = str(path.relative_to(_REPO))

    assert undeclared == {}, (
        "imported but not declared in ingestion/pyproject.toml (or portopt-db): "
        f"{undeclared}"
    )
