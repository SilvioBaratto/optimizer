"""Guards for the operator scripts under ``ingestion/scripts``.

Nothing in the daemon imports these scripts, so a refactor of ``app`` can
break one without any other test noticing. These guards parse the scripts
instead of running them: a script is never executed, so no database, network
or API key is touched.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

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
