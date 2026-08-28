"""Guard: portopt-db must never import the ``optimizer`` library or declare the
optimization stack (``skfolio``).

portopt-db is the shared data layer (models / repositories / engine / Alembic).
It sits *below* ingestion and any future consumer; pulling in the optimizer /
skfolio stack would couple the database package to portfolio optimization.
Source-blind scan of the package source + its pyproject.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src" / "portopt_db"
_PYPROJECT = _PKG_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

_OPTIMIZER_IMPORT = re.compile(
    r"^\s*(from optimizer\b|import optimizer\b)", re.MULTILINE
)
_FORBIDDEN_PACKAGES = ("skfolio",)


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or path.resolve() == _SELF:
            continue
        yield path


def find_optimizer_import_violations(root: Path) -> list[str]:
    offending: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _OPTIMIZER_IMPORT.search(text):
            offending.append(str(path.relative_to(root)))
    return offending


def find_forbidden_stack_dependencies(text: str) -> list[str]:
    lowered = text.lower()
    return [pkg for pkg in _FORBIDDEN_PACKAGES if pkg in lowered]


def test_when_src_is_scanned_then_no_optimizer_import_is_found():
    assert find_optimizer_import_violations(_SRC) == []


def test_when_pyproject_is_read_then_no_optimization_stack_dependency_is_declared():
    assert (
        find_forbidden_stack_dependencies(_PYPROJECT.read_text(encoding="utf-8")) == []
    )


def test_when_an_optimizer_import_is_injected_then_the_guard_fails(tmp_path):
    (tmp_path / "offender.py").write_text("import optimizer\n", encoding="utf-8")
    assert find_optimizer_import_violations(tmp_path)


def test_when_a_forbidden_stack_package_is_injected_then_the_guard_fails():
    assert find_forbidden_stack_dependencies("skfolio==1.0.0\n") == ["skfolio"]
