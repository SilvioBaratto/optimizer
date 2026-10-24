"""Guard: the optimizer library (``portopt-core``) must never import
``portopt_db`` / SQLAlchemy, nor declare the DB packages as dependencies.

The library is deliberately database-free — it takes DataFrames and is published
as a standalone wheel. Depending on ``portopt-db`` (or SQLAlchemy / psycopg2 /
alembic) would drag the whole data stack into that wheel and break the "usable
without a database" property. Inverse of ingestion's no-optimizer guard.

Source-blind scan of ``packages/portopt-core/optimizer`` (imports are read from
the syntax tree, so ``import os, portopt_db`` and
``importlib.import_module("portopt_db")`` are caught and a docstring quoting an
import is not) + a tomllib parse of
the root ``pyproject`` dependencies (parsing, not raw text, so the
``[tool.uv.workspace]`` member entry ``packages/portopt-db`` does not
false-positive).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import tomllib

from ._import_scan import imported_module_names

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPTIMIZER_SRC = _REPO_ROOT / "packages" / "portopt-core" / "optimizer"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_FORBIDDEN_IMPORT_ROOTS = frozenset({"portopt_db", "sqlalchemy", "psycopg2", "alembic"})
_FORBIDDEN_DEP_TOKENS = (
    "portopt-db",
    "portopt_db",
    "sqlalchemy",
    "psycopg2",
    "alembic",
)


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def find_db_import_violations(root: Path) -> list[str]:
    offending: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            module.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS
            for module in imported_module_names(text, str(path))
        ):
            offending.append(str(path.relative_to(root)))
    return offending


def find_db_dependencies(pyproject_text: str) -> list[str]:
    """Return forbidden DB tokens present in the declared dependencies only."""
    data = tomllib.loads(pyproject_text)
    project = data.get("project", {})
    deps: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        deps.extend(group)
    lowered = " ".join(deps).lower()
    return [tok for tok in _FORBIDDEN_DEP_TOKENS if tok in lowered]


def test_when_optimizer_src_is_scanned_then_no_db_import_is_found():
    # A missing path makes rglob yield nothing, silently disabling the guard, so
    # require the directory to resolve — a future layout move then fails loudly.
    assert _OPTIMIZER_SRC.is_dir(), f"optimizer source not found at {_OPTIMIZER_SRC}"
    assert find_db_import_violations(_OPTIMIZER_SRC) == []


def test_when_root_pyproject_is_read_then_no_db_package_is_declared():
    assert find_db_dependencies(_PYPROJECT.read_text(encoding="utf-8")) == []


def test_when_a_portopt_db_import_is_injected_then_the_guard_fails(tmp_path):
    (tmp_path / "offender.py").write_text("import portopt_db\n", encoding="utf-8")
    assert find_db_import_violations(tmp_path)


def test_when_a_db_dependency_is_injected_then_the_guard_fails():
    manifest = '[project]\ndependencies = ["portopt-db", "numpy==2.5.2"]\n'
    assert "portopt-db" in find_db_dependencies(manifest)


@pytest.mark.parametrize(
    "source",
    [
        "import numpy, portopt_db\n",
        "from  sqlalchemy import select\n",
        "from\talembic import op\n",
        "x = 1; import psycopg2\n",
        "if True: import portopt_db\n",
        "import numpy, \\\n    sqlalchemy\n",
        "import importlib\nimportlib.import_module('portopt_db.models')\n",
        "__import__('sqlalchemy')\n",
    ],
    ids=[
        "comma-list",
        "double-space",
        "tab",
        "semicolon",
        "inline-if",
        "backslash-continuation",
        "import_module",
        "dunder-import",
    ],
)
def test_when_a_non_canonical_db_import_is_injected_then_the_guard_fails(
    tmp_path, source
):
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    assert find_db_import_violations(tmp_path) == ["offender.py"]


def test_when_a_docstring_quotes_a_db_import_then_the_guard_allows_it(tmp_path):
    (tmp_path / "clean.py").write_text(
        '"""Persist with:\n\nfrom portopt_db import Base\n"""\n', encoding="utf-8"
    )
    assert find_db_import_violations(tmp_path) == []
