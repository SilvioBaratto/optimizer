"""Guard: portopt-db must never import the ingestion daemon (``app``).

portopt-db sits *below* ingestion: the daemon depends on the shared DB layer,
never the reverse. An import of ``app`` from the models, repositories or an
Alembic revision would turn that into a cycle that only works while both
packages happen to share one venv. Sibling of ``test_no_optimizer_import.py``
and ``test_no_agent_stack_import.py``; imports are read from the syntax tree so
``import os, app`` and ``importlib.import_module("app.worker")`` are caught and
a docstring quoting an import is not.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from _import_scan import imported_module_names

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src" / "portopt_db"
_ALEMBIC = _PKG_ROOT / "alembic"


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def find_ingestion_import_violations(root: Path) -> list[str]:
    """Return paths of files that import the ingestion daemon's ``app`` package.

    Args:
        root: Directory tree to scan recursively for ``.py`` files.

    Returns:
        Paths relative to ``root`` of every file with an absolute import whose
        top-level module is ``app``, in discovery order. Empty when clean.
    """
    offending: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            module.split(".", 1)[0] == "app"
            for module in imported_module_names(text, str(path))
        ):
            offending.append(str(path.relative_to(root)))
    return offending


@pytest.mark.parametrize("tree", [_SRC, _ALEMBIC], ids=["src", "alembic"])
def test_when_a_package_tree_is_scanned_then_it_is_non_empty_and_has_no_ingestion_import(
    tree,
):
    assert any(_iter_python_files(tree)), f"{tree} yields no files to scan"
    assert find_ingestion_import_violations(tree) == []


@pytest.mark.parametrize(
    "source",
    [
        "import app\n",
        "from app.services import worker\n",
        "import numpy, app\n",
        "x = 1; import app\n",
        "import importlib\nimportlib.import_module('app.worker')\n",
        "__import__('app')\n",
    ],
)
def test_when_an_ingestion_import_is_injected_then_the_guard_fails(tmp_path, source):
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    assert find_ingestion_import_violations(tmp_path) == ["offender.py"]


@pytest.mark.parametrize(
    "source",
    [
        "app = build_application()\n",
        "from portopt_db.app_state import x\n",
        "import appdirs\n",
        "from . import app\n",
        '"""Example:\n\nfrom app.worker import main\n"""\n',
    ],
)
def test_when_source_only_resembles_an_ingestion_import_then_the_guard_allows_it(
    tmp_path, source
):
    (tmp_path / "clean.py").write_text(source, encoding="utf-8")
    assert find_ingestion_import_violations(tmp_path) == []
