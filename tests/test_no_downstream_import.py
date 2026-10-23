"""Guard: the optimizer library (``portopt-core``) never imports the packages
built on top of it.

The dependency arrows point one way: ``fund`` and ``ingestion`` (import package
``app``) consume ``optimizer``, never the reverse, and the agent runtime
(``deepagents`` / ``langgraph*`` / ``langchain*``) lives only in ``fund``.
In one shared venv all of them are importable, so this scan is the only thing
that keeps the published ``portopt-core`` wheel free of an accidental import of
a package it does not declare. Sibling of ``test_no_portopt_db_import.py``,
which holds the database direction.

Imports are read from the syntax tree, so ``import os, fund`` and
``importlib.import_module("app.worker")`` are caught and a docstring quoting an
import is not.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ._import_scan import imported_module_names

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPTIMIZER_SRC = _REPO_ROOT / "packages" / "portopt-core" / "optimizer"

# `app` is the ingestion daemon's import package. The LangChain/LangGraph
# families are matched by prefix because their imports are `langchain_core`,
# `langchain_ollama`, `langgraph.checkpoint...`.
_FORBIDDEN_EXACT_ROOTS = frozenset({"fund", "app", "deepagents"})
_FORBIDDEN_ROOT_PREFIXES = ("langgraph", "langchain")


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _is_downstream_module(module: str) -> bool:
    top_level = module.split(".", 1)[0]
    return top_level in _FORBIDDEN_EXACT_ROOTS or top_level.startswith(
        _FORBIDDEN_ROOT_PREFIXES
    )


def find_downstream_import_violations(root: Path) -> list[str]:
    """Return paths of files that import a package layered above the library.

    Args:
        root: Directory tree to scan recursively for ``.py`` files.

    Returns:
        Paths relative to ``root`` of every file importing ``fund``, ``app``,
        ``deepagents`` or a ``langgraph*`` / ``langchain*`` module, in
        discovery order. Empty when clean.
    """
    offending: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            _is_downstream_module(module)
            for module in imported_module_names(text, str(path))
        ):
            offending.append(str(path.relative_to(root)))
    return offending


def test_when_optimizer_src_is_scanned_then_no_downstream_import_is_found():
    # A missing path makes rglob yield nothing, silently disabling the guard, so
    # require real files to be visited — a future layout move then fails loudly.
    assert any(_iter_python_files(_OPTIMIZER_SRC)), (
        f"no optimizer source found at {_OPTIMIZER_SRC}"
    )
    assert find_downstream_import_violations(_OPTIMIZER_SRC) == []


@pytest.mark.parametrize(
    "source",
    [
        "import fund\n",
        "from fund.tools import prices\n",
        "from app.services import worker\n",
        "import deepagents\n",
        "from langgraph.graph import StateGraph\n",
        "from langchain_core.messages import BaseMessage\n",
        "import langchain_ollama as lo\n",
        "import numpy, fund\n",
        "x = 1; import app\n",
        "import importlib\nimportlib.import_module('app.worker')\n",
        "__import__('deepagents')\n",
    ],
)
def test_when_a_downstream_import_is_injected_then_the_guard_fails(tmp_path, source):
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    assert find_downstream_import_violations(tmp_path) == ["offender.py"]


@pytest.mark.parametrize(
    "source",
    [
        "refund = compute_refund()\n",
        "app = typer.Typer()\n",
        "from optimizer.moments import build_prior\n",
        "from . import app\n",
        '"""Example:\n\nfrom fund import run\n"""\n',
    ],
)
def test_when_source_only_resembles_a_downstream_import_then_the_guard_allows_it(
    tmp_path, source
):
    (tmp_path / "clean.py").write_text(source, encoding="utf-8")
    assert find_downstream_import_violations(tmp_path) == []
