"""Guard: portopt-db must never import the deep-agent stack or declare it.

portopt-db is the shared data layer, sitting *below* both ingestion and the
future ``fund`` bridge. It must import neither ``optimizer`` (guarded by
``test_no_optimizer_import.py``) NOR the agent runtime (``deepagents`` /
``langgraph`` / ``langchain``) NOR the ``fund`` package. Source-blind scan of
the package source + its pyproject; sibling of the optimizer guard.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from _import_scan import imported_module_names

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src" / "portopt_db"
_PYPROJECT = _PKG_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

# The LangChain/LangGraph families are matched by prefix because their imports
# are `langchain_core`, `langchain_ollama`, `langgraph.checkpoint...`; the exact
# match on `fund` keeps `refund` from being mistaken for the bridge package.
_AGENT_EXACT_ROOTS = frozenset({"deepagents", "fund"})
_AGENT_ROOT_PREFIXES = ("langgraph", "langchain")
_FORBIDDEN_DISTS = ("deepagents", "langgraph", "langchain", "portopt-fund")


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or path.resolve() == _SELF:
            continue
        yield path


def _is_agent_module(module: str) -> bool:
    top_level = module.split(".", 1)[0]
    return top_level in _AGENT_EXACT_ROOTS or top_level.startswith(_AGENT_ROOT_PREFIXES)


def find_agent_import_violations(root: Path) -> list[str]:
    """Scan Python files under root for imports of the forbidden agent stack.

    Args:
        root: Directory tree to scan recursively for ``*.py`` files.

    Returns:
        Paths of offending files, relative to root, in discovery order.
        Empty list means no violations.
    """
    offending: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            _is_agent_module(module)
            for module in imported_module_names(text, str(path))
        ):
            offending.append(str(path.relative_to(root)))
    return offending


def find_forbidden_agent_dependencies(text: str) -> list[str]:
    """Return forbidden distribution names found in the given pyproject text.

    Args:
        text: Raw contents of a ``pyproject.toml`` to inspect.

    Returns:
        Subset of ``_FORBIDDEN_DISTS`` that appear (case-insensitively) in text.
        Empty list means no forbidden deps declared.
    """
    lowered = text.lower()
    return [dist for dist in _FORBIDDEN_DISTS if dist in lowered]


def test_when_src_is_scanned_then_no_agent_stack_import_is_found():
    assert find_agent_import_violations(_SRC) == []


def test_when_pyproject_is_read_then_no_agent_stack_dependency_is_declared():
    assert (
        find_forbidden_agent_dependencies(_PYPROJECT.read_text(encoding="utf-8")) == []
    )


def test_when_a_deepagents_import_is_injected_then_the_guard_fails(tmp_path):
    (tmp_path / "offender.py").write_text(
        "from deepagents import x\n", encoding="utf-8"
    )
    assert find_agent_import_violations(tmp_path)


def test_when_the_fund_dist_is_injected_then_the_guard_fails():
    assert find_forbidden_agent_dependencies('"portopt-fund",\n') == ["portopt-fund"]


@pytest.mark.parametrize(
    "source",
    [
        "from langchain_core.messages import BaseMessage\n",
        "import langchain_ollama\n",
        "import langchain_ollama as lo\n",
        "import numpy, langgraph\n",
        "from langgraph.checkpoint.postgres import PostgresSaver\n",
        "import os; import deepagents\n",
    ],
)
def test_when_a_non_canonical_agent_import_is_injected_then_the_guard_fails(
    tmp_path, source
):
    """`langchain_*` distributions and multi-name imports are agent-stack too."""
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    assert find_agent_import_violations(tmp_path) == ["offender.py"]


def test_when_a_docstring_mentions_an_agent_import_then_the_guard_allows_it(tmp_path):
    (tmp_path / "clean.py").write_text(
        '"""Example:\n\nfrom deepagents import create_deep_agent\n"""\n',
        encoding="utf-8",
    )
    assert find_agent_import_violations(tmp_path) == []
