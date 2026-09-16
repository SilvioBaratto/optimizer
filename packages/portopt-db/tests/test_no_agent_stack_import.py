"""Guard: portopt-db must never import the deep-agent stack or declare it.

portopt-db is the shared data layer, sitting *below* both ingestion and the
future ``fund`` bridge. It must import neither ``optimizer`` (guarded by
``test_no_optimizer_import.py``) NOR the agent runtime (``deepagents`` /
``langgraph`` / ``langchain``) NOR the ``fund`` package. Source-blind scan of
the package source + its pyproject; sibling of the optimizer guard.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src" / "portopt_db"
_PYPROJECT = _PKG_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

_AGENT_IMPORT = re.compile(
    r"^\s*(from|import)\s+(deepagents|langgraph|langchain|fund)\b", re.MULTILINE
)
_FORBIDDEN_DISTS = ("deepagents", "langgraph", "langchain", "portopt-fund")


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or path.resolve() == _SELF:
            continue
        yield path


def find_agent_import_violations(root: Path) -> list[str]:
    offending: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _AGENT_IMPORT.search(text):
            offending.append(str(path.relative_to(root)))
    return offending


def find_forbidden_agent_dependencies(text: str) -> list[str]:
    lowered = text.lower()
    return [dist for dist in _FORBIDDEN_DISTS if dist in lowered]


def test_when_src_is_scanned_then_no_agent_stack_import_is_found():
    assert find_agent_import_violations(_SRC) == []


def test_when_pyproject_is_read_then_no_agent_stack_dependency_is_declared():
    assert find_forbidden_agent_dependencies(_PYPROJECT.read_text(encoding="utf-8")) == []


def test_when_a_deepagents_import_is_injected_then_the_guard_fails(tmp_path):
    (tmp_path / "offender.py").write_text("from deepagents import x\n", encoding="utf-8")
    assert find_agent_import_violations(tmp_path)


def test_when_the_fund_dist_is_injected_then_the_guard_fails():
    assert find_forbidden_agent_dependencies('"portopt-fund",\n') == ["portopt-fund"]
