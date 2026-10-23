"""Guard: the ingestion daemon never imports the deep-agent stack.

Principle #3 of the fund roadmap: ``deepagents``/``langgraph`` (and the ``fund``
bridge itself) live ONLY in ``fund/``. ``ingestion`` and ``portopt-db`` stay
clean — they import neither ``optimizer`` (guarded by
``test_no_optimizer_import.py``) NOR the agent runtime. In one shared venv the
whole stack is importable, so this static source-scan is the only thing keeping
a stray ``import deepagents`` out of the daemon. Sibling of the optimizer guard;
same source-blind, injectable-root shape.

``Path(__file__).resolve().parents[3]`` resolves to ``ingestion/`` (hygiene →
unit → tests → ingestion), never ``Path.cwd()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.hygiene._shared_scan import (
    _imported_module_names,
    _iter_python_files,
)

_INGESTION_ROOT = Path(__file__).resolve().parents[3]
_APP_ROOT = _INGESTION_ROOT / "app"
_PYPROJECT_FILE = _INGESTION_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

# Top-level module names of the agent runtime plus the fund bridge. The
# LangChain/LangGraph families are matched by prefix because their imports are
# `langchain_core`, `langchain_ollama`, `langgraph.checkpoint...`; the exact
# match on `fund` keeps `refund` from being mistaken for the bridge package.
_AGENT_EXACT_ROOTS = frozenset({"deepagents", "fund"})
_AGENT_ROOT_PREFIXES = ("langgraph", "langchain")
# Forbidden distributions (substring match on lowered manifest text). `langgraph`
# also catches `langgraph-checkpoint-postgres`; `langchain` catches
# `langchain-ollama`; `portopt-fund` is the fund dist (NOT portopt-core/db).
_FORBIDDEN_DISTS = ("deepagents", "langgraph", "langchain", "portopt-fund")


def _is_agent_module(module: str) -> bool:
    top_level = module.split(".", 1)[0]
    return top_level in _AGENT_EXACT_ROOTS or top_level.startswith(_AGENT_ROOT_PREFIXES)


def find_agent_import_violations(root: Path) -> list[str]:
    """Return one relative path per file that imports the agent stack."""
    offending: list[str] = []
    for path in _iter_python_files(root, exclude=_SELF):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            _is_agent_module(module)
            for module in _imported_module_names(text, str(path))
        ):
            offending.append(str(path.relative_to(root)))
    return offending


def find_forbidden_agent_dependencies(dependencies_text: str) -> list[str]:
    """Return which agent-stack distributions a manifest declares."""
    lowered = dependencies_text.lower()
    return [dist for dist in _FORBIDDEN_DISTS if dist in lowered]


@pytest.mark.criterion("global-6")
def test_when_ingestion_app_is_scanned_then_no_agent_stack_import_is_found():
    assert find_agent_import_violations(_APP_ROOT) == []


@pytest.mark.criterion("global-6")
def test_when_ingestion_pyproject_is_read_then_no_agent_stack_dependency_is_declared():
    content = _PYPROJECT_FILE.read_text(encoding="utf-8")
    assert find_forbidden_agent_dependencies(content) == []


def test_when_a_deepagents_import_is_injected_then_the_guard_fails(tmp_path):
    (tmp_path / "offender.py").write_text("import deepagents\n", encoding="utf-8")
    assert find_agent_import_violations(tmp_path)


def test_when_a_fund_import_is_injected_then_the_guard_fails(tmp_path):
    (tmp_path / "offender.py").write_text(
        "from fund.tools import prices\n", encoding="utf-8"
    )
    assert find_agent_import_violations(tmp_path)


def test_when_a_langgraph_checkpoint_dep_is_injected_then_the_guard_fails():
    violations = find_forbidden_agent_dependencies('"langgraph-checkpoint-postgres",\n')
    assert "langgraph" in violations


def test_when_the_fund_dist_is_injected_then_the_guard_fails():
    assert find_forbidden_agent_dependencies('"portopt-fund",\n') == ["portopt-fund"]


def test_when_a_refund_token_is_scanned_then_the_guard_allows_it(tmp_path):
    """`fund` is matched as an exact module name, so `refund` is not a hit."""
    (tmp_path / "clean.py").write_text("refund = compute_refund()\n", encoding="utf-8")
    assert find_agent_import_violations(tmp_path) == []


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
