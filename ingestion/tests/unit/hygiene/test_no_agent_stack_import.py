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

import re
from pathlib import Path

import pytest

from tests.unit.hygiene._shared_scan import _iter_python_files

_INGESTION_ROOT = Path(__file__).resolve().parents[3]
_APP_ROOT = _INGESTION_ROOT / "app"
_PYPROJECT_FILE = _INGESTION_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

# `import fund` / `from langgraph ...` etc. `\b` keeps `refund`/`langchainx`
# from matching a bare module token; the alternation is the agent runtime + the
# fund bridge package.
_AGENT_IMPORT_PATTERN = re.compile(
    r"^\s*(from|import)\s+(deepagents|langgraph|langchain|fund)\b", re.MULTILINE
)
# Forbidden distributions (substring match on lowered manifest text). `langgraph`
# also catches `langgraph-checkpoint-postgres`; `langchain` catches
# `langchain-ollama`; `portopt-fund` is the fund dist (NOT portopt-core/db).
_FORBIDDEN_DISTS = ("deepagents", "langgraph", "langchain", "portopt-fund")


def find_agent_import_violations(root: Path) -> list[str]:
    """Return one relative path per file that imports the agent stack."""
    offending: list[str] = []
    for path in _iter_python_files(root, exclude=_SELF):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _AGENT_IMPORT_PATTERN.search(text):
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
    """`\\bfund\\b` must not match `refund` — a bare word, not a module import."""
    (tmp_path / "clean.py").write_text("refund = compute_refund()\n", encoding="utf-8")
    assert find_agent_import_violations(tmp_path) == []
