"""Guard: only the adviser's approve can record which candidate is placed.

A run's candidate set is computed by the allocator, but which candidate becomes the
paper ticket is the adviser's decision alone. The choice is written by
``AgentRunRepository.append_candidate_selection``, reached through
``fund.agents.graph_candidates.record_candidate_selection`` /
``approve_weights``, which only ``fund.agents.graph._approved_weights`` calls on an
approve. The scan parses the syntax tree of every ``.py`` file under ``fund/src`` and
reports each call of one of those names, and each string literal equal to the
``select_candidate`` decision kind, outside that chain, so no bound tool, subagent,
scheduler, worker, TUI or Ask module can record a choice. It imports no
implementation module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# hygiene -> unit -> tests -> fund, then the src layout's package root.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "fund"
_GRAPH = "agents/graph.py"
_GRAPH_CANDIDATES = "agents/graph_candidates.py"
_REPOSITORY = "audit/repository.py"

# Each writer of the adviser's choice and the only modules allowed to call it.
_ALLOWED_CALLERS: dict[str, frozenset[str]] = {
    "append_candidate_selection": frozenset({_GRAPH_CANDIDATES}),
    "record_candidate_selection": frozenset({_GRAPH_CANDIDATES}),
    "approve_weights": frozenset({_GRAPH}),
    "_approved_weights": frozenset({_GRAPH}),
}
_SELECTION_KIND = "select_candidate"


def _callee(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _scan() -> tuple[list[tuple[str, str]], list[str]]:
    """``(module, writer)`` per writer call, and the modules spelling the kind."""
    calls: list[tuple[str, str]] = []
    spelled: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        module = path.relative_to(_PACKAGE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee(node) in _ALLOWED_CALLERS:
                calls.append((module, str(_callee(node))))
            elif isinstance(node, ast.Constant) and node.value == _SELECTION_KIND:
                spelled.append(module)
    return calls, spelled


@pytest.mark.criterion("REQ-FUND-051")
def test_only_the_adviser_path_records_a_candidate_selection() -> None:
    """Validates fund.agents.graph_candidates and fund.audit.repository: the writers
    of the adviser's candidate choice are called only along the approve chain, and
    the ``select_candidate`` decision kind is spelled only by the repository that
    writes it."""
    calls, spelled = _scan()

    outside = [
        (module, writer)
        for module, writer in calls
        if module not in _ALLOWED_CALLERS[writer]
    ]
    assert outside == []
    assert {writer for _, writer in calls} == set(_ALLOWED_CALLERS)
    assert spelled == [_REPOSITORY]
