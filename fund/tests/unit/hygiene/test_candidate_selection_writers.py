"""Guard: only the user's approve or weight edit records the choice that is placed.

A run's candidate set is computed by the allocator, but which weights become the paper
ticket is the user's decision alone, recorded in one of two ways. An approve that
names a candidate writes it through ``AgentRunRepository.append_candidate_selection``,
reached through ``fund.agents.graph_candidates.record_candidate_selection`` /
``approve_weights``, which only ``fund.agents.graph._approved_weights`` calls. A weight
edit (``fund.actions._apply_edit``) writes an allocator ``optimize_portfolio`` row whose
``hitl_decision`` carries the ``edit`` kind, and a later approve without a candidate id
places those weights. Both record a passing risk verdict, without which
``place_orders`` refuses the weights.

The scan parses the syntax tree of every ``.py`` file under ``fund/src`` and reports
each call of one of those writers outside its listed callers, and each string literal
equal to the ``select_candidate`` kind outside the repository that writes it. It also
records, as ``(module, enclosing function, what)``, every reference to a choice kind
(``EDIT_DECISION``, ``SELECT_CANDIDATE_DECISION``), a kind literal put in a
``"decision"`` slot, a renamed import of a kind, and a raw write of the choice row or
the risk-verdict row (a call passing ``step="optimize_portfolio"``, ``agent="risk"`` or
``step="risk_check"``). Each such site is listed below with its reason, and only the
two writers may put a kind in a ``"decision"`` slot. A new site fails the guard until
someone writes down why, and an entry nothing matches any more fails it too. So no
bound tool, subagent, scheduler, worker, TUI or Ask module can record a choice. It
imports no implementation module.

What it cannot see: a kind held under another name first (``kind = "edit"``) or reached
through a computed name, which no syntax-tree scan resolves.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

# hygiene -> unit -> tests -> fund, then the src layout's package root.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "fund"
_MODULE_SCOPE = "<module>"
_ACTIONS = "actions.py"
_CLI = "cli.py"
_GRAPH = "agents/graph.py"
_GRAPH_CANDIDATES = "agents/graph_candidates.py"
_REPOSITORY = "audit/repository.py"

# Each writer of the user's choice or its verdict and the only modules that may call it.
_ALLOWED_CALLERS: dict[str, frozenset[str]] = {
    "append_candidate_selection": frozenset({_GRAPH_CANDIDATES}),
    "record_candidate_selection": frozenset({_GRAPH_CANDIDATES}),
    "approve_weights": frozenset({_GRAPH}),
    "_approved_weights": frozenset({_GRAPH}),
    "append_risk_verdict": frozenset({_ACTIONS, _GRAPH_CANDIDATES}),
}
_SELECTION_KIND = "select_candidate"

_EDIT = "EDIT_DECISION"
_SELECT = "SELECT_CANDIDATE_DECISION"
_KIND_NAMES = frozenset({_EDIT, _SELECT})
_KIND_VALUES = frozenset({"edit", _SELECTION_KIND})
_KIND_SPELLINGS = _KIND_NAMES | {repr(value) for value in _KIND_VALUES}
_DECISION_KEY = "decision"
# Keyword arguments that write the row a choice or a risk verdict is read from.
_ROW_KEYWORDS = frozenset(
    {("step", "optimize_portfolio"), ("agent", "risk"), ("step", "risk_check")}
)
_EDIT_ROW = "step='optimize_portfolio'"

_CHOICE_SITES: dict[tuple[str, str, str], str] = {
    (_ACTIONS, "_apply_edit", _EDIT): (
        "writes the weights the user typed, once they passed the constraint check"
    ),
    (_ACTIONS, "_apply_edit", _EDIT_ROW): "the weight edit's choice row",
    (_REPOSITORY, "AgentRunRepository.append_candidate_selection", _SELECT): (
        "writes the candidate the user named at the approve"
    ),
    (_REPOSITORY, "AgentRunRepository.append_risk_verdict", "agent='risk'"): (
        "the one writer of a risk verdict"
    ),
    (_REPOSITORY, "AgentRunRepository.append_risk_verdict", "step='risk_check'"): (
        "the one writer of a risk verdict"
    ),
    (_REPOSITORY, _MODULE_SCOPE, _EDIT): (
        "the kinds current_adviser_choice accepts as the user's choice"
    ),
    (_REPOSITORY, _MODULE_SCOPE, _SELECT): (
        "the kinds current_adviser_choice accepts as the user's choice"
    ),
    (_GRAPH_CANDIDATES, "_edit_in_force", _EDIT): (
        "reads the kind: an approve without an id places only an edit in force"
    ),
    (_CLI, "_adviser_choice", _EDIT): (
        "reads the kind so `fund report` prints edited weights, not a candidate"
    ),
}
# The only places a kind may be written into a ``"decision"`` slot.
_KIND_WRITERS = frozenset(
    {
        (_ACTIONS, "_apply_edit", _EDIT),
        (_REPOSITORY, "AgentRunRepository.append_candidate_selection", _SELECT),
    }
)


@dataclass(frozen=True)
class _Site:
    """One reference to a choice kind or one raw write of a choice or verdict row."""

    module: str
    scope: str
    what: str
    writes_kind: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.module, self.scope, self.what)


def _callee(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_decision_key(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == _DECISION_KEY


def _slot_values(node: ast.AST) -> Iterator[ast.AST]:
    """The expressions ``node`` puts under a ``"decision"`` key, if any."""
    if isinstance(node, ast.Dict):
        pairs = zip(node.keys, node.values, strict=True)
        yield from (value for key, value in pairs if _is_decision_key(key))
    elif isinstance(node, ast.keyword) and node.arg == _DECISION_KEY:
        yield node.value
    elif isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is not None and any(
            isinstance(t, ast.Subscript) and _is_decision_key(t.slice) for t in targets
        ):
            yield node.value
    elif isinstance(node, ast.Call) and node.args and _is_decision_key(node.args[0]):
        yield from node.args[1:]  # ``setdefault("decision", ...)`` and kin


def _decision_slot_nodes(tree: ast.AST) -> set[int]:
    return {
        id(part)
        for node in ast.walk(tree)
        for value in _slot_values(node)
        for part in ast.walk(value)
    }


def _referenced_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.attr
    return None


def _what_at(node: ast.AST, slots: set[int]) -> Iterator[str]:
    if isinstance(node, ast.ImportFrom):
        # A renamed import would hide every later reference from the name match.
        for alias in node.names:
            if alias.name in _KIND_NAMES and alias.asname not in (None, alias.name):
                yield f"{alias.name} as {alias.asname}"
    elif _referenced_name(node) in _KIND_NAMES:
        yield str(_referenced_name(node))
    elif isinstance(node, ast.Constant) and node.value in _KIND_VALUES:
        if id(node) in slots:
            yield repr(node.value)
    elif isinstance(node, ast.Call):
        for passed in node.keywords:
            value = passed.value
            if isinstance(value, ast.Constant) and (passed.arg, value.value) in (
                _ROW_KEYWORDS
            ):
                yield f"{passed.arg}={value.value!r}"


def choice_sites(source: str, module: str) -> set[_Site]:
    """Every choice-kind reference and raw choice or verdict write in ``source``.

    Args:
        source: Python source text.
        module: Its path under the package root, used as the site's module.

    Returns:
        One site per ``(enclosing function, what)``; ``writes_kind`` is set when a
        kind sits in a ``"decision"`` slot.
    """
    tree = ast.parse(source, filename=module)
    slots = _decision_slot_nodes(tree)
    sites: set[_Site] = set()

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        """Collect the sites under ``node``, ``scope`` naming its enclosing defs."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                visit(child, (*scope, child.name))
                continue
            label = ".".join(scope) or _MODULE_SCOPE
            in_slot = id(child) in slots
            for what in _what_at(child, slots):
                sites.add(_Site(module, label, what, writes_kind=in_slot))
            visit(child, scope)

    visit(tree, ())
    return sites


def writer_calls(source: str, module: str) -> list[tuple[str, str]]:
    """``(module, writer)`` for each call in ``source`` of a listed writer."""
    return [
        (module, str(_callee(node)))
        for node in ast.walk(ast.parse(source, filename=module))
        if isinstance(node, ast.Call) and _callee(node) in _ALLOWED_CALLERS
    ]


def _package_sources() -> Iterator[tuple[str, str]]:
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        yield path.relative_to(_PACKAGE_ROOT).as_posix(), path.read_text("utf-8")


def _outside_callers(calls: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (module, writer)
        for module, writer in calls
        if module not in _ALLOWED_CALLERS[writer]
    ]


def _unlisted(sites: set[_Site]) -> list[tuple[str, str, str]]:
    return sorted(site.key for site in sites if site.key not in _CHOICE_SITES)


def _kind_writes(sites: set[_Site]) -> list[tuple[str, str, str]]:
    return sorted(site.key for site in sites if site.writes_kind)


@pytest.mark.criterion("REQ-FUND-051")
def test_only_the_approve_chain_records_a_candidate_selection() -> None:
    """Validates fund.agents.graph_candidates, fund.actions and fund.audit.repository:
    the writers of the user's candidate choice and of a risk verdict are called only
    by their listed callers, and the ``select_candidate`` kind is spelled only by the
    repository that writes it."""
    calls: list[tuple[str, str]] = []
    spelled: list[str] = []
    for module, source in _package_sources():
        calls += writer_calls(source, module)
        spelled += [
            module
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and node.value == _SELECTION_KIND
        ]

    assert _outside_callers(calls) == []
    assert {writer for _, writer in calls} == set(_ALLOWED_CALLERS)
    assert spelled == [_REPOSITORY]


@pytest.mark.criterion("REQ-FUND-051")
def test_only_the_weight_edit_and_the_selection_write_a_choice_kind() -> None:
    """Validates fund.actions._apply_edit, AgentRunRepository.append_candidate_selection
    and AgentRunRepository.append_risk_verdict: every choice-kind reference and raw
    choice or verdict write under fund/src is listed with its reason, every listed
    entry still exists, and only those two writers put a kind in a decision slot."""
    sites = set().union(*(choice_sites(src, mod) for mod, src in _package_sources()))

    assert _unlisted(sites) == []
    assert sorted(set(_CHOICE_SITES) - {site.key for site in sites}) == []
    assert all(reason.strip() for reason in _CHOICE_SITES.values())
    assert _kind_writes(sites) == sorted(_KIND_WRITERS)


_SNEAK = "_sneak"
_INJECTED_CHOICES = [
    (
        "worker.py",
        "def _sneak(repo, rid):\n    repo.append_decision(rid, agent='allocator', "
        "step='optimize_portfolio', hitl_decision={'decision': EDIT_DECISION})\n",
        [(_SNEAK, _EDIT), (_SNEAK, _EDIT_ROW)],
    ),
    (
        "scheduler.py",
        "def _sneak(rid):\n    return {'decision': 'edit', 'human_override': True}\n",
        [(_SNEAK, "'edit'")],
    ),
    (
        "agents/toolsets.py",
        "def _sneak(payload):\n    payload['decision'] = SELECT_CANDIDATE_DECISION\n",
        [(_SNEAK, _SELECT)],
    ),
    (
        "agents/ask_agent.py",
        "def _sneak():\n"
        "    return dict(decision=fund.audit.repository.EDIT_DECISION)\n",
        [(_SNEAK, _EDIT)],
    ),
    (
        "scheduler.py",
        "def _sneak(repo, rid):\n    repo.append_decision(rid, agent='risk', "
        "step='risk_check', llm_response='{}')\n",
        [(_SNEAK, "agent='risk'"), (_SNEAK, "step='risk_check'")],
    ),
    (
        "agents/ask_agent.py",
        "from fund.audit.repository import EDIT_DECISION as KIND\n",
        [(_MODULE_SCOPE, "EDIT_DECISION as KIND")],
    ),
]
_INJECTED_WRITER_CALLS = [
    ("scheduler.py", "append_risk_verdict"),
    ("worker.py", "append_risk_verdict"),
    ("agents/toolsets.py", "append_risk_verdict"),
    ("agents/ask_agent.py", "append_candidate_selection"),
]


def _original(module: str) -> str:
    return (_PACKAGE_ROOT / module).read_text(encoding="utf-8")


def _added_sites(module: str, injected: str) -> set[_Site]:
    """The sites ``injected`` adds when appended to a copy of ``module``'s source."""
    original = _original(module)
    return choice_sites(f"{original}\n\n{injected}", module) - choice_sites(
        original, module
    )


@pytest.mark.criterion("REQ-FUND-051")
@pytest.mark.parametrize(("module", "injected", "expected"), _INJECTED_CHOICES)
def test_the_guard_flags_a_choice_written_outside_the_writers(
    module: str, injected: str, expected: list[tuple[str, str]]
) -> None:
    """Validates fund.scheduler, fund.worker, fund.agents.toolsets and
    fund.agents.ask_agent as the guard reads them: a choice kind, an edit row or a
    risk-verdict row added to a copy of the module is a new site with no listed
    reason, and a kind it puts in a decision slot is a write no writer made."""
    added = _added_sites(module, injected)

    assert _unlisted(added) == sorted((module, scope, what) for scope, what in expected)
    assert _kind_writes(added) == [
        key for key in _unlisted(added) if key[2] in _KIND_SPELLINGS
    ]


@pytest.mark.criterion("REQ-FUND-051")
@pytest.mark.parametrize(("module", "writer"), _INJECTED_WRITER_CALLS)
def test_the_guard_flags_a_writer_called_outside_its_callers(
    module: str, writer: str
) -> None:
    """Validates fund.scheduler, fund.worker, fund.agents.toolsets and
    fund.agents.ask_agent as the guard reads them: a call of a choice or verdict
    writer added to a copy of the module is outside the writer's listed callers."""
    injected = f"def _sneak(repo, rid, w):\n    repo.{writer}(rid, weights=w)\n"
    source = f"{_original(module)}\n\n{injected}"

    assert (module, writer) in _outside_callers(writer_calls(source, module))


@pytest.mark.criterion("REQ-FUND-051")
def test_the_guard_ignores_a_decision_that_records_no_choice() -> None:
    """Validates fund.scheduler as the guard reads it: a decision slot holding another
    kind, or a kind only compared against, adds no write site."""
    added = _added_sites(
        "scheduler.py",
        "def _sneak(row):\n    note = {'decision': 'reject', 'via': 'edit'}\n"
        "    return note, row.get('decision') == 'edit'\n",
    )

    assert added == set()
