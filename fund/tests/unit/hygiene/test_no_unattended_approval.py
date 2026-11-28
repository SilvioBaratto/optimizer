"""Guard: a paused run is resumed with approve only from a human entry point.

A paused run (a rebalance at ``place_orders``, a profile at ``save_profile``) moves on
only through a ``Command(resume=...)``. The scan parses the syntax tree of every
``.py`` file under ``fund/src`` and records each reference to a callable on that
resume chain, from the ``Command`` up to the CLI command and the TUI workers that
start it, as ``(module, enclosing function, callee)``. A reference whose decision is
a string literal that cannot approve (``"reject"``, ``"profile"``) is left out.

Every other reference must be listed below with a one-line reason: a human entry
point, ``apply_action``'s own handlers, or a link of the resume mechanism that only
forwards the decision it is handed. A new reference fails the guard until someone
writes down why a human stands behind it, and an entry nothing matches any more fails
it too. Entries carry no line numbers, so they survive edits elsewhere in a file. The
daemon, the agent graphs with their toolsets and subagents, and the Ask agent must
hold no reference at all. It imports no implementation module.

What it cannot see: a callee reached through a computed name (``getattr`` with a
string), which no syntax-tree scan resolves; and an approve that a listed scope of the
resume mechanism passes through a named constant (``APPROVE = "approve"``), because the
check that the mechanism only forwards a decision reads string literals, not constants.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

# hygiene -> unit -> tests -> fund, then the src layout's package root.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "fund"
_MODULE_SCOPE = "<module>"

# The decisions that end in an approval: a plain approve, or one with edited weights.
_APPROVING = frozenset({"approve", "edit"})

# Each callable on the resume chain and where a call passes its decision, as
# ``(keyword, position)``; ``None`` when every call approves (an edit). ``Command``
# carries the decision inside its ``resume`` payload and is read separately.
# ``ACTION_HANDLERS`` is apply_action's dispatch table, keyed by the action kind: it is
# only ever subscripted, never called, so each reference to it counts as forwarding.
_DECISION_ARGUMENT: dict[str, tuple[str, int | None] | None] = {
    "Command": None,
    "ACTION_HANDLERS": ("kind", None),
    "_resume_command": ("decision", 0),
    "_resume_agent": ("decision", 2),
    "_resume_at_gate": ("decision", 2),
    "_resume_gate": ("decision", None),
    "resume": ("decision", 0),
    "resume_fund": ("decision", 1),
    "resume_profiler": ("decision", 1),
    "_resume_profiler_gate": ("decision", 1),
    "resume_run": ("decision", 1),
    "_apply_resume": ("decision", None),
    "_handle_resume": ("kind", 0),
    "_apply_edit": None,
    "_handle_edit": None,
    "apply_action": ("action", 0),
    "_dispatch": ("action", 0),
    "_action_worker": ("decision", 2),
    "_edit_worker": None,
}

_CLI = "cli.py"
_DECISIONS = "tui/_mixins/decisions.py"
_ACTIONS = "actions.py"
_GRAPH = "agents/graph.py"
_GRAPH_HITL = "agents/graph_hitl.py"
_PROFILER = "agents/profiler.py"

_HUMAN_ENTRY_POINTS: dict[tuple[str, str, str], str] = {
    (_CLI, "_resume", "apply_action"): (
        "`fund approve` / `fund reject`: the user types the run id and the candidate"
    ),
    (_DECISIONS, "DecisionsMixin._decide", "_action_worker"): (
        "the `a` / `r` key on the gate the user selected; a rebalance approve goes "
        "through the candidate confirm instead"
    ),
    (_DECISIONS, "DecisionsMixin._approve_candidate._confirmed", "_action_worker"): (
        "Confirm on the candidate confirm, which names the id, weights and measures"
    ),
    (_DECISIONS, "DecisionsMixin.action_edit_decision._edited", "_edit_worker"): (
        "Commit in the weight editor the user filled in"
    ),
    ("tui/_mixins/ask.py", "AskMixin._dispatch_ask_action", "_action_worker"): (
        "an Ask action after the user's Confirm; Ask refuses a rebalance approve"
    ),
    (_DECISIONS, "DecisionsMixin._action_worker", "apply_action"): (
        "the TUI action worker, started only by the key, confirm and Ask entries"
    ),
    (_DECISIONS, "DecisionsMixin._edit_worker", "apply_action"): (
        "the TUI edit worker, started only by the weight editor's Commit"
    ),
    (_ACTIONS, "apply_action", "_dispatch"): (
        "hands its dispatcher the action apply_action was called with"
    ),
    (_ACTIONS, "_dispatch", "ACTION_HANDLERS"): (
        "runs the handler of the action apply_action was called with"
    ),
    (_ACTIONS, _MODULE_SCOPE, "_handle_resume"): (
        "ACTION_HANDLERS: apply_action routes approve/reject/respond/killswitch here"
    ),
    (_ACTIONS, _MODULE_SCOPE, "_handle_edit"): (
        "ACTION_HANDLERS: apply_action routes edit here"
    ),
    (_ACTIONS, "_handle_resume", "_apply_resume"): (
        "forwards the decision apply_action was called with"
    ),
    (_ACTIONS, "_handle_edit", "_apply_edit"): (
        "apply_action's edit: approves with the weights the user edited"
    ),
    (_ACTIONS, "_apply_resume", "resume_run"): (
        "resumes with the user's decision once its audit row is written"
    ),
    (_ACTIONS, "_apply_edit", "resume_run"): (
        "approves once the user's edited weights passed the mandate check"
    ),
    (_GRAPH, "resume_run", "resume_fund"): "forwards a rebalance gate's decision",
    (_GRAPH, "resume_run", "_resume_profiler_gate"): (
        "forwards a profile gate's decision"
    ),
    (_GRAPH, "_resume_profiler_gate", "resume_profiler"): "forwards the decision",
    (_GRAPH, "resume_fund", "_resume_gate"): "forwards the decision",
    (_GRAPH, "FundRun.resume", "_resume_gate"): (
        "the in-process run handle; no fund module calls `.resume`"
    ),
    (_GRAPH, "_resume_gate", "_resume_agent"): "forwards the decision",
    (_GRAPH_HITL, "_resume_agent", "_resume_command"): "forwards the decision",
    (_GRAPH_HITL, "_resume_command", "Command"): (
        "builds the Command from the decision it is handed"
    ),
    (_PROFILER, "ProfilerRun.resume", "_resume_at_gate"): (
        "the in-process profile handle; no fund module calls `.resume`"
    ),
    (_PROFILER, "resume_profiler", "_resume_at_gate"): "forwards the decision",
    (_PROFILER, "_resume_at_gate", "Command"): (
        "builds the Command from the decision it is handed"
    ),
}

# The daemon and its sweeps run with nobody at the keyboard, and a model drives the
# agent graphs, their tools and subagents and the Ask agent: none may reach a resume.
_UNATTENDED = ("scheduler.py", "worker.py", "tools/", "agents/")
# The resume mechanism lives among the agent modules; it may only forward a decision.
_RESUME_MECHANISM = frozenset({_GRAPH, _GRAPH_HITL, _PROFILER})


@dataclass(frozen=True)
class _Site:
    """One reference to a callable on the resume chain."""

    module: str
    scope: str
    callee: str
    supplies_approve: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.module, self.scope, self.callee)


def _referenced_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.attr
    return None


def _literal_choices(node: ast.expr | None) -> set[str] | None:
    """The strings ``node`` can evaluate to, or ``None`` when it is not a literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        body, orelse = _literal_choices(node.body), _literal_choices(node.orelse)
        return None if body is None or orelse is None else body | orelse
    return None


def _argument(call: ast.Call, keyword: str, position: int | None) -> ast.expr | None:
    for passed in call.keywords:
        if passed.arg == keyword:
            return passed.value
    if position is None or position >= len(call.args):
        return None
    if any(isinstance(arg, ast.Starred) for arg in call.args[: position + 1]):
        return None
    return call.args[position]


def _command_approval(node: ast.AST) -> bool | None:
    """Whether a ``Command`` supplies an approve; ``None`` when it carries no resume.

    A ``**`` mapping and a positional argument may hold the ``resume`` payload, so they
    count as one; only named keywords other than ``resume`` are known not to.
    """
    if not isinstance(node, ast.Call):
        return None  # a type annotation: no decision
    payloads = [kw.value for kw in node.keywords if kw.arg in ("resume", None)]
    payloads.extend(node.args)
    if not payloads:
        return None  # a ``Command(goto=...)`` or ``Command(update=...)``
    return any(
        isinstance(part, ast.Constant) and part.value in _APPROVING
        for payload in payloads
        for part in ast.walk(payload)
    )


def _approval(node: ast.AST, callee: str) -> bool | None:
    """Whether a reference supplies an approve itself; ``None`` when it cannot approve.

    A reference handed on as a value (a ``partial``, a handler table) and a decision
    that is no literal both count, as forwarding: the scan cannot tell what they pass.
    """
    if callee == "Command":
        return _command_approval(node)
    where = _DECISION_ARGUMENT[callee]
    if where is None:
        return True
    if not isinstance(node, ast.Call):
        return False
    choices = _literal_choices(_argument(node, *where))
    if choices is None:
        return False
    return True if choices & _APPROVING else None


def _sites_at(
    node: ast.AST, module: str, scope: str, call_targets: set[int]
) -> Iterator[_Site]:
    if isinstance(node, ast.ImportFrom):
        # A renamed import would hide every later call from the name match.
        for alias in node.names:
            if alias.name in _DECISION_ARGUMENT and alias.asname not in (
                None,
                alias.name,
            ):
                yield _Site(module, scope, alias.name, supplies_approve=False)
        return
    target = node.func if isinstance(node, ast.Call) else node
    callee = _referenced_name(target)
    if callee not in _DECISION_ARGUMENT or id(node) in call_targets:
        return
    approval = _approval(node, callee)
    if approval is not None:
        yield _Site(module, scope, callee, approval)


def approval_sites(source: str, module: str) -> set[_Site]:
    """Every reference in ``source`` that can resume a paused run with approve.

    Args:
        source: Python source text.
        module: Its path under the package root, used as the site's module.

    Returns:
        One site per ``(enclosing function, callee)`` with a decision that may
        approve; a call is examined once, never again through its callee name.
    """
    tree = ast.parse(source, filename=module)
    call_targets = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    sites: set[_Site] = set()

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        """Collect the sites under ``node``, ``scope`` naming its enclosing defs."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                visit(child, (*scope, child.name))
                continue
            label = ".".join(scope) or _MODULE_SCOPE
            sites.update(_sites_at(child, module, label, call_targets))
            visit(child, scope)

    visit(tree, ())
    return sites


def _package_sites() -> set[_Site]:
    sites: set[_Site] = set()
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        module = path.relative_to(_PACKAGE_ROOT).as_posix()
        sites |= approval_sites(path.read_text(encoding="utf-8"), module)
    return sites


def _unlisted(sites: Iterable[_Site]) -> list[tuple[str, str, str]]:
    return sorted(site.key for site in sites if site.key not in _HUMAN_ENTRY_POINTS)


def _unattended(sites: Iterable[_Site]) -> list[tuple[str, str, str]]:
    return sorted(
        site.key
        for site in sites
        if site.module.startswith(_UNATTENDED) and site.module not in _RESUME_MECHANISM
    )


def _approved_by_the_mechanism(sites: Iterable[_Site]) -> list[tuple[str, str, str]]:
    return sorted(
        site.key
        for site in sites
        if site.module in _RESUME_MECHANISM and site.supplies_approve
    )


@pytest.mark.criterion("REQ-FUND-051")
def test_no_code_path_resumes_a_run_with_approve_without_a_human() -> None:
    """Validates fund.actions.apply_action, fund.agents.graph.resume_run and their
    callers in fund.cli and fund.tui: every reference that can resume a paused run
    with approve is a listed human entry point, one of apply_action's handlers or a
    link that forwards their decision, and every listed entry still exists."""
    sites = _package_sites()

    assert _unlisted(sites) == []
    assert sorted(set(_HUMAN_ENTRY_POINTS) - {site.key for site in sites}) == []
    assert all(reason.strip() for reason in _HUMAN_ENTRY_POINTS.values())


@pytest.mark.criterion("REQ-FUND-051")
def test_scheduler_worker_agents_and_tools_never_supply_an_approve() -> None:
    """Validates fund.scheduler._drive_rebalance, fund.worker, fund.agents (graphs,
    toolsets, subagents, the Ask agent) and fund.tools: none reaches the resume chain,
    and the resume mechanism in fund.agents.graph, graph_hitl and profiler only
    forwards the decision it is handed."""
    sites = _package_sites()

    assert all((_PACKAGE_ROOT / path).exists() for path in _UNATTENDED)
    assert all((_PACKAGE_ROOT / path).is_file() for path in _RESUME_MECHANISM)
    assert _unattended(sites) == []
    assert _approved_by_the_mechanism(sites) == []


_SNEAK = "_sneak"
_INJECTED_APPROVALS = [
    (
        "scheduler.py",
        "def _sneak(rid, s):\n    resume_run(rid, 'approve', session=s)\n",
        _SNEAK,
        "resume_run",
    ),
    (
        "scheduler.py",
        "def _sneak(rid, s, verb):\n    apply_action(verb, session=s, run_id=rid)\n",
        _SNEAK,
        "apply_action",
    ),
    (
        "worker.py",
        "def _sneak(run):\n    run.resume('approve', candidate_id='hrp')\n",
        _SNEAK,
        "resume",
    ),
    (
        "agents/ask_agent.py",
        "def _sneak(s, rid):\n    apply_action('approve', session=s, run_id=rid)\n",
        _SNEAK,
        "apply_action",
    ),
    (
        "scheduler.py",
        "def _sneak(req):\n    ACTION_HANDLERS[ActionKind.APPROVE](req)\n",
        _SNEAK,
        "ACTION_HANDLERS",
    ),
    (
        "scheduler.py",
        "def _sneak(req):\n    _dispatch('approve', req)\n",
        _SNEAK,
        "_dispatch",
    ),
    (
        "agents/toolsets.py",
        "def _sneak(agent):\n"
        "    agent.invoke(Command(resume={'decisions': [{'type': 'approve'}]}))\n",
        _SNEAK,
        "Command",
    ),
    (
        "agents/toolsets.py",
        "def _sneak(agent, payload):\n    agent.invoke(Command(**payload))\n",
        _SNEAK,
        "Command",
    ),
    (
        "agents/toolsets.py",
        "def _sneak(agent, payload):\n    agent.invoke(Command(None, None, payload))\n",
        _SNEAK,
        "Command",
    ),
    (
        "tools/orders.py",
        "from fund.agents.graph import resume_fund as settle\n",
        _MODULE_SCOPE,
        "resume_fund",
    ),
]


def _added_sites(module: str, injected: str) -> set[_Site]:
    """The sites ``injected`` adds when appended to a copy of ``module``'s source."""
    original = (_PACKAGE_ROOT / module).read_text(encoding="utf-8")
    return approval_sites(f"{original}\n\n{injected}", module) - approval_sites(
        original, module
    )


@pytest.mark.criterion("REQ-FUND-051")
@pytest.mark.parametrize(("module", "injected", "scope", "callee"), _INJECTED_APPROVALS)
def test_the_guard_flags_an_approve_added_to_unattended_code(
    module: str, injected: str, scope: str, callee: str
) -> None:
    """Validates fund.scheduler, fund.worker, fund.agents.ask_agent,
    fund.agents.toolsets and fund.tools.orders as the guard reads them: an approve
    added to a copy of the module is a new site with no listed reason, in code that
    must hold none."""
    added = _added_sites(module, injected)

    assert _unlisted(added) == [(module, scope, callee)]
    assert _unattended(added) == [(module, scope, callee)]


@pytest.mark.criterion("REQ-FUND-051")
def test_the_guard_flags_an_approve_the_resume_mechanism_supplies() -> None:
    """Validates fund.agents.graph as the guard reads it: a resume that passes its
    own approve instead of the caller's decision is reported."""
    added = _added_sites(
        _GRAPH,
        "def _sneak(rid, s):\n    return resume_fund(rid, 'approve', session=s)\n",
    )

    assert _approved_by_the_mechanism(added) == [(_GRAPH, _SNEAK, "resume_fund")]


@pytest.mark.criterion("REQ-FUND-051")
@pytest.mark.parametrize(
    "decision", ["'reject'", "'killswitch'", "'profile' if fresh else 'run'"]
)
def test_the_guard_ignores_a_decision_that_cannot_approve(decision: str) -> None:
    """Validates fund.scheduler as the guard reads it: a resume with a literal decision
    other than approve or edit is no approval site."""
    added = _added_sites(
        "scheduler.py",
        f"def _sneak(rid, s, fresh):\n    apply_action({decision}, session=s, "
        "run_id=rid)\n",
    )

    assert added == set()
