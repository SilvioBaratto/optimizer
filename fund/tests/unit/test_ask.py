"""``fund.tui.ask`` — the model-free classify / target-resolve / retrieve helpers.

Pure functions over an injected session + ``MemorySaver`` (no Textual, no model, no
agent stack): a turn is classified ``query`` | ``action`` by its leading decision verb;
an action turn resolves its target against the pending-HITL gates / autonomy scope /
focus, never on a guess; ``build_context`` assembles a compact grounded snapshot. A
subprocess guard pins the agent-stack-light (and optimizer/skfolio-free) import.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, date, datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt
from portopt_db.models import AgentRun

from fund.tui import ask

_PID_A = uuid.UUID("aaaa0001-0000-4000-8000-000000000001")
_PID_B = uuid.UUID("bbbb0002-0000-4000-8000-000000000002")
_ASOF = date(2026, 10, 3)
_CHANNEL_VERSION = 1


def _make_paused_run(
    session: Any, *, portfolio_id: uuid.UUID, thread_id: str
) -> AgentRun:
    """Insert a paused ``agent_runs`` row carrying ``thread_id`` (a gate candidate)."""
    run = AgentRun(
        portfolio_id=portfolio_id,
        asof=_ASOF,
        seed=1,
        universe=["AAA"],
        optimizer_config={},
        status="paused",
        thread_id=thread_id,
        created_at=datetime(2026, 10, 3, tzinfo=UTC),
    )
    session.add(run)
    session.flush()
    return run


def _seed_interrupt(saver: MemorySaver, thread_id: str) -> None:
    """Put a top-level checkpoint with a live ``__interrupt__`` write for the thread."""
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["messages"] = [HumanMessage(content="go")]
    checkpoint["channel_versions"]["messages"] = _CHANNEL_VERSION
    saved = saver.put(
        config,
        checkpoint,
        {"source": "loop", "step": 1},
        {"messages": _CHANNEL_VERSION},
    )
    saver.put_writes(
        saved,
        [
            (
                "__interrupt__",
                Interrupt(value={"action_requests": [{"name": "place_orders"}]}),
            )
        ],
        "task-1",
    )


def _seed_gate(db_session, saver, *, portfolio_id, thread_id):
    """Seed one pending-HITL gate (paused run + live interrupt); return its run id."""
    run = _make_paused_run(db_session, portfolio_id=portfolio_id, thread_id=thread_id)
    _seed_interrupt(saver, thread_id)
    return run.id


# --- classify ---------------------------------------------------------------


def test_classify_tags_decision_verbs_as_action():
    """Every decision verb (incl. synonyms + hyphenated re-arm) classifies as action."""
    for turn in (
        "approve the gate",
        "reject it",
        "killswitch the run",
        "freeze the fund",
        "re-arm the fund",
        "run a rebalance",
        "steer the PM toward defence",
    ):
        assert ask.classify(turn) == "action", turn


def test_classify_tags_questions_as_query():
    """A question — even one mentioning a verb mid-sentence — classifies as query."""
    for turn in (
        "why did the allocator trim energy?",
        "what's my worst-drawdown portfolio?",
        "how did we do this week?",
        "",
        "   ",
    ):
        assert ask.classify(turn) == "query", turn


# --- resolve_target: resume verbs (gates) -----------------------------------


def test_resolve_target_single_gate_is_unambiguous(db_session):
    """One pending gate resolves a resume verb with no disambiguation needed."""
    saver = MemorySaver()
    run_id = _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")

    res = ask.resolve_target("approve the gate", db_session, saver, None)

    assert res.status == "resolved"
    assert res.action is not None
    assert res.action.verb == "approve"
    assert res.action.run_id == run_id


def test_resolve_target_two_gates_is_ambiguous(db_session):
    """Two pending gates with no name make a resume verb ambiguous (never guessed)."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")

    res = ask.resolve_target("approve the gate", db_session, saver, None)

    assert res.status == "ambiguous"
    assert res.action is None
    assert len(res.candidates) == 2
    assert res.prompt  # a non-empty follow-up question


def test_resolve_target_named_gate_resolves_among_many(db_session):
    """Naming a gate by its short id resolves it even with other gates pending."""
    saver = MemorySaver()
    run_a = _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")
    short = str(run_a).split("-", 1)[0]

    res = ask.resolve_target(f"approve gate #{short}", db_session, saver, None)

    assert res.status == "resolved"
    assert res.action is not None and res.action.run_id == run_a


def test_resolve_target_no_gate_is_unresolved(db_session):
    """A resume verb with no pending gate is a dead end, not an ambiguity."""
    saver = MemorySaver()

    res = ask.resolve_target("approve the gate", db_session, saver, None)

    assert res.status == "unresolved"
    assert res.action is None


# --- resolve_target: scope + rebalance verbs --------------------------------


def test_resolve_target_freeze_fund_scope(db_session):
    """'freeze the fund' resolves to fund scope with no portfolio."""
    res = ask.resolve_target("freeze the fund", db_session, MemorySaver(), None)

    assert res.status == "resolved"
    assert res.action is not None
    assert res.action.verb == "freeze" and res.action.scope == "fund"
    assert res.action.portfolio_id is None


def test_resolve_target_freeze_portfolio_scope_with_focus(db_session):
    """'freeze it' with a focused portfolio resolves to that portfolio's scope."""
    res = ask.resolve_target("freeze it", db_session, MemorySaver(), _PID_A)

    assert res.status == "resolved"
    assert res.action is not None
    assert res.action.scope == "portfolio" and res.action.portfolio_id == _PID_A


def test_resolve_target_freeze_without_focus_is_ambiguous(db_session):
    """'freeze it' with no scope word and no focus asks which, never guesses."""
    res = ask.resolve_target("freeze it", db_session, MemorySaver(), None)

    assert res.status == "ambiguous"
    assert res.action is None


def test_resolve_target_rearm_fund_scope(db_session):
    """'re-arm the fund' resolves to the fund-scope rearm verb."""
    res = ask.resolve_target("re-arm the fund", db_session, MemorySaver(), None)

    assert res.status == "resolved"
    assert res.action is not None
    assert res.action.verb == "rearm" and res.action.scope == "fund"


def test_resolve_target_run_rebalance_needs_focus(db_session):
    """'run a rebalance' is ambiguous fund-wide but resolves to the focus."""
    saver = MemorySaver()
    assert (
        ask.resolve_target("run a rebalance", db_session, saver, None).status
        == "ambiguous"
    )
    res = ask.resolve_target("run a rebalance", db_session, saver, _PID_A)
    assert res.status == "resolved"
    assert res.action is not None and res.action.verb == "run"
    assert res.action.portfolio_id == _PID_A


# --- build_context ----------------------------------------------------------


def test_build_context_is_nonempty_and_grounded(db_session):
    """A focused build pulls the fund roster, the pending gate, and the focus block."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")

    context = ask.build_context(db_session, saver, _PID_A, "how are we doing?")

    assert context  # never empty
    assert "FUND STATE" in context
    assert "pending gates: 1" in context
    assert "FOCUS portfolio" in context


def test_build_context_fund_wide_without_focus(db_session):
    """A fund-wide build omits the focus block but still grounds the fund state."""
    context = ask.build_context(db_session, MemorySaver(), None, "any gates?")

    assert "FUND STATE" in context
    assert "FOCUS portfolio" not in context  # nothing focused → no focus block


# --- import hygiene: model-free + agent-stack-free --------------------------


def test_import_fund_tui_ask_is_light():
    """Importing the module pulls no agent stack, model, Textual, or optimizer."""
    code = textwrap.dedent(
        """
        import sys
        from fund.tui import ask
        for name in ask.__all__:
            getattr(ask, name)
        forbidden = {
            "deepagents",
            "langchain",
            "langchain_core",
            "langchain_ollama",
            "langgraph",
            "optimizer",
            "skfolio",
            "textual",
            "app",
        }
        leaked = sorted({m.split(".")[0] for m in sys.modules} & forbidden)
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
