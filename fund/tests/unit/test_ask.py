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


def test_resolve_target_stale_gate_id_is_not_retargeted_to_the_sole_gate(db_session):
    """A typed id that names no live gate is unresolved, never swapped for the only
    live gate (the operator may have meant a gate that is already decided)."""
    saver = MemorySaver()
    live = _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")

    for turn in (
        "approve gate #aaaa1111",
        "approve 1a2b3c4d",
        "reject #deadbeef",
        "approve aaaa1111-0000-4000-8000-000000000001",
    ):
        res = ask.resolve_target(turn, db_session, saver, None)
        assert res.status == "unresolved", turn
        assert res.action is None, turn

    short = str(live).split("-", 1)[0]
    named = ask.resolve_target(f"approve gate #{short}", db_session, saver, None)
    assert named.status == "resolved"
    assert named.action is not None and named.action.run_id == live


def test_resolve_target_stale_gate_id_with_many_gates_is_unresolved(db_session):
    """With several live gates a stale id is still a dead end, not a gate pick-list."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")

    res = ask.resolve_target("approve #deadbeef", db_session, saver, None)

    assert res.status == "unresolved"
    assert res.action is None


def test_resolve_target_live_and_stale_ids_together_are_unresolved(db_session):
    """Naming a live gate alongside an id that matches nothing is contradictory."""
    saver = MemorySaver()
    live = _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")
    short = str(live).split("-", 1)[0]

    res = ask.resolve_target(f"approve #{short} and #deadbeef", db_session, saver, None)

    assert res.status == "unresolved"
    assert res.action is None


def test_resolve_target_gate_named_by_portfolio_short_id_resolves(db_session):
    """A gate may also be named by its portfolio's short id."""
    saver = MemorySaver()
    run_a = _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")

    res = ask.resolve_target("approve aaaa0001", db_session, saver, None)

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


def test_resolve_target_freeze_substring_fund_stays_on_focus(db_session):
    """A focused freeze whose turn only *contains* 'fund' (e.g. 'fundamentals') stays on
    the focused portfolio — fund scope needs the standalone word, never a substring."""
    res = ask.resolve_target(
        "freeze it, the fundamentals look weak", db_session, MemorySaver(), _PID_A
    )

    assert res.status == "resolved"
    assert res.action is not None
    assert res.action.scope == "portfolio" and res.action.portfolio_id == _PID_A


def test_resolve_target_scope_and_run_verbs_carry_no_gate_run_id(db_session):
    """Only resume verbs name a gate: a focused freeze / rearm / run resolves with
    ``run_id=None`` rather than smuggling the portfolio id into the gate slot."""
    for turn in ("freeze it", "re-arm it", "run a rebalance"):
        res = ask.resolve_target(turn, db_session, MemorySaver(), _PID_A)

        assert res.status == "resolved", turn
        assert res.action is not None
        assert res.action.run_id is None, turn
        assert res.action.portfolio_id == _PID_A, turn


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


def test_resolve_target_rebalance_naming_the_fund_is_not_narrowed_to_the_focus(
    db_session,
):
    """A rebalance is per portfolio: a turn that names the whole fund asks which one,
    rather than quietly rebalancing the focused portfolio it never mentioned."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")

    for focus in (None, _PID_A):
        for turn in (
            "rebalance the fund",
            "run a rebalance for the whole fund",
            "rebalance the fund and bbbb0002",
        ):
            res = ask.resolve_target(turn, db_session, saver, focus)

            assert res.status == "ambiguous", (turn, focus)
            assert res.action is None, (turn, focus)


def test_resolve_target_scope_and_run_actions_carry_no_run_id(db_session):
    """run_id is the paused run a resume verb targets; freeze / re-arm / run name a
    portfolio, so they must not smuggle it in as a (non-existent) gate."""
    for turn in ("freeze it", "rearm it", "run a rebalance"):
        res = ask.resolve_target(turn, db_session, MemorySaver(), _PID_A)

        assert res.status == "resolved", turn
        assert res.action is not None
        assert res.action.run_id is None, turn
        assert res.action.portfolio_id == _PID_A, turn


def test_resolve_target_scope_follows_the_portfolio_named_in_the_turn(db_session):
    """A portfolio short id typed in the turn beats the focused portfolio."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")

    frozen = ask.resolve_target("freeze portfolio bbbb0002", db_session, saver, _PID_A)
    rebalanced = ask.resolve_target(
        "run a rebalance for #bbbb0002", db_session, saver, _PID_A
    )

    assert frozen.action is not None
    assert frozen.action.scope == "portfolio" and frozen.action.portfolio_id == _PID_B
    assert rebalanced.action is not None
    assert rebalanced.action.verb == "run" and rebalanced.action.portfolio_id == _PID_B


def test_resolve_target_scope_unknown_portfolio_id_is_unresolved(db_session):
    """An id that names no known portfolio is a dead end, never the focused one."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")

    for turn in ("freeze portfolio 71443ed2", "run a rebalance for #deadbeef"):
        res = ask.resolve_target(turn, db_session, saver, _PID_A)

        assert res.status == "unresolved", turn
        assert res.action is None, turn


def test_resolve_target_blanket_words_are_not_narrowed_to_the_focus(db_session):
    """'freeze everything' must not quietly freeze only the focused portfolio."""
    for turn in (
        "freeze everything",
        "freeze all portfolios",
        "rearm every portfolio",
        "run a rebalance for all portfolios",
    ):
        res = ask.resolve_target(turn, db_session, MemorySaver(), _PID_A)

        assert res.status == "ambiguous", turn
        assert res.action is None, turn


def test_resolve_target_blanket_word_with_the_fund_is_fund_scope(db_session):
    """A blanket word that also names the fund is unambiguous: the whole fund."""
    res = ask.resolve_target(
        "freeze everything in the fund", db_session, MemorySaver(), _PID_A
    )

    assert res.status == "resolved"
    assert res.action is not None and res.action.scope == "fund"


def test_resolve_target_negated_turn_never_resolves_on_the_negated_target(db_session):
    """'not the fund' names the fund only to exclude it; a negation asks instead."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")

    for turn in (
        "freeze portfolio bbbb0002, not the fund",
        "freeze it but don't touch the fund",
        "freeze it, not bbbb0002",
    ):
        res = ask.resolve_target(turn, db_session, saver, _PID_A)

        assert res.status == "ambiguous", turn
        assert res.action is None, turn


def test_resolve_target_negated_gate_turn_never_resolves_on_the_excluded_gate(
    db_session,
):
    """A resume verb is the irreversible path: 'approve not #X' must ask, with or
    without a focus, rather than approve the very gate it excludes."""
    saver, run_a, run_b = _two_gates(db_session)
    short_a, short_b = (str(run).split("-", 1)[0] for run in (run_a, run_b))

    for focus in (None, _PID_A, _PID_B):
        for turn in (
            f"approve not #{short_b}",
            f"approve #{short_a} not #{short_b}",
            f"approve the other one, not #{short_b}",
            f"reject but don't touch #{short_b}",
            f"killswitch everything except #{short_b}",
            f"steer anything but #{short_b}, not the other",
        ):
            res = ask.resolve_target(turn, db_session, saver, focus)

            assert res.status == "ambiguous", (turn, focus)
            assert res.action is None, (turn, focus)
            assert {gate.run_id for gate in res.candidates} == {run_a, run_b}


def test_resolve_target_negated_gate_turn_can_be_answered_by_naming_a_gate(
    db_session,
):
    """The negation follow-up carries every live gate, so a bare '#id' completes it."""
    saver, run_a, run_b = _two_gates(db_session)
    short_a, short_b = (str(run).split("-", 1)[0] for run in (run_a, run_b))
    turn = f"approve the other one, not #{short_b}"

    res = ask.resolve_target(turn, db_session, saver, None)
    pending = ask.pending_action(turn, res)
    answered = ask.resolve_pending(pending, f"#{short_a}", db_session, saver, None)

    assert pending.verb == "approve"
    assert answered.status == "resolved"
    assert answered.action is not None and answered.action.run_id == run_a


def test_negated_follow_up_after_a_pending_approve_is_not_an_answer_or_a_target(
    db_session,
):
    """'approve the other one, not #X' after 'approve which?' neither answers the
    question nor resolves, as a fresh turn, onto the excluded gate #X."""
    saver, run_a, run_b = _two_gates(db_session)
    short_b = str(run_b).split("-", 1)[0]
    pending = ask.PendingAction("approve", (run_a, run_b))
    turn = f"approve the other one, not #{short_b}"

    answered = ask.resolve_pending(pending, turn, db_session, saver, None)
    fresh = ask.resolve_target(turn, db_session, saver, None)

    assert answered.status == "unresolved" and answered.action is None
    assert fresh.status == "ambiguous" and fresh.action is None


def test_resolve_target_negation_with_no_gate_paused_is_still_a_dead_end(db_session):
    """Nothing to decide stays 'unresolved' — the negation does not mask that."""
    res = ask.resolve_target("approve not #deadbeef", db_session, MemorySaver(), None)

    assert res.status == "unresolved"
    assert res.action is None


# --- resolve_pending: a follow-up must answer the question that was asked ---


def _two_gates(db_session) -> tuple[MemorySaver, uuid.UUID, uuid.UUID]:
    """Seed one gate per portfolio; return the saver and both run ids."""
    saver = MemorySaver()
    run_a = _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")
    run_b = _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")
    return saver, run_a, run_b


def test_resolve_pending_gate_answer_names_the_gate_under_the_pending_verb(db_session):
    """'#id', 'gate #id' and the pending verb + id all answer 'which gate?'."""
    saver, _run_a, run_b = _two_gates(db_session)
    pending = ask.PendingAction("approve", (_run_a, run_b))
    short = str(run_b).split("-", 1)[0]

    for turn in (f"#{short}", f"gate #{short}", f"approve #{short}"):
        res = ask.resolve_pending(pending, turn, db_session, saver, None)

        assert res.status == "resolved", turn
        assert res.action is not None, turn
        assert res.action.verb == "approve" and res.action.run_id == run_b, turn


def test_resolve_pending_follow_up_with_its_own_verb_is_not_an_answer(db_session):
    """'reject #id' after a pending approve is a new request, never an approve."""
    saver, run_a, run_b = _two_gates(db_session)
    pending = ask.PendingAction("approve", (run_a, run_b))
    short = str(run_b).split("-", 1)[0]

    for turn in (
        f"reject #{short}",
        f"killswitch #{short}",
        f"do not approve #{short}",
        f"not #{short}, the other one",
        f"what is the status of #{short}?",
    ):
        res = ask.resolve_pending(pending, turn, db_session, saver, None)

        assert res.status == "unresolved", turn
        assert res.action is None, turn


def test_resolve_pending_scope_needs_an_explicit_target(db_session):
    """A question or another verb mentioning 'fund' does not answer 'freeze what?'."""
    pending = ask.PendingAction("freeze", ())

    for focus in (None, _PID_A):
        for turn in (
            "how is the fund doing?",
            "what is the fund drawdown?",
            "rearm the fund",
            "what is the drift?",
            "freeze it",
        ):
            res = ask.resolve_pending(pending, turn, db_session, MemorySaver(), focus)

            assert res.status == "unresolved", (turn, focus)
            assert res.action is None, (turn, focus)


def test_resolve_pending_scope_answered_by_the_fund_or_a_portfolio_id(db_session):
    """'the fund' / 'freeze the fund' / a portfolio id answer the scope question."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")
    pending = ask.PendingAction("freeze", ())

    for turn in ("the fund", "freeze the fund"):
        res = ask.resolve_pending(pending, turn, db_session, saver, None)

        assert res.action is not None and res.action.scope == "fund", turn

    res = ask.resolve_pending(pending, "bbbb0002", db_session, saver, None)
    assert res.action is not None and res.action.portfolio_id == _PID_B


def test_resolve_pending_run_needs_an_explicit_portfolio_id(db_session):
    """The focus alone never answers 'run which portfolio?'; a typed id does."""
    saver = MemorySaver()
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")
    pending = ask.PendingAction("run", ())

    for turn in ("what is the drift?", "run a rebalance"):
        res = ask.resolve_pending(pending, turn, db_session, saver, _PID_A)

        assert res.status == "unresolved", turn
        assert res.action is None, turn

    named = ask.resolve_pending(pending, "bbbb0002", db_session, saver, _PID_A)
    assert named.action is not None
    assert named.action.verb == "run" and named.action.portfolio_id == _PID_B


# --- resolve_pending --------------------------------------------------------


def _pending_approve(db_session, saver):
    """Two gates + the pending 'approve' an unnamed approve turn carries forward."""
    run_a = _seed_gate(db_session, saver, portfolio_id=_PID_A, thread_id="t-a")
    _seed_gate(db_session, saver, portfolio_id=_PID_B, thread_id="t-b")
    ambiguous = ask.resolve_target("approve the gate", db_session, saver, None)
    pending = ask.pending_action("approve the gate", ambiguous)
    return run_a, str(run_a).split("-", 1)[0], pending


def test_resolve_pending_follow_up_naming_the_gate_keeps_the_pending_verb(db_session):
    """A bare short id answers the question, so the carried verb still applies."""
    saver = MemorySaver()
    run_a, short, pending = _pending_approve(db_session, saver)

    for follow_up in (f"#{short}", f"approve #{short}"):
        res = ask.resolve_pending(pending, follow_up, db_session, saver, None)

        assert res.status == "resolved", follow_up
        assert res.action is not None
        assert res.action.verb == "approve" and res.action.run_id == run_a


def test_resolve_pending_follow_up_with_a_different_verb_is_not_resolved(db_session):
    """'reject #id' answering an 'approve' question is a new command, not an approve."""
    saver = MemorySaver()
    _, short, pending = _pending_approve(db_session, saver)

    res = ask.resolve_pending(pending, f"reject #{short}", db_session, saver, None)

    assert res.status != "resolved"
    assert res.action is None


def test_resolve_pending_scope_follow_up_with_a_different_verb_is_not_resolved(
    db_session,
):
    """A pending 'freeze' is not answered by 're-arm the fund' (it would freeze it)."""
    pending = ask.pending_action(
        "freeze it", ask.resolve_target("freeze it", db_session, MemorySaver(), None)
    )

    res = ask.resolve_pending(
        pending, "re-arm the fund", db_session, MemorySaver(), None
    )

    assert res.status != "resolved"
    assert res.action is None


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
