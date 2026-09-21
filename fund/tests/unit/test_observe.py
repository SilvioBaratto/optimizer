"""T5 — ``fund.observe`` shared model-free read model.

The single read model both frontends (CLI + TUI) use. It builds **no** LLM: every
function takes an injected ``Session`` (read-only, no ``commit``) and — where a run
transcript / pending-HITL cross-check is needed — an already-bootstrapped
``saver`` (typed ``Any``). These tests drive it on the in-memory ``db_session``
harness plus a real :class:`~langgraph.checkpoint.memory.MemorySaver` seeded
directly (no compiled graph, no model), asserting:

* ``drift_l1`` is exact on hand-checked vectors (union of tickers, empty sides);
* ``list_portfolio_runs`` is newest-first with ``awaiting_hitl`` set on paused rows;
* ``pending_hitl`` returns only paused runs whose checkpoint carries a **live**
  interrupt (the §4f cross-check) — a paused row with no interrupt is excluded;
* ``load_run_transcript`` interleaves checkpoint messages **and** ``agent_decisions``
  in order, tolerating agents that log no decision row;
* ``portfolio_state`` computes ``current`` (from ``positions``) vs ``target`` (latest
  completed run's weights, else the latest allocator proposal) with the right drift,
  and ``metrics`` stay best-effort (never fabricated);
* ``get_mandate`` rehydrates the pydantic ``PortfolioMandate`` from the JSON column;
* ``import fund.observe`` drags in no ``deepagents`` / ``langchain`` / ``langgraph``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt
from portopt_db.models import AgentRun

from fund import observe
from fund.audit.mandate_repository import MandateRepository
from fund.audit.positions_repository import PositionRepository
from fund.audit.repository import AgentRunRepository
from fund.schemas.mandate import PortfolioMandate, RunTriggers

# A letter-bearing UUID: ``AgentRun.portfolio_id`` is a Postgres ``UUID`` column,
# which SQLite gives NUMERIC affinity — an all-decimal UUID would be coerced to a
# REAL under the test harness and crash the UUID result processor. Real portfolio
# ids are ``uuid4`` (always mixed-hex), so this only bites test fixtures.
_PID = uuid.UUID("aaaa1111-2222-3333-4444-555566667777")
_ASOF = date(2026, 1, 30)
_CHANNEL_VERSION = "00000000000000000000000000000001.0.1"


# --- seeding helpers --------------------------------------------------------


def _make_run(
    session: Any,
    *,
    portfolio_id: uuid.UUID | None,
    status: str = "pending",
    weights: dict[str, float] | None = None,
    created_at: datetime,
    thread_id: str | None = None,
    finished_at: datetime | None = None,
) -> AgentRun:
    """Insert an ``agent_runs`` row with ``created_at`` set explicitly so the
    newest-first ordering is deterministic under SQLite's coarse ``func.now()``."""
    run = AgentRun(
        portfolio_id=portfolio_id,
        asof=_ASOF,
        seed=1,
        universe=["AAA"],
        optimizer_config={},
        status=status,
        weights=weights,
        thread_id=thread_id,
        created_at=created_at,
        finished_at=finished_at,
    )
    session.add(run)
    session.flush()
    return run


def _task_msg(subagent: str) -> AIMessage:
    """A PM turn delegating to ``subagent`` via the built-in ``task`` tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"description": f"Run {subagent}.", "subagent_type": subagent},
                "id": f"call_{subagent}",
            }
        ],
    )


def _seed_thread(
    saver: MemorySaver,
    thread_id: str,
    messages: list[Any],
    *,
    interrupt: dict[str, Any] | None = None,
) -> None:
    """Put a top-level checkpoint carrying ``messages`` and, optionally, a pending
    ``__interrupt__`` write — the exact shape ``observe`` reads from a paused run."""
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["messages"] = list(messages)
    checkpoint["channel_versions"]["messages"] = _CHANNEL_VERSION
    saved = saver.put(
        config,
        checkpoint,
        {"source": "loop", "step": 1},
        {"messages": _CHANNEL_VERSION},
    )
    if interrupt is not None:
        writes = [("__interrupt__", Interrupt(value=interrupt))]
        saver.put_writes(saved, writes, "task-1")


def _mandate(portfolio_id: str) -> PortfolioMandate:
    return PortfolioMandate(
        portfolio_id=portfolio_id,
        capital=Decimal("250000"),
        base_currency="EUR",
        drift_l1_threshold=0.15,
        triggers=RunTriggers(cron=True, drift=False),
        benchmark="^STOXX50E",
    )


# --- drift_l1 (pure util) ---------------------------------------------------


def test_drift_l1_matches_hand_computed_value():
    current = {"AAA": 0.5, "BBB": 0.5}
    target = {"AAA": 0.6, "BBB": 0.4}
    # |0.5-0.6| + |0.5-0.4| = 0.2
    assert observe.drift_l1(current, target) == pytest.approx(0.2)


def test_drift_l1_over_disjoint_tickers_sums_the_union():
    # A ticker missing on one side counts as weight 0 there.
    assert observe.drift_l1({"AAA": 1.0}, {"BBB": 1.0}) == 2.0


def test_drift_l1_handles_empty_sides():
    assert observe.drift_l1({}, {}) == 0.0
    assert observe.drift_l1({}, {"AAA": 1.0}) == 1.0
    assert observe.drift_l1({"AAA": 0.3}, {}) == 0.3


# --- list_portfolio_runs ----------------------------------------------------


def test_list_portfolio_runs_newest_first_with_awaiting_hitl(db_session):
    older = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
        finished_at=datetime(2026, 1, 10, 1, tzinfo=UTC),
    )
    newer = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )

    summaries = observe.list_portfolio_runs(db_session, _PID)

    assert [s.run_id for s in summaries] == [newer.id, older.id]
    paused, completed = summaries
    assert paused.awaiting_hitl is True
    assert paused.status == "paused"
    assert paused.n_weights == 0
    assert completed.awaiting_hitl is False
    assert completed.n_weights == 1
    assert completed.finished_at == older.finished_at


def test_list_portfolio_runs_empty_for_unknown_portfolio(db_session):
    assert observe.list_portfolio_runs(db_session, uuid.uuid4()) == []


# --- pending_hitl (paused + live-interrupt cross-check, §4f) -----------------


def test_pending_hitl_returns_only_paused_runs_with_a_live_interrupt(db_session):
    saver = MemorySaver()

    live = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-live",
    )
    # Paused in the DB but its checkpoint carries NO pending interrupt.
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
        thread_id="thread-stale",
    )
    # A completed run must never surface as pending HITL.
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
        thread_id="thread-done",
    )

    _seed_thread(
        saver,
        "thread-live",
        [HumanMessage(content="go")],
        interrupt={"action_requests": [{"action": "place_orders"}]},
    )
    _seed_thread(saver, "thread-stale", [HumanMessage(content="done")])

    pending = observe.pending_hitl(db_session, saver, _PID)

    assert [p.run_id for p in pending] == [live.id]
    assert pending[0].awaiting_hitl is True


def test_pending_hitl_excludes_run_whose_thread_is_absent_from_saver(db_session):
    saver = MemorySaver()
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="never-seeded",
    )
    assert observe.pending_hitl(db_session, saver, _PID) == []


# --- load_run_transcript (messages + decisions interleaved) -----------------


def test_load_run_transcript_interleaves_messages_and_decisions_in_order(db_session):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-1",
    )
    audit = AgentRunRepository(db_session)
    # Allocator logs the load-bearing weights; economist/risk log NO decision row.
    audit.append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"AAA": 0.6, "BBB": 0.4}}),
    )
    audit.append_decision(
        run.id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "pending"},
    )
    db_session.expire(run, ["decisions"])

    _seed_thread(
        saver,
        "thread-1",
        [
            HumanMessage(content="Produce an allocation."),
            _task_msg("economist"),
            _task_msg("allocator"),
            _task_msg("risk"),
            AIMessage(
                content="",
                tool_calls=[{"name": "place_orders", "args": {}, "id": "call_orders"}],
            ),
        ],
    )

    entries = observe.load_run_transcript(db_session, saver, run)

    # The allocator decision anchors right after the ``task(allocator)`` delegation;
    # the orchestrator decision right after the ``place_orders`` tool call. The
    # economist/risk turns appear with no decision row (tolerated).
    shape = [(e.source, e.agent, e.step) for e in entries]
    assert shape == [
        ("message", "human", None),
        ("message", "ai", None),  # economist delegation
        ("message", "ai", None),  # allocator delegation
        ("decision", "allocator", "optimize_portfolio"),
        ("message", "ai", None),  # risk delegation
        ("message", "ai", None),  # place_orders call
        ("decision", "orchestrator", "place_orders"),
    ]
    alloc = entries[3]
    assert alloc.payload == {"weights": {"AAA": 0.6, "BBB": 0.4}}
    order = entries[6]
    assert order.payload == {"hitl_decision": {"decision": "pending"}}


def test_load_run_transcript_trails_unanchored_decisions(db_session):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-2",
    )
    AgentRunRepository(db_session).append_decision(
        run.id, agent="allocator", step="optimize_portfolio"
    )
    db_session.expire(run, ["decisions"])
    # No message references the allocator/optimize_portfolio → the decision trails.
    _seed_thread(saver, "thread-2", [HumanMessage(content="lone narrative")])

    entries = observe.load_run_transcript(db_session, saver, run)

    assert [(e.source, e.agent) for e in entries] == [
        ("message", "human"),
        ("decision", "allocator"),
    ]


def test_load_run_transcript_empty_when_thread_absent(db_session):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="pending",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="ghost",
    )
    assert observe.load_run_transcript(db_session, saver, run) == []


# --- portfolio_state --------------------------------------------------------


def test_portfolio_state_current_target_and_drift(db_session):
    PositionRepository(db_session).set_holdings(
        _PID,
        [{"ticker": "AAA", "weight": 0.5}, {"ticker": "BBB", "weight": 0.5}],
        asof=_ASOF,
    )
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 0.6, "BBB": 0.4},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.current == {"AAA": 0.5, "BBB": 0.5}
    assert state.target == {"AAA": 0.6, "BBB": 0.4}
    assert state.drift_l1 == pytest.approx(0.2)
    assert state.metrics == {}  # best-effort — none persisted, so empty (not faked)


def test_portfolio_state_target_falls_back_to_latest_allocator_proposal(db_session):
    PositionRepository(db_session).set_holdings(
        _PID, [{"ticker": "AAA", "weight": 1.0}], asof=_ASOF
    )
    # No completed run — only a paused run with an allocator proposal in the trail.
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    AgentRunRepository(db_session).append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"AAA": 0.7, "CCC": 0.3}}),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"AAA": 0.7, "CCC": 0.3}
    # |1-0.7| + |0-0.3| = 0.6
    assert state.drift_l1 == pytest.approx(0.6)


def test_portfolio_state_reads_best_effort_metrics_when_present(db_session):
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    AgentRunRepository(db_session).append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps(
            {"weights": {"AAA": 1.0}, "metrics": {"sharpe": 1.2, "vol": 0.18}}
        ),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.metrics == {"sharpe": 1.2, "vol": 0.18}


def test_portfolio_state_empty_portfolio(db_session):
    state = observe.portfolio_state(db_session, uuid.uuid4())
    assert state.current == {}
    assert state.target == {}
    assert state.drift_l1 == 0.0
    assert state.metrics == {}


# --- get_mandate (JSON → pydantic rehydration) ------------------------------


def test_get_mandate_rehydrates_pydantic_from_json(db_session):
    pid = uuid.uuid4()
    mandate = _mandate(str(pid))
    MandateRepository(db_session).upsert(mandate)

    fetched = observe.get_mandate(db_session, pid)

    assert isinstance(fetched, PortfolioMandate)
    assert fetched == mandate


def test_get_mandate_none_for_unknown_portfolio(db_session):
    assert observe.get_mandate(db_session, uuid.uuid4()) is None


# --- edge branches / Checkpoint-B review hardening --------------------------


def test_pending_hitl_ignores_paused_run_with_null_thread_id(db_session):
    # A legacy paused row predating per-run threading (thread_id NULL) can't be
    # cross-checked against the saver, so it is never offered as Approve-able.
    saver = MemorySaver()
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id=None,
    )
    assert observe.pending_hitl(db_session, saver, _PID) == []


def test_load_run_transcript_empty_when_thread_id_is_none(db_session):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="pending",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id=None,
    )
    assert observe.load_run_transcript(db_session, saver, run) == []


def test_load_run_transcript_uses_message_name_and_flattens_block_content(db_session):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-blocks",
    )
    db_session.expire(run, ["decisions"])
    _seed_thread(
        saver,
        "thread-blocks",
        [
            AIMessage(
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ],
                name="economist",
            )
        ],
    )

    (entry,) = observe.load_run_transcript(db_session, saver, run)

    # A named subagent message reports its name; list-block content flattens to text.
    assert entry.agent == "economist"
    assert entry.text == "hello world"


def test_message_tokens_tolerates_malformed_tool_calls():
    # _message_tokens guards against non-dict entries, name-less calls, and non-dict
    # args — a malformed checkpoint must never crash the transcript merge.
    message = SimpleNamespace(
        tool_calls=[
            "not-a-dict",  # skipped
            {"args": {"subagent_type": "risk"}},  # no name; subagent captured
            {"name": "place_orders", "args": "not-a-dict"},  # name only; args ignored
            {"name": "task", "args": {}},  # name only; no subagent_type
        ]
    )

    assert observe._message_tokens(message) == {"risk", "place_orders", "task"}


def test_load_run_transcript_decision_payload_carries_constraint_and_views(db_session):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-cs",
    )
    AgentRunRepository(db_session).append_decision(
        run.id,
        agent="economist",
        step="analyze",
        constraint_set={"max_weight": 0.1},
        views={"AAA": "bullish"},
    )
    db_session.expire(run, ["decisions"])
    # No message anchors "economist"/"analyze", so the decision trails at the end.
    _seed_thread(saver, "thread-cs", [HumanMessage(content="narrative")])

    entries = observe.load_run_transcript(db_session, saver, run)

    assert entries[-1].payload == {
        "constraint_set": {"max_weight": 0.1},
        "views": {"AAA": "bullish"},
    }


def test_load_run_transcript_decision_payload_none_for_unparseable_response(db_session):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-bad",
    )
    AgentRunRepository(db_session).append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response="not valid json{",
    )
    db_session.expire(run, ["decisions"])
    _seed_thread(saver, "thread-bad", [HumanMessage(content="narrative")])

    entries = observe.load_run_transcript(db_session, saver, run)

    # Unparseable llm_response yields no weights and no other structured field.
    assert entries[-1].payload is None


def test_portfolio_state_target_fallback_skips_run_without_a_proposal(db_session):
    # Newest run has no allocator proposal; the fallback must skip it and read the
    # older run's proposal rather than returning {} on the first empty candidate.
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
    )
    older = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    AgentRunRepository(db_session).append_decision(
        older.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"XYZ": 1.0}}),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"XYZ": 1.0}


def test_metrics_uses_newest_allocator_decision_within_a_run(db_session):
    # Regression for the confirmed review finding: a rebuild-to-resume appends a
    # second allocator optimize_portfolio decision to the SAME run. The metrics must
    # come from the NEWEST decision (aligned with the target-weights source), not the
    # stale first one — and non-allocator decisions in between are skipped.
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    audit = AgentRunRepository(db_session)
    audit.append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"AAA": 0.5}, "metrics": {"sharpe": 0.8}}),
    )
    audit.append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"AAA": 1.0}, "metrics": {"sharpe": 1.5}}),
    )
    audit.append_decision(
        run.id,
        agent="orchestrator",
        step="place_orders",
        hitl_decision={"decision": "approve"},
    )

    state = observe.portfolio_state(db_session, _PID)

    # Newest allocator proposal's metrics (1.5), not the first proposal's stale 0.8.
    assert state.metrics == {"sharpe": 1.5}


def test_message_text_stringifies_non_str_non_list_content():
    # Defensive fallback for a message whose content is neither str nor block-list.
    assert observe._message_text(SimpleNamespace(content=None)) == "None"


def test_metrics_from_response_empty_for_unparseable_json():
    # Best-effort: an unparseable allocator payload yields no metrics (never faked).
    assert observe._metrics_from_response("not valid json{") == {}


# --- agent-stack-free import invariant --------------------------------------


def test_import_fund_observe_is_agent_stack_free():
    code = textwrap.dedent(
        """
        import sys
        from fund import observe
        for name in observe.__all__:
            getattr(observe, name)
        forbidden = {
            "deepagents",
            "langchain",
            "langchain_core",
            "langchain_ollama",
            "langgraph",
            "app",
        }
        leaked = sorted({m.split(".")[0] for m in sys.modules} & forbidden)
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
