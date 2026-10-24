"""``fund.observe`` shared model-free read model.

The single read model both frontends (CLI + TUI) use. It builds **no** LLM: every
function takes an injected ``Session`` (read-only, no ``commit``) and — where a run
transcript / pending-HITL cross-check is needed — an already-bootstrapped
``saver`` (typed ``Any``). These tests drive it on the in-memory ``db_session``
harness plus a real :class:`~langgraph.checkpoint.memory.MemorySaver` seeded
directly (no compiled graph, no model), asserting:

* ``drift_l1`` is exact on hand-checked vectors (union of tickers, empty sides);
* ``list_portfolio_runs`` is newest-first with ``awaiting_hitl`` set on paused rows;
* ``pending_hitl`` returns only paused runs whose checkpoint carries a **live**
  interrupt — a paused row with no interrupt is excluded;
* ``load_run_transcript`` interleaves checkpoint messages **and** ``agent_decisions``
  in order, tolerating agents that log no decision row;
* ``portfolio_state`` computes ``current`` (from ``positions``) vs ``target`` (latest
  completed run's weights, else the latest allocator proposal) with the right drift,
  and ``metrics`` stay best-effort (never fabricated);
* ``get_mandate`` rehydrates the pydantic ``PortfolioMandate`` from the JSON column;
* ``default_decision_bar`` picks the day before the bar the universe has reached, so
  the next close ``place_orders`` fills at is already stored;
* ``import fund.observe`` drags in no ``deepagents`` / ``langchain`` / ``langgraph``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt
from portopt_db.models import AgentRun, Exchange, Instrument, PriceHistory
from portopt_db.repositories.market_data.market_journal_repository import (
    MarketJournalRepository,
)
from portopt_db.repositories.orders.portfolio_journal_repository import (
    PortfolioJournalRepository,
)
from sqlalchemy import event

from fund import observe
from fund.audit.mandate_repository import MandateRepository
from fund.audit.mifid_repository import MifidProfileRepository
from fund.audit.nav_history_repository import NavHistoryRepository
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

# Distinct letter-bearing portfolio ids for the list_portfolios union (same
# SQLite UUID-affinity caveat as _PID — all-decimal ids coerce to REAL).
_PID_RUN = uuid.UUID("aaaa0010-0000-4000-8000-000000000010")
_PID_MANDATE = uuid.UUID("bbbb0020-0000-4000-8000-000000000020")
_PID_MIFID = uuid.UUID("cccc0030-0000-4000-8000-000000000030")


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


class _DelegatingModel(BaseChatModel):
    """Scripted chat model: the PM delegates once to ``economist``, then finishes.

    The economist subagent (recognised by its system prompt) answers with
    ``narrative``. No network, no tools beyond deepagents' built-in ``task``.
    """

    narrative: str = "economist read"
    subagent: str = "economist"

    @property
    def _llm_type(self) -> str:
        return "scripted-delegating"

    def bind_tools(self, tools: Any, **_: Any) -> _DelegatingModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = next(
            (str(m.content) for m in messages if isinstance(m, SystemMessage)), ""
        )
        if "SUBAGENT-PROMPT" in system:
            reply = AIMessage(content=self.narrative)
        elif any(isinstance(m, ToolMessage) for m in messages):
            reply = AIMessage(content="allocation done")
        else:
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Run the analysis.",
                            "subagent_type": self.subagent,
                        },
                        "id": "call_delegate",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _run_real_deep_agent(
    saver: MemorySaver,
    thread_id: str,
    *,
    narrative: str = "economist read",
    subagent: str = "economist",
) -> None:
    """Drive a real ``create_deep_agent`` graph so the checkpoint has production shape.

    deepagents keeps ``messages`` in a ``DeltaChannel``: the saver stores only
    per-step writes (no ``channel_values['messages']``), and a subagent's report
    reaches the PM thread as ``ToolMessage(name='task')``. Hand-built checkpoints
    cannot reproduce either, so the read-model tests that matter run against this.
    """
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=_DelegatingModel(narrative=narrative, subagent=subagent),
        subagents=[
            {
                "name": subagent,
                "description": f"The fund's {subagent}.",
                "system_prompt": "SUBAGENT-PROMPT",
            }
        ],
        checkpointer=saver,
    )
    agent.invoke(
        {"messages": [HumanMessage(content="Produce an allocation.")]},
        {"configurable": {"thread_id": thread_id}},
    )


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


def _capture_sql(session: Any):
    """Context manager collecting every SQL statement the session's connection runs."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        statements: list[str] = []

        def _record(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        bind = session.get_bind()
        event.listen(bind, "before_cursor_execute", _record)
        try:
            yield statements
        finally:
            event.remove(bind, "before_cursor_execute", _record)

    return _cm()


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


def test_list_portfolio_runs_does_not_load_agent_decisions(db_session):
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    AgentRunRepository(db_session).append_decision(
        run.id, agent="allocator", step="optimize_portfolio", llm_response="{}"
    )
    db_session.expunge_all()

    with _capture_sql(db_session) as statements:
        summaries = observe.list_portfolio_runs(db_session, _PID)

    assert len(summaries) == 1
    assert not [s for s in statements if "agent_decisions" in s]


def test_list_portfolio_runs_limit_keeps_only_the_newest(db_session):
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    newest = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )

    summaries = observe.list_portfolio_runs(db_session, _PID, limit=1)

    assert [s.run_id for s in summaries] == [newest.id]


def test_list_portfolio_runs_flags_responded_run_as_awaiting_hitl(db_session):
    # A barge-in ``respond`` that re-gated the PM leaves the run "responded": it is
    # still in the Approve queue (``list_paused_runs``), so the history flag agrees.
    responded = _make_run(
        db_session,
        portfolio_id=_PID,
        status="responded",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="rejected",
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )

    summaries = observe.list_portfolio_runs(db_session, _PID)

    by_id = {s.run_id: s for s in summaries}
    assert by_id[responded.id].awaiting_hitl is True
    assert [r.id for r in AgentRunRepository(db_session).list_paused_runs(_PID)] == [
        responded.id
    ]
    assert [s.awaiting_hitl for s in summaries] == [True, False]


def test_list_portfolio_runs_empty_for_unknown_portfolio(db_session):
    assert observe.list_portfolio_runs(db_session, uuid.uuid4()) == []


# --- list_portfolios (distinct ids across runs ∪ mandates ∪ profiles) --------


def test_list_portfolios_unions_runs_mandates_and_profiles(db_session):
    _make_run(
        db_session,
        portfolio_id=_PID_RUN,
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    MandateRepository(db_session).upsert(_mandate(str(_PID_MANDATE)))
    MifidProfileRepository(db_session).add_version(
        portfolio_id=_PID_MIFID,
        questionnaire={},
        constraint_set={},
        suitability={},
        store_key="constraint_set",
    )

    assert observe.list_portfolios(db_session) == sorted(
        [_PID_RUN, _PID_MANDATE, _PID_MIFID]
    )


def test_list_portfolios_dedups_across_sources_and_drops_null_run_ids(db_session):
    # Same id present as both a run and a mandate is counted once; a fund-wide run
    # (portfolio_id NULL) contributes nothing.
    _make_run(
        db_session,
        portfolio_id=_PID_RUN,
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    _make_run(
        db_session,
        portfolio_id=None,
        created_at=datetime(2026, 1, 11, tzinfo=UTC),
    )
    MandateRepository(db_session).upsert(_mandate(str(_PID_RUN)))

    assert observe.list_portfolios(db_session) == [_PID_RUN]


def test_list_portfolios_empty_fund_returns_empty_list(db_session):
    assert observe.list_portfolios(db_session) == []


def test_list_portfolios_is_exported_from_observe():
    assert "list_portfolios" in observe.__all__


# --- pending_hitl (paused + live-interrupt cross-check) ---------------------


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


def test_pending_hitl_surfaces_responded_run_with_a_live_interrupt(db_session):
    """A respond barge-in that re-gates sets status ``"responded"`` (not back to
    ``"paused"``); it must still surface as awaiting HITL so the re-gated run stays
    visible and approvable."""
    saver = MemorySaver()
    responded = _make_run(
        db_session,
        portfolio_id=_PID,
        status="responded",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-responded",
    )
    _seed_thread(
        saver,
        "thread-responded",
        [HumanMessage(content="go")],
        interrupt={"action_requests": [{"action": "place_orders"}]},
    )

    pending = observe.pending_hitl(db_session, saver, _PID)

    assert [p.run_id for p in pending] == [responded.id]
    assert pending[0].awaiting_hitl is True


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


def test_load_run_transcript_anchors_each_redelegation_to_its_own_message(db_session):
    # Risk rejects, the PM re-delegates to the allocator: the second proposal must
    # follow the rejection that caused it, not pile up at the first delegation.
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-redelegate",
    )
    audit = AgentRunRepository(db_session)
    for weights in ({"AAA": 0.9, "BBB": 0.1}, {"AAA": 0.6, "BBB": 0.4}):
        audit.append_decision(
            run.id,
            agent="allocator",
            step="optimize_portfolio",
            llm_response=json.dumps({"weights": weights}),
        )
    db_session.expire(run, ["decisions"])
    _seed_thread(
        saver,
        "thread-redelegate",
        [
            HumanMessage(content="Produce an allocation."),
            _task_msg("allocator"),
            ToolMessage(content="proposal 1", tool_call_id="call_allocator"),
            _task_msg("risk"),
            ToolMessage(content="REJECT: concentration", tool_call_id="call_risk"),
            _task_msg("allocator"),
            ToolMessage(content="proposal 2", tool_call_id="call_allocator"),
            _task_msg("risk"),
            ToolMessage(content="PASS", tool_call_id="call_risk"),
        ],
    )

    entries = observe.load_run_transcript(db_session, saver, run)

    assert [(e.source, e.index) for e in entries] == [
        ("message", 0),  # human
        ("message", 1),  # task(allocator) #1
        ("decision", 0),
        ("message", 2),  # proposal 1
        ("message", 3),  # task(risk) #1
        ("message", 4),  # REJECT
        ("message", 5),  # task(allocator) #2
        ("decision", 1),
        ("message", 6),  # proposal 2
        ("message", 7),  # task(risk) #2
        ("message", 8),  # PASS
    ]


def test_load_run_transcript_keeps_all_rows_of_a_single_delegation_together(
    db_session,
):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="thread-multi-row",
    )
    audit = AgentRunRepository(db_session)
    for _ in range(2):
        audit.append_decision(run.id, agent="allocator", step="optimize_portfolio")
    db_session.expire(run, ["decisions"])
    _seed_thread(
        saver,
        "thread-multi-row",
        [_task_msg("allocator"), _task_msg("risk")],
    )

    entries = observe.load_run_transcript(db_session, saver, run)

    assert [(e.source, e.index) for e in entries] == [
        ("message", 0),
        ("decision", 0),
        ("decision", 1),
        ("message", 1),
    ]


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


def test_load_run_transcript_reads_messages_from_a_real_deep_agent_checkpoint(
    db_session,
):
    saver = MemorySaver()
    _run_real_deep_agent(saver, "real-thread", narrative="risk-on tilt")
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="real-thread",
    )
    db_session.expire(run, ["decisions"])

    entries = observe.load_run_transcript(db_session, saver, run)

    texts = [e.text for e in entries if e.source == "message"]
    assert texts == [
        "Produce an allocation.",
        "",  # the PM's task delegation
        "risk-on tilt",  # the subagent report
        "allocation done",
    ]


def test_load_run_transcript_unwraps_a_delta_snapshot_at_the_target_checkpoint(
    db_session,
):
    from langgraph.checkpoint.serde.types import _DeltaSnapshot

    saver = MemorySaver()
    config = {"configurable": {"thread_id": "snap-thread", "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["messages"] = _DeltaSnapshot(
        [HumanMessage(content="first"), AIMessage(content="second")]
    )
    checkpoint["channel_versions"]["messages"] = _CHANNEL_VERSION
    saver.put(
        config,
        checkpoint,
        {"source": "loop", "step": 1},
        {"messages": _CHANNEL_VERSION},
    )
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="snap-thread",
    )
    db_session.expire(run, ["decisions"])

    entries = observe.load_run_transcript(db_session, saver, run)

    assert [(e.agent, e.text) for e in entries] == [
        ("human", "first"),
        ("ai", "second"),
    ]


def test_load_run_transcript_folds_writes_onto_the_nearest_delta_snapshot(db_session):
    from langgraph.checkpoint.serde.types import _DeltaSnapshot

    saver = MemorySaver()
    config = {"configurable": {"thread_id": "chain-thread", "checkpoint_ns": ""}}
    first = empty_checkpoint()
    first["channel_values"]["messages"] = _DeltaSnapshot(
        [HumanMessage(content="first", id="m1")]
    )
    first["channel_versions"]["messages"] = _CHANNEL_VERSION
    saved = saver.put(
        config,
        first,
        {"source": "loop", "step": 1},
        {"messages": _CHANNEL_VERSION},
    )
    # The step run from checkpoint 1 wrote an AI turn; checkpoint 2 omits the channel,
    # exactly as langgraph stores a DeltaChannel between snapshots.
    saver.put_writes(saved, [("messages", [AIMessage(content="second", id="m2")])], "t")
    second = empty_checkpoint()
    second["channel_versions"]["messages"] = "00000000000000000000000000000002.0.1"
    saver.put(
        saved,
        second,
        {"source": "loop", "step": 2},
        {"messages": "00000000000000000000000000000002.0.1"},
    )
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="chain-thread",
    )
    db_session.expire(run, ["decisions"])

    entries = observe.load_run_transcript(db_session, saver, run)

    assert [(e.agent, e.text) for e in entries] == [
        ("human", "first"),
        ("ai", "second"),
    ]


@pytest.mark.parametrize("subagent", ["economist", "allocator", "risk"])
def test_load_run_transcript_attributes_a_task_report_to_its_subagent(
    db_session, subagent
):
    # Production never names the subagent's report: it reaches the PM thread as a
    # ToolMessage(name='task'), tied to its delegation only by ``tool_call_id``.
    saver = MemorySaver()
    _run_real_deep_agent(
        saver, f"attr-{subagent}", narrative="the report", subagent=subagent
    )
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id=f"attr-{subagent}",
    )
    db_session.expire(run, ["decisions"])

    entries = observe.load_run_transcript(db_session, saver, run)

    reports = [e for e in entries if e.text == "the report"]
    assert [e.agent for e in reports] == [subagent]


def test_load_run_transcript_keeps_the_tool_name_for_a_non_task_tool_message(
    db_session,
):
    saver = MemorySaver()
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        created_at=datetime(2026, 3, 10, tzinfo=UTC),
        thread_id="attr-other-tool",
    )
    db_session.expire(run, ["decisions"])
    _seed_thread(
        saver,
        "attr-other-tool",
        [
            _task_msg("economist"),
            ToolMessage(content="ordered", name="place_orders", tool_call_id="x"),
            ToolMessage(content="orphan", name="task", tool_call_id="unknown"),
        ],
    )

    entries = observe.load_run_transcript(db_session, saver, run)

    # An unmatched task result keeps the tool name rather than guessing a subagent.
    assert [e.agent for e in entries] == ["ai", "place_orders", "task"]


def test_daily_brief_reads_the_economist_from_a_real_deep_agent_run(db_session):
    saver = MemorySaver()
    _run_real_deep_agent(saver, "brief-real", narrative="risk-on, tilt to tech")
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 1, 30, tzinfo=UTC),
        thread_id="brief-real",
    )

    brief = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 2, 2))

    assert brief.economist_read == "risk-on, tilt to tech"
    assert brief.economist_read_date == _ASOF
    assert brief.deliberation_run_id == run.id


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


@pytest.mark.parametrize("status", ["rejected", "incomplete", "blocked", "failed"])
def test_portfolio_state_ignores_proposal_of_a_run_that_ended_without_approval(
    db_session, status
):
    # A run that ended rejected/incomplete/blocked/failed leaves its allocator
    # proposal in the audit trail, but that proposal was never adopted: with nothing
    # held, adopting it as the target reads as drift 1.0 and re-triggers a rebalance.
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status=status,
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    AgentRunRepository(db_session).append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"AAA": 0.6, "BBB": 0.4}}),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.current == {}
    assert state.target == {}
    assert state.drift_l1 == 0.0


def test_portfolio_state_target_fallback_keeps_a_responded_runs_proposal(db_session):
    # ``responded`` is a live HITL gate like ``paused`` (a barge-in that re-gated), so
    # its pending proposal is still the target the adviser is looking at.
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="responded",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    AgentRunRepository(db_session).append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"AAA": 0.5, "BBB": 0.5}}),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"AAA": 0.5, "BBB": 0.5}


def test_portfolio_state_proposal_fallback_skips_ended_runs_for_a_live_gate(
    db_session,
):
    # A newer rejected run must not shadow an older run still awaiting a decision.
    live = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 1, 10, tzinfo=UTC),
    )
    ended = _make_run(
        db_session,
        portfolio_id=_PID,
        status="rejected",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    audit = AgentRunRepository(db_session)
    audit.append_decision(
        live.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"LIVE": 1.0}}),
    )
    audit.append_decision(
        ended.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"DECLINED": 1.0}}),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"LIVE": 1.0}


def _seed_eur_book(
    db_session,
    *,
    aaa_close_now: float,
    bbb_close_now: float,
    pre_fill_days: int = 0,
) -> None:
    """An approved 60/40 EUR book filled at 100/100, re-priced to the given closes.

    ``pre_fill_days`` adds that many daily bars (at 90) before the fill, the listing
    history a real instrument carries long before the book was built.
    """
    from portopt_db.models.market_data.yfinance_data import PriceHistory
    from portopt_db.models.orders.paper_order import PaperOrder
    from portopt_db.models.universe.universe import Exchange, Instrument

    fill_day, now_day = date(2026, 1, 5), date(2026, 1, 29)
    exchange = Exchange(name="X")
    db_session.add(exchange)
    db_session.flush()
    for ticker, close_now in (("AAA", aaa_close_now), ("BBB", bbb_close_now)):
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=exchange, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        db_session.add_all(
            [
                PriceHistory(
                    instrument_id=inst.id,
                    date=fill_day,
                    close=Decimal("100"),
                    price_unit="EUR",
                ),
                PriceHistory(
                    instrument_id=inst.id,
                    date=now_day,
                    close=Decimal(str(close_now)),
                    price_unit="EUR",
                ),
            ]
        )
        db_session.add_all(
            PriceHistory(
                instrument_id=inst.id,
                date=fill_day - timedelta(days=offset),
                close=Decimal("90"),
                price_unit="EUR",
            )
            for offset in range(1, pre_fill_days + 1)
        )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=date(2026, 1, 4),
            weights_hash="h1",
            weights={"AAA": 0.6, "BBB": 0.4},
            fill_date=fill_day,
            lines=[
                {"ticker": "AAA", "notional": 6000.0, "fill_price": 100.0},
                {"ticker": "BBB", "notional": 4000.0, "fill_price": 100.0},
            ],
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    PositionRepository(db_session).set_holdings(
        _PID,
        [{"ticker": "AAA", "weight": 0.6}, {"ticker": "BBB", "weight": 0.4}],
        asof=_ASOF,
    )
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 0.6, "BBB": 0.4},
        created_at=datetime(2026, 1, 4, tzinfo=UTC),
    )
    db_session.flush()


def test_portfolio_state_current_is_marked_to_market_so_a_rally_shows_as_drift(
    db_session,
):
    _seed_eur_book(db_session, aaa_close_now=130.0, bbb_close_now=100.0)

    state = observe.portfolio_state(db_session, _PID)

    # 60 sh x 130 = 7800 vs 40 sh x 100 = 4000 -> 66.1% / 33.9% against a 60/40 target.
    assert state.current["AAA"] == pytest.approx(7800 / 11800)
    assert state.current["BBB"] == pytest.approx(4000 / 11800)
    assert state.drift_l1 == pytest.approx(2 * (7800 / 11800 - 0.6))
    assert state.drift_l1 > 0.1


def test_portfolio_state_marks_from_the_bars_since_the_fill_not_the_full_history(
    db_session, monkeypatch
):
    """The TUI polls ``portfolio_state`` on its event loop, so it must stay bounded."""
    from portopt_db.repositories.market_data.yfinance_repository import (
        YFinanceRepository,
    )

    _seed_eur_book(
        db_session, aaa_close_now=130.0, bbb_close_now=100.0, pre_fill_days=400
    )
    rows_read: list[int] = []
    original = YFinanceRepository.get_price_history

    def spy(self, *args, **kwargs):
        rows = original(self, *args, **kwargs)
        rows_read.append(len(rows))
        return rows

    monkeypatch.setattr(YFinanceRepository, "get_price_history", spy)

    state = observe.portfolio_state(db_session, _PID)

    assert state.current["AAA"] == pytest.approx(7800 / 11800)
    assert state.drift_l1 > 0.1
    # 2 tickers x 400 pre-fill bars is 800+ rows when the whole history is loaded.
    assert sum(rows_read) < 150


def test_portfolio_state_without_a_priced_book_keeps_the_positions_snapshot(
    db_session,
):
    PositionRepository(db_session).set_holdings(
        _PID, [{"ticker": "AAA", "weight": 1.0}], asof=_ASOF
    )
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.current == {"AAA": 1.0}
    assert state.drift_l1 == pytest.approx(0.0)


def test_portfolio_state_marks_a_proposal_nobody_approved_as_unapproved(db_session):
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
        llm_response=json.dumps({"weights": {"AAA": 0.6, "BBB": 0.4}}),
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"AAA": 0.6, "BBB": 0.4}  # still shown for display
    assert state.target_approved is False


def test_portfolio_state_marks_a_completed_runs_weights_as_approved(db_session):
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )

    assert observe.portfolio_state(db_session, _PID).target_approved is True


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


def _portfolio_state_statement_count(db_session, n_runs: int) -> int:
    """SQL statements ``portfolio_state`` issues over ``n_runs`` proposal-less runs."""
    pid = uuid.uuid4()
    for i in range(n_runs):
        run = _make_run(
            db_session,
            portfolio_id=pid,
            status="paused",
            created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i),
        )
        AgentRunRepository(db_session).append_decision(
            run.id, agent="pm", step="note", llm_response="x"
        )
    db_session.expunge_all()

    with _capture_sql(db_session) as statements:
        observe.portfolio_state(db_session, pid)
    return len(statements)


def test_portfolio_state_query_count_does_not_grow_with_run_history(db_session):
    few = _portfolio_state_statement_count(db_session, 3)
    many = _portfolio_state_statement_count(db_session, 12)

    assert many == few


def _append_proposal(db_session, run, weights, metrics=None) -> None:
    body: dict[str, Any] = {"weights": weights}
    if metrics is not None:
        body["metrics"] = metrics
    AgentRunRepository(db_session).append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps(body),
    )


def test_portfolio_state_metrics_come_from_the_run_that_supplied_the_target(
    db_session,
):
    completed = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    _append_proposal(db_session, completed, {"AAA": 1.0}, {"sharpe": 1.0})
    pending = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 11, tzinfo=UTC),
    )
    _append_proposal(db_session, pending, {"BBB": 1.0}, {"sharpe": 9.9})

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"AAA": 1.0}
    assert state.metrics == {"sharpe": 1.0}


def test_portfolio_state_metrics_empty_when_the_target_run_persisted_none(
    db_session,
):
    completed = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    _append_proposal(db_session, completed, {"AAA": 1.0})
    pending = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 11, tzinfo=UTC),
    )
    _append_proposal(db_session, pending, {"BBB": 1.0}, {"sharpe": 9.9})

    state = observe.portfolio_state(db_session, _PID)

    assert state.metrics == {}


def test_portfolio_state_metrics_follow_a_pending_proposal_that_is_the_target(
    db_session,
):
    pending = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 11, tzinfo=UTC),
    )
    _append_proposal(db_session, pending, {"BBB": 1.0}, {"sharpe": 2.5})

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"BBB": 1.0}
    assert state.metrics == {"sharpe": 2.5}


def test_run_metrics_reads_the_requested_runs_proposal_not_the_newest_run(
    db_session,
):
    older = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    _append_proposal(db_session, older, {"AAA": 1.0}, {"sharpe": 1.0})
    newer = _make_run(
        db_session,
        portfolio_id=_PID,
        status="paused",
        created_at=datetime(2026, 2, 11, tzinfo=UTC),
    )
    _append_proposal(db_session, newer, {"BBB": 1.0}, {"sharpe": 9.9})

    assert observe.run_metrics(older) == {"sharpe": 1.0}
    assert observe.run_metrics(newer) == {"sharpe": 9.9}


def test_run_metrics_empty_when_the_run_has_no_allocator_proposal(db_session):
    run = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )

    assert observe.run_metrics(run) == {}


def test_portfolio_state_empty_portfolio(db_session):
    state = observe.portfolio_state(db_session, uuid.uuid4())
    assert state.current == {}
    assert state.target == {}
    assert state.drift_l1 == 0.0
    assert state.metrics == {}


def _seed_marked_book(
    db_session,
    *,
    last_aaa: str,
    fx_rate: float | None = 1.0,
    weight: float = 0.5,
    stale_rows: int = 0,
):
    """Seed an approved AAA/BBB book (positions == run target) + its prices.

    Both lines hold ``weight`` of a 10 000 book and fill at 100 on 10-02; on 10-03 AAA
    closes at ``last_aaa`` and BBB is unchanged, so the market weights drift away from
    the stored target snapshot. ``weight`` below 0.5 leaves a cash residual (a
    ``Bounds.budget`` below 1). ``stale_rows`` adds that many closes per ticker (and
    FX rows) a year *before* the fill, i.e. history the mark never needs.
    """
    from portopt_db.models.market_data.yfinance_data import PriceHistory
    from portopt_db.models.orders.paper_order import PaperOrder
    from portopt_db.models.universe.universe import Exchange, Instrument
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    d1, d2 = date(2026, 10, 2), date(2026, 10, 3)
    stale_days = [d1 - timedelta(days=365 + i) for i in range(stale_rows)]
    closes = {"AAA": ("100", last_aaa), "BBB": ("100", "100")}
    for ticker, (first, last) in closes.items():
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        db_session.add_all(
            [
                PriceHistory(
                    instrument_id=inst.id,
                    date=d1,
                    close=Decimal(first),
                    price_unit="USD",
                ),
                PriceHistory(
                    instrument_id=inst.id,
                    date=d2,
                    close=Decimal(last),
                    price_unit="USD",
                ),
                *(
                    PriceHistory(
                        instrument_id=inst.id,
                        date=d,
                        close=Decimal("1"),
                        price_unit="USD",
                    )
                    for d in stale_days
                ),
            ]
        )
    if fx_rate is not None:
        FxRatesRepository(db_session).upsert_rates(
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "date": d,
                    "rate": fx_rate,
                }
                for d in (d1, d2, *stale_days)
            ]
        )
    weights = {"AAA": weight, "BBB": weight}
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=date(2026, 10, 1),
            weights_hash="h1",
            weights=weights,
            fill_date=d1,
            lines=[
                {
                    "ticker": t,
                    "weight": weight,
                    "notional": 10000.0 * weight,
                    "fill_price": 100.0,
                }
                for t in weights
            ],
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    PositionRepository(db_session).set_holdings(
        _PID,
        [{"ticker": t, "weight": w} for t, w in weights.items()],
        asof=d1,
    )
    _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights=weights,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    db_session.flush()


def test_portfolio_state_drift_reflects_market_moves_after_approval(db_session):
    # After an approval the stored snapshot equals the target by construction, so
    # drift must come from the marked book: AAA +40% -> 0.5833 / 0.4167 vs 0.5 / 0.5.
    _seed_marked_book(db_session, last_aaa="140")

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"AAA": 0.5, "BBB": 0.5}
    assert state.current["AAA"] == pytest.approx(7000 / 12000)
    assert state.current["BBB"] == pytest.approx(5000 / 12000)
    assert state.drift_l1 == pytest.approx(2 * (7000 / 12000 - 0.5))


def test_portfolio_state_drift_is_zero_when_market_has_not_moved(db_session):
    _seed_marked_book(db_session, last_aaa="100")

    assert observe.portfolio_state(db_session, _PID).drift_l1 == pytest.approx(0.0)


def test_portfolio_state_falls_back_to_snapshot_when_book_is_unpriceable(db_session):
    # Foreign holdings with zero FX coverage cannot be marked; the stored snapshot is
    # the honest fallback rather than a fabricated or crashing figure.
    _seed_marked_book(db_session, last_aaa="140", fx_rate=None)

    state = observe.portfolio_state(db_session, _PID)

    assert state.current == {"AAA": 0.5, "BBB": 0.5}
    assert state.drift_l1 == pytest.approx(0.0)


def test_portfolio_state_marks_only_the_window_since_the_ticket_fill(
    db_session, monkeypatch
):
    # portfolio_state runs on the TUI's synchronous poll tick, so marking must not
    # read each holding's whole price/FX history: a year of pre-fill rows exists, but
    # only the fill-to-now window may be loaded.
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )
    from portopt_db.repositories.market_data.yfinance_repository import (
        YFinanceRepository,
    )

    _seed_marked_book(db_session, last_aaa="140", stale_rows=40)
    price_reads: list[tuple[date | None, list[date]]] = []
    fx_starts: list[date | None] = []
    real_prices = YFinanceRepository.get_price_history
    real_rates = FxRatesRepository.get_rates

    def spy_prices(self, instrument_id, **kwargs):
        rows = real_prices(self, instrument_id, **kwargs)
        price_reads.append((kwargs.get("start_date"), [r.date for r in rows]))
        return rows

    def spy_rates(self, base, quotes=None, start=None, end=None):
        fx_starts.append(start)
        return real_rates(self, base, quotes, start, end)

    monkeypatch.setattr(YFinanceRepository, "get_price_history", spy_prices)
    monkeypatch.setattr(FxRatesRepository, "get_rates", spy_rates)

    state = observe.portfolio_state(db_session, _PID)

    assert state.current["AAA"] == pytest.approx(7000 / 12000)
    window_floor = date(2026, 9, 1)
    assert price_reads
    assert all(start is not None and start >= window_floor for start, _ in price_reads)
    assert all(d >= window_floor for _, dates in price_reads for d in dates)
    assert fx_starts
    assert all(start is not None and start >= window_floor for start in fx_starts)


def test_portfolio_state_survives_a_marking_failure(db_session, monkeypatch):
    # The mark is best-effort: any failure inside it must degrade to the snapshot
    # rather than abort portfolio_state (which would blind the attention bus).
    _seed_marked_book(db_session, last_aaa="140")

    def exploding_mark(*args, **kwargs):
        raise RuntimeError("price feed exploded")

    monkeypatch.setattr("fund.performance.marked_weights", exploding_mark)

    state = observe.portfolio_state(db_session, _PID)

    assert state.current == {"AAA": 0.5, "BBB": 0.5}
    assert state.drift_l1 == pytest.approx(0.0)


def test_portfolio_state_rolls_a_failed_mark_back_to_its_own_savepoint(
    db_session, monkeypatch
):
    # On Postgres a failed statement poisons the whole transaction; the mark must run
    # in a SAVEPOINT so the snapshot fallback read can still use the session.
    from sqlalchemy import event

    _seed_marked_book(db_session, last_aaa="140")
    statements: list[str] = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    def exploding_mark(*args, **kwargs):
        raise RuntimeError("price feed exploded")

    monkeypatch.setattr("fund.performance.marked_weights", exploding_mark)
    bind = db_session.get_bind()
    event.listen(bind, "before_cursor_execute", record)
    try:
        observe.portfolio_state(db_session, _PID)
    finally:
        event.remove(bind, "before_cursor_execute", record)

    assert any("ROLLBACK TO SAVEPOINT" in statement for statement in statements)


def test_portfolio_state_budget_below_one_has_no_phantom_drift(db_session):
    # Bounds.budget < 1 leaves a cash residual, so the approved weights sum to 0.8.
    # Marking against NAV *including* that cash keeps ``current`` on the target's
    # basis; normalising by the invested total alone would read 0.5/0.5 vs 0.4/0.4
    # and report a permanent L1 drift of 0.2 with no market move.
    _seed_marked_book(db_session, last_aaa="100", weight=0.4)

    state = observe.portfolio_state(db_session, _PID)

    assert state.target == {"AAA": 0.4, "BBB": 0.4}
    assert state.current["AAA"] == pytest.approx(0.4)
    assert state.current["BBB"] == pytest.approx(0.4)
    assert state.drift_l1 == pytest.approx(0.0)


def test_portfolio_state_budget_below_one_still_shows_market_drift(db_session):
    # AAA +40%: lines 5600 / 4000 plus 2000 cash = 11600 NAV against a 0.4 / 0.4 target.
    _seed_marked_book(db_session, last_aaa="140", weight=0.4)

    state = observe.portfolio_state(db_session, _PID)

    assert state.current["AAA"] == pytest.approx(5600 / 11600)
    assert state.current["BBB"] == pytest.approx(4000 / 11600)
    assert state.drift_l1 == pytest.approx(1600 / 11600)


# --- get_mandate (JSON → pydantic rehydration) ------------------------------


def test_get_mandate_rehydrates_pydantic_from_json(db_session):
    pid = uuid.uuid4()
    mandate = _mandate(str(pid))
    MandateRepository(db_session).upsert(mandate)

    fetched = observe.get_mandate(db_session, pid)

    assert isinstance(fetched, PortfolioMandate)
    assert fetched == mandate


def test_get_mandate_clamps_a_legacy_over_bound_drift_threshold(db_session):
    # A row saved before the schema capped the threshold at the L1 bound of 2 must
    # keep loading (clamped) rather than raising on every read.
    pid = uuid.uuid4()
    row = MandateRepository(db_session).upsert(_mandate(str(pid)))
    row.mandate = {**row.mandate, "drift_l1_threshold": 5.0}
    row.drift_l1_threshold = 5.0
    db_session.flush()

    fetched = observe.get_mandate(db_session, pid)

    assert fetched is not None
    assert fetched.drift_l1_threshold == pytest.approx(2.0)
    assert fetched.drift_l1_threshold < 2.0  # the schema's open bound


def test_get_mandate_none_for_unknown_portfolio(db_session):
    assert observe.get_mandate(db_session, uuid.uuid4()) is None


# --- edge branches -----------------------------------------------------------


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
    # A rebuild-to-resume appends a second allocator optimize_portfolio decision to
    # the same run. The metrics must come from the newest decision (aligned with the
    # target-weights source), not the stale first one — non-allocator decisions in
    # between are skipped.
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


def test_metrics_are_not_carried_over_from_a_superseded_proposal(db_session):
    # An adviser edit appends an allocator decision with no metrics (the optimizer's
    # in-sample figures describe the weights it solved, not the edited vector). The
    # state panel must not show the earlier proposal's metrics beside the edit.
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
        llm_response=json.dumps({"weights": {"AAA": 1.0}}),
        hitl_decision={"decision": "edit", "human_override": True},
    )

    state = observe.portfolio_state(db_session, _PID)

    assert state.metrics == {}


def test_run_metrics_never_borrows_another_runs_numbers(db_session):
    # `portfolio_state.metrics` falls back across the portfolio's runs; `run_metrics`
    # is for per-run readers and must report only the run's own allocator proposal.
    older = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    newer = _make_run(
        db_session,
        portfolio_id=_PID,
        status="completed",
        weights={"AAA": 1.0},
        created_at=datetime(2026, 2, 10, tzinfo=UTC),
    )
    AgentRunRepository(db_session).append_decision(
        older.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": {"AAA": 1.0}, "metrics": {"sharpe": 0.8}}),
    )
    db_session.refresh(older)
    db_session.refresh(newer)

    assert observe.run_metrics(older) == {"sharpe": 0.8}
    assert observe.run_metrics(newer) == {}  # not the older run's 0.8


def test_every_public_observe_function_is_exported():
    public = {
        name
        for name, value in vars(observe).items()
        if not name.startswith("_")
        and callable(value)
        and getattr(value, "__module__", None) == observe.__name__
    }

    assert public <= set(observe.__all__)


def test_message_text_stringifies_non_str_non_list_content():
    # Defensive fallback for a message whose content is neither str nor block-list.
    assert observe._message_text(SimpleNamespace(content=None)) == "None"


def test_metrics_from_response_empty_for_unparseable_json():
    # Best-effort: an unparseable allocator payload yields no metrics (never faked).
    assert observe._metrics_from_response("not valid json{") == {}


# --- agent-stack-free import invariant --------------------------------------


def test_recent_events_returns_market_and_portfolio_events(db_session):
    MarketJournalRepository(db_session).upsert_journal(
        _ASOF,
        macro_deltas={},
        market_moves={},
        news_themes={"themes": {"rates": 1}},
        narrative="risk-off",
        source_counts={},
    )
    PortfolioJournalRepository(db_session).upsert(
        _PID,
        date(2026, 1, 20),
        run_id=None,
        trades={},
        allocation={"AAA": 1.0},
        drift={},
        narrative="rebalanced",
    )

    digest = observe.recent_events(db_session, _PID, asof=_ASOF)

    assert isinstance(digest, observe.EventDigest)
    assert digest.portfolio_id == _PID
    assert len(digest.market_events) == 1
    assert digest.market_events[0]["as_of"] == _ASOF.isoformat()
    assert digest.market_events[0]["narrative"] == "risk-off"
    assert len(digest.portfolio_events) == 1
    assert digest.portfolio_events[0]["narrative"] == "rebalanced"


def test_recent_events_excludes_market_digests_outside_the_window(db_session):
    MarketJournalRepository(db_session).upsert_journal(
        date(2025, 12, 1),  # well before _ASOF - window
        macro_deltas={},
        market_moves={},
        news_themes={},
        narrative="stale",
        source_counts={},
    )

    digest = observe.recent_events(db_session, _PID, asof=_ASOF, window_days=14)

    assert digest.market_events == ()


def test_recent_events_empty_for_portfolio_without_history(db_session):
    digest = observe.recent_events(db_session, uuid.uuid4(), asof=_ASOF)

    assert isinstance(digest, observe.EventDigest)
    assert digest.market_events == ()
    assert digest.portfolio_events == ()


_AUDIT_PID_A = uuid.UUID("aaaa0001-0000-4000-8000-000000000001")
_AUDIT_PID_B = uuid.UUID("bbbb0002-0000-4000-8000-000000000002")


def _seed_audit(db_session) -> None:
    """One agent decision (pf A) + an approve (pf A) + a flagged override (pf B)."""
    from fund.audit.human_audit_repository import HumanAuditLogRepository

    repo = AgentRunRepository(db_session)
    run = repo.create_run(
        portfolio_id=_AUDIT_PID_A,
        asof=date(2026, 1, 2),
        seed=1,
        universe=[],
        optimizer_config={},
    )
    repo.append_decision(run.id, agent="allocator", step="optimize_portfolio")
    hlog = HumanAuditLogRepository(db_session)
    hlog.append(action="approve", portfolio_id=_AUDIT_PID_A, run_id=run.id)
    hlog.append(
        action="guardrail_override",
        portfolio_id=_AUDIT_PID_B,
        override_flag=True,
        suitability_flag=True,
    )


def test_audit_timeline_unions_agent_and_human_newest_first(db_session):
    _seed_audit(db_session)

    rows = observe.audit_timeline(db_session)
    assert len(rows) == 3  # 1 agent decision + 2 human actions
    assert {row.kind for row in rows} == {"agent", "human"}
    # Newest-first by created_at.
    assert all(rows[i].when >= rows[i + 1].when for i in range(len(rows) - 1))


def test_audit_timeline_filters_compose(db_session):
    _seed_audit(db_session)

    assert len(observe.audit_timeline(db_session, kind="agent")) == 1
    human = observe.audit_timeline(db_session, kind="human")
    assert len(human) == 2
    assert all(row.kind == "human" for row in human)
    # overrides-only drops every agent row + the unflagged human approve.
    overrides = observe.audit_timeline(db_session, overrides_only=True)
    assert len(overrides) == 1
    assert overrides[0].override_flag is True
    # portfolio scope keeps only that portfolio's rows (both sides).
    assert len(observe.audit_timeline(db_session, portfolio_id=_AUDIT_PID_A)) == 2
    assert len(observe.audit_timeline(db_session, portfolio_id=_AUDIT_PID_B)) == 1


def test_audit_timeline_carries_the_target_and_rationale_of_human_rows(db_session):
    from fund.audit.human_audit_repository import HumanAuditLogRepository

    HumanAuditLogRepository(db_session).append(
        action="rearm",
        target="fund",
        new_value={"state": "armed"},
        rationale="drawdown reviewed with the risk committee",
    )

    (row,) = observe.audit_timeline(db_session, kind="human")

    assert row.target == "fund"
    assert row.rationale == "drawdown reviewed with the risk committee"


def test_audit_timeline_agent_rows_carry_no_target_or_rationale(db_session):
    _seed_audit(db_session)

    (agent_row,) = observe.audit_timeline(db_session, kind="agent")

    assert agent_row.target is None
    assert agent_row.rationale is None


def _seed_one_transaction_run(db_session, steps: list[str]) -> None:
    """Append ``steps`` as one run's decisions, all stamped with the same instant.

    Postgres ``now()`` is the transaction start time, so every decision a run flushes
    inside one caller-owned transaction carries an identical ``created_at``.
    """
    from portopt_db.models import AgentDecision
    from sqlalchemy import update

    repo = AgentRunRepository(db_session)
    run = repo.create_run(
        portfolio_id=_AUDIT_PID_A,
        asof=date(2026, 1, 2),
        seed=1,
        universe=[],
        optimizer_config={},
    )
    for step in steps:
        repo.append_decision(run.id, agent="allocator", step=step)
    db_session.execute(
        update(AgentDecision).values(created_at=datetime(2026, 1, 2, 9, 0, tzinfo=UTC))
    )
    db_session.flush()
    db_session.expire_all()


def test_audit_timeline_orders_same_instant_decisions_newest_step_first(db_session):
    steps = ["optimize_portfolio", "place_orders", "report"]
    _seed_one_transaction_run(db_session, steps)

    rows = observe.audit_timeline(db_session, kind="agent")

    assert [row.action for row in rows] == list(reversed(steps))


def test_audit_timeline_limit_keeps_the_latest_step_of_a_same_instant_run(db_session):
    _seed_one_transaction_run(
        db_session, ["optimize_portfolio", "place_orders", "report"]
    )

    rows = observe.audit_timeline(db_session, kind="agent", limit=1)

    assert [row.action for row in rows] == ["report"]


def test_audit_timeline_entries_carry_a_stable_unique_identity(db_session):
    _seed_audit(db_session)

    first = observe.audit_timeline(db_session)
    again = observe.audit_timeline(db_session)

    ids = [row.entry_id for row in first]
    assert len(set(ids)) == len(ids) == 3
    assert {i.split(":")[0] for i in ids} == {"agent", "human"}
    assert sorted(ids) == sorted(row.entry_id for row in again)


def test_audit_timeline_limit_caps_the_merge_and_none_lifts_it(db_session):
    _seed_audit(db_session)

    assert len(observe.audit_timeline(db_session, limit=2)) == 2
    assert len(observe.audit_timeline(db_session, limit=None)) == 3


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


# --- performance_summary (Phase 05 T11) -------------------------------------


def _seed_navs(db_session, navs, start=date(2026, 10, 1)):
    """Upsert a NAV series for _PID over consecutive days (no commit)."""
    rows = [
        {
            "portfolio_id": _PID,
            "date": start + timedelta(days=i),
            "nav": Decimal(str(nav)),
            "benchmark": None,
        }
        for i, nav in enumerate(navs)
    ]
    NavHistoryRepository(db_session).upsert_navs(rows)
    db_session.flush()


def test_performance_summary_reads_persisted_navs(db_session):
    """It reads the persisted nav_history series and derives the metrics."""
    _seed_navs(db_session, [100.0, 102.0, 101.0, 103.0, 106.0])

    summary = observe.performance_summary(db_session, _PID)

    assert summary.portfolio_id == _PID
    assert summary.returns["max"] == pytest.approx(106.0 / 100.0 - 1.0)
    assert summary.volatility is not None


def test_performance_summary_empty_is_graceful(db_session):
    """A portfolio with no persisted NAV yields an all-None summary, no raise."""
    summary = observe.performance_summary(db_session, _PID)

    assert summary.portfolio_id == _PID
    assert summary.volatility is None
    assert summary.returns["max"] is None


# --- holdings_summary (Phase 08 T1) -----------------------------------------


def _seed_holding_book(db_session, *, with_mandate: bool = False):
    """Seed a one-line USD paper book + two days of prices/FX for _PID (EUR base)."""
    from portopt_db.models.market_data.yfinance_data import PriceHistory
    from portopt_db.models.orders.paper_order import PaperOrder
    from portopt_db.models.universe.universe import Exchange, Instrument
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker="US1", short_name="US1", exchange=ex, yfinance_ticker="US1"
    )
    db_session.add(inst)
    db_session.flush()
    d1, d2 = date(2026, 10, 2), date(2026, 10, 3)
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=inst.id, date=d1, close=Decimal("100"), price_unit="USD"
            ),
            PriceHistory(
                instrument_id=inst.id, date=d2, close=Decimal("110"), price_unit="USD"
            ),
        ]
    )
    FxRatesRepository(db_session).upsert_rates(
        [
            {"base_currency": "EUR", "quote_currency": "USD", "date": d1, "rate": 0.90},
            {"base_currency": "EUR", "quote_currency": "USD", "date": d2, "rate": 0.92},
        ]
    )
    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=date(2026, 10, 1),
            weights_hash="h1",
            weights={"US1": 1.0},
            fill_date=d1,
            lines=[{"ticker": "US1", "notional": 9000.0, "fill_price": 100.0}],
            notional=9000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )
    if with_mandate:
        MandateRepository(db_session).upsert(
            PortfolioMandate(
                portfolio_id=str(_PID),
                capital=Decimal("100000"),
                base_currency="EUR",
                drift_l1_threshold=0.1,
                triggers=RunTriggers(cron=True, drift=True),
            )
        )
    db_session.flush()


def test_holdings_summary_marks_current_book_default_base(db_session):
    """With no mandate the base defaults to EUR; the current book is marked."""
    _seed_holding_book(db_session)

    marks = observe.holdings_summary(db_session, _PID)

    assert len(marks) == 1
    us1 = marks[0]
    assert us1.ticker == "US1"
    # shares 9000/90 marked at 110*0.92 = 101.2 base.
    assert us1.value == pytest.approx((9000.0 / 90.0) * 101.2, rel=1e-6)
    assert us1.weight == pytest.approx(1.0, rel=1e-6)


def test_holdings_summary_uses_mandate_base_currency(db_session):
    """An EUR mandate is read for the base currency (the non-default branch)."""
    _seed_holding_book(db_session, with_mandate=True)

    marks = observe.holdings_summary(db_session, _PID)

    assert [m.ticker for m in marks] == ["US1"]
    assert marks[0].value is not None


def test_holdings_summary_marks_in_the_mandate_capital(db_session):
    """Holdings are valued on the mandate's capital, not the ticket's paper notional."""
    _seed_holding_book(db_session, with_mandate=True)  # ticket 9,000; capital 100,000

    marks = observe.holdings_summary(db_session, _PID)

    # shares 100000/90 marked at 110*0.92 = 101.2 base.
    assert marks[0].value == pytest.approx((100_000.0 / 90.0) * 101.2, rel=1e-6)


def test_holdings_summary_unmarked_is_empty(db_session):
    """A portfolio with no paper orders degrades to an empty list, not a crash."""
    assert observe.holdings_summary(db_session, _PID) == []


# --- macro_attribution_summary (Phase 06 T3) --------------------------------


def _seed_navs_and_fred(db_session, *, n: int = 20, start=date(2026, 1, 5)):
    """Persist a NAV+benchmark series and matching FRED levels over ``n`` days."""
    from portopt_db.models.macro.macro_regime import FredObservation

    dates = [start + timedelta(days=i) for i in range(n)]
    rows = [
        {
            "portfolio_id": _PID,
            "date": d,
            "nav": Decimal(str(100.0 + i + (i % 3))),
            "benchmark": Decimal(str(200.0 + i)),
        }
        for i, d in enumerate(dates)
    ]
    NavHistoryRepository(db_session).upsert_navs(rows)
    for series_id, base in (("DGS10", 4.0), ("BAA10Y", 1.8), ("DTWEXBGS", 100.0)):
        # A non-monotonic wobble so each factor's return has positive variance (a
        # constant level would diff to a zero-variance factor and be dropped).
        db_session.add_all(
            FredObservation(
                series_id=series_id, date=d, value=base + 0.05 * ((i * 13) % 7)
            )
            for i, d in enumerate(dates)
        )
    db_session.flush()


def test_macro_attribution_summary_reconstructs_total(db_session):
    """It reads persisted NAV + FRED and attributes; pieces sum to the total."""
    from fund.attribution import AttributionResult

    _seed_navs_and_fred(db_session)

    result = observe.macro_attribution_summary(db_session, _PID)

    assert isinstance(result, AttributionResult)
    assert result.factor_names  # a real multi-factor fit ran
    assert sum(result.contributions.values()) + result.residual == pytest.approx(
        result.total_return, abs=1e-9
    )


def test_macro_attribution_summary_empty_is_graceful(db_session):
    """A portfolio with no persisted NAV degrades to a zeroed, flagged result."""
    result = observe.macro_attribution_summary(db_session, _PID)

    assert result.total_return == pytest.approx(0.0)
    assert result.contributions == {}
    assert result.insufficient_history


# --- daily_brief (Phase 07 T1) ----------------------------------------------


def _seed_economist_run(
    db_session,
    saver,
    *,
    asof,
    created_at,
    thread_id,
    narrative,
    status="completed",
):
    """Insert a run for _PID whose thread carries one named economist message."""
    run = AgentRun(
        portfolio_id=_PID,
        asof=asof,
        seed=1,
        universe=["AAA"],
        optimizer_config={},
        status=status,
        weights={"AAA": 1.0},
        thread_id=thread_id,
        created_at=created_at,
        finished_at=created_at,
    )
    db_session.add(run)
    db_session.flush()
    _seed_thread(
        saver,
        thread_id,
        [
            HumanMessage(content="Produce an allocation."),
            AIMessage(content=narrative, name="economist"),
        ],
    )
    return run


def test_daily_brief_full_day_stitches_all_five_sections(db_session):
    """A fully-marked day populates P&L, attribution, events, read, and overlays."""
    saver = MemorySaver()
    _seed_navs_and_fred(db_session)  # consecutive marks 2026-01-05 .. 2026-01-24
    day = date(2026, 1, 24)
    MarketJournalRepository(db_session).upsert_journal(
        day,
        macro_deltas={},
        market_moves={},
        news_themes={},
        narrative="CPI < expected",
        source_counts={},
    )
    PortfolioJournalRepository(db_session).upsert(
        _PID,
        day,
        run_id=None,
        trades={},
        allocation={"AAA": 1.0},
        drift={},
        narrative="rebalanced",
    )
    run = _seed_economist_run(
        db_session,
        saver,
        asof=day,
        created_at=datetime(2026, 1, 24, 10, tzinfo=UTC),
        thread_id="brief-full",
        narrative="risk-on; your tech tilt benefited",
    )

    brief = observe.daily_brief(db_session, saver, _PID, asof=day)

    assert isinstance(brief, observe.DailyBrief)
    assert brief.asof == day
    assert brief.prev_day == date(2026, 1, 23)
    assert brief.next_day is None  # the latest marked day
    assert brief.performance is not None
    assert brief.performance.returns["1D"] is not None
    assert brief.attribution is not None
    assert brief.attribution.factor_names  # a real multi-factor fit ran
    recon = sum(brief.attribution.contributions.values()) + brief.attribution.residual
    assert recon == pytest.approx(brief.attribution.total_return, abs=1e-9)
    assert any("CPI" in event["narrative"] for event in brief.market_events)
    assert any("rebalanced" in event["narrative"] for event in brief.fund_events)
    assert brief.economist_read == "risk-on; your tech tilt benefited"
    assert brief.economist_read_date == day
    assert brief.deliberation_run_id == run.id


def test_daily_brief_carries_forward_economist_read_dated(db_session):
    """A day with no run reuses the most-recent-on-or-before run's dated read."""
    saver = MemorySaver()
    _seed_navs(db_session, [100.0, 102.0, 101.0, 103.0, 106.0])  # 10-01 .. 10-05
    run = _seed_economist_run(
        db_session,
        saver,
        asof=date(2026, 10, 2),
        created_at=datetime(2026, 10, 2, tzinfo=UTC),
        thread_id="brief-cf",
        narrative="watch ECB on rates",
    )

    brief = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 10, 4))

    assert brief.economist_read == "watch ECB on rates"
    assert brief.economist_read_date == date(2026, 10, 2)
    assert brief.deliberation_run_id == run.id


def test_daily_brief_economist_tie_break_prefers_later_created_at(db_session):
    """Two runs sharing an asof resolve to the later-created one (not list order)."""
    saver = MemorySaver()
    _seed_navs(db_session, [100.0, 102.0, 101.0])  # 10-01 .. 10-03
    day = date(2026, 10, 3)
    _seed_economist_run(
        db_session,
        saver,
        asof=day,
        created_at=datetime(2026, 10, 3, 9, tzinfo=UTC),
        thread_id="brief-tie-early",
        narrative="earlier read",
    )
    later = _seed_economist_run(
        db_session,
        saver,
        asof=day,
        created_at=datetime(2026, 10, 3, 17, tzinfo=UTC),
        thread_id="brief-tie-late",
        narrative="later read",
    )

    brief = observe.daily_brief(db_session, saver, _PID, asof=day)

    assert brief.economist_read == "later read"
    assert brief.deliberation_run_id == later.id


def _seed_profiler_run(db_session, saver, *, asof, created_at, thread_id):
    """A profiling attempt: same portfolio, ``step == 'profiler'``, no economist."""
    run = AgentRun(
        portfolio_id=_PID,
        asof=asof,
        seed=1,
        universe=[],
        optimizer_config={"step": "profiler"},
        status="completed",
        weights={},
        thread_id=thread_id,
        created_at=created_at,
        finished_at=created_at,
    )
    db_session.add(run)
    db_session.flush()
    _seed_thread(saver, thread_id, [HumanMessage(content="Profile this client.")])
    return run


def test_daily_brief_economist_read_skips_a_later_profiler_run(db_session):
    """A re-profile after the last rebalance must not shadow its economist read."""
    saver = MemorySaver()
    fund_run = _seed_economist_run(
        db_session,
        saver,
        asof=date(2026, 10, 1),
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        thread_id="brief-fund",
        narrative="watch ECB on rates",
    )
    _seed_profiler_run(
        db_session,
        saver,
        asof=date(2026, 10, 23),
        created_at=datetime(2026, 10, 23, tzinfo=UTC),
        thread_id="brief-profiler",
    )

    brief = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 10, 23))

    assert brief.economist_read == "watch ECB on rates"
    assert brief.economist_read_date == date(2026, 10, 1)
    assert brief.deliberation_run_id == fund_run.id


def test_daily_brief_economist_read_skips_a_same_day_profiler_run(db_session):
    """A profiler run created after a same-asof fund run is not its tie-break winner."""
    saver = MemorySaver()
    day = date(2026, 10, 3)
    fund_run = _seed_economist_run(
        db_session,
        saver,
        asof=day,
        created_at=datetime(2026, 10, 3, 9, tzinfo=UTC),
        thread_id="brief-same-fund",
        narrative="rebalance read",
    )
    _seed_profiler_run(
        db_session,
        saver,
        asof=day,
        created_at=datetime(2026, 10, 3, 17, tzinfo=UTC),
        thread_id="brief-same-profiler",
    )

    brief = observe.daily_brief(db_session, saver, _PID, asof=day)

    assert brief.economist_read == "rebalance read"
    assert brief.deliberation_run_id == fund_run.id


def test_daily_brief_economist_read_blank_when_only_profiler_runs_exist(db_session):
    saver = MemorySaver()
    _seed_profiler_run(
        db_session,
        saver,
        asof=date(2026, 10, 2),
        created_at=datetime(2026, 10, 2, tzinfo=UTC),
        thread_id="brief-only-profiler",
    )

    brief = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 10, 4))

    assert brief.economist_read is None
    assert brief.economist_read_date is None
    assert brief.deliberation_run_id is None


def test_daily_brief_blank_economist_read_when_never_run(db_session):
    """A portfolio that never ran yields no read, date, or deliberation id."""
    saver = MemorySaver()
    _seed_navs(db_session, [100.0, 102.0])

    brief = observe.daily_brief(db_session, saver, _PID)

    assert brief.economist_read is None
    assert brief.economist_read_date is None
    assert brief.deliberation_run_id is None


def test_daily_brief_attribution_covers_only_the_briefed_day(db_session):
    """The brief's "today" attribution explains that day's NAV return only."""
    saver = MemorySaver()
    _seed_navs_and_fred(db_session)  # nav = 100 + i + (i % 3), one mark per day

    latest = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 1, 24))
    prior = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 1, 23))

    assert latest.attribution is not None and prior.attribution is not None
    assert latest.attribution.total_return == pytest.approx(120.0 / 118.0 - 1.0)
    assert prior.attribution.total_return == pytest.approx(118.0 / 119.0 - 1.0)
    for brief in (latest, prior):
        recon = sum(brief.attribution.contributions.values()) + (
            brief.attribution.residual
        )
        assert recon == pytest.approx(brief.attribution.total_return, abs=1e-12)


def test_daily_brief_repeat_poll_does_not_refit_the_exposures(db_session, monkeypatch):
    """The cockpit polls the brief every tick; unchanged data must not refit."""
    import optimizer.factors as factors_mod

    saver = MemorySaver()
    _seed_navs_and_fred(db_session)
    real = factors_mod.fit_factor_prior
    fits: list[int] = []

    def counting(*args, **kwargs):
        fits.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(factors_mod, "fit_factor_prior", counting)

    first = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 1, 24))
    second = observe.daily_brief(db_session, saver, _PID, asof=date(2026, 1, 24))

    assert len(fits) == 1
    assert second.attribution == first.attribution


def test_daily_brief_unmarked_day_degrades_without_crash(db_session):
    """No NAV / journal / run → None performance, flagged attribution, empty tuples."""
    saver = MemorySaver()

    brief = observe.daily_brief(db_session, saver, _PID)

    assert brief.performance is None
    assert brief.attribution is not None
    assert brief.attribution.insufficient_history
    assert brief.market_events == ()
    assert brief.fund_events == ()
    assert brief.economist_read is None
    assert brief.prev_day is None
    assert brief.next_day is None


def test_daily_brief_defaults_to_latest_nav_mark(db_session):
    """With asof=None the brief opens on the latest NAV-marked day."""
    saver = MemorySaver()
    _seed_navs(db_session, [100.0, 102.0, 101.0, 103.0])  # 10-01 .. 10-04

    brief = observe.daily_brief(db_session, saver, _PID)

    assert brief.asof == date(2026, 10, 4)
    assert brief.prev_day == date(2026, 10, 3)
    assert brief.next_day is None


def test_daily_brief_is_public_and_reachable():
    """The assembler is exported so the agent-stack-free guard covers it."""
    assert "daily_brief" in observe.__all__
    assert "DailyBrief" in observe.__all__
    assert callable(observe.daily_brief)


# --- default_decision_bar ---------------------------------------------------

_TODAY = date(2026, 10, 24)  # Saturday; the panel below ends Friday 10-23
_FRI = date(2026, 10, 23)


def _seed_bars(session, last_bar: date, *, n_bars: int = 5, n: int = 1) -> None:
    """Seed ``n`` instruments with ``n_bars`` consecutive daily closes ending at
    ``last_bar``; each call adds fresh instruments."""
    exchange = Exchange(name=f"EX-{uuid.uuid4().hex[:8]}")
    session.add(exchange)
    session.flush()
    for _ in range(n):
        inst = Instrument(
            ticker=uuid.uuid4().hex[:8],
            short_name="X",
            exchange_id=exchange.id,
            instrument_type="EQUITY",
            asset_class="equity",
        )
        session.add(inst)
        session.flush()
        for i in range(n_bars):
            session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=last_bar - timedelta(days=n_bars - 1 - i),
                    close=100.0 + i,
                )
            )
    session.flush()


def test_default_decision_bar_is_the_day_before_the_latest_stored_bar(db_session):
    _seed_bars(db_session, _FRI, n=3)

    bar = observe.default_decision_bar(db_session, today=_TODAY)

    assert bar == _FRI - timedelta(days=1)  # the Friday close is the fill bar


def test_default_decision_bar_ignores_a_bar_only_one_instrument_has(db_session):
    _seed_bars(db_session, _FRI, n=9)
    _seed_bars(db_session, _TODAY, n_bars=1, n=1)  # a lone weekend (e.g. crypto) bar

    bar = observe.default_decision_bar(db_session, today=_TODAY)

    assert bar == _FRI - timedelta(days=1)  # the other 9 would have no later bar


def test_default_decision_bar_tolerates_a_small_lagging_share(db_session):
    _seed_bars(db_session, _FRI, n=9)
    _seed_bars(db_session, _FRI - timedelta(days=1), n=1)  # one market on holiday

    bar = observe.default_decision_bar(db_session, today=_TODAY)

    assert bar == _FRI - timedelta(days=1)


def test_default_decision_bar_waits_for_a_large_lagging_share(db_session):
    _seed_bars(db_session, _FRI, n=5)
    _seed_bars(db_session, _FRI - timedelta(days=1), n=5)  # half not yet ingested

    bar = observe.default_decision_bar(db_session, today=_TODAY)

    assert bar == _FRI - timedelta(days=2)  # every ticker has a bar after it


def test_default_decision_bar_ignores_dormant_and_future_bars(db_session):
    _seed_bars(db_session, _FRI, n=3)
    _seed_bars(db_session, _FRI - timedelta(days=60), n=5)  # delisted: no recent bar
    _seed_bars(db_session, _TODAY + timedelta(days=3), n_bars=1, n=10)  # after today

    bar = observe.default_decision_bar(db_session, today=_TODAY)

    assert bar == _FRI - timedelta(days=1)


def test_default_decision_bar_falls_back_to_today_without_recent_prices(db_session):
    assert observe.default_decision_bar(db_session, today=_TODAY) == _TODAY
