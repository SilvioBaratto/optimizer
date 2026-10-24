"""``stream_fund`` — the live-deliberation producer under a scripted, no-network round.

Drives :func:`fund.agents.graph.stream_fund` over the same ``_fund_fakes`` stack the
TUI tests use (``ScriptedFundModel`` + ``MemorySaver`` + ``InMemoryStore`` + in-memory
SQLite) and asserts the T1 contract: events arrive in **emission order** carrying the
agent tag + step identity, the stream ends on a single terminal ``paused`` event, the
run is flipped ``paused`` so the Decisions queue finds it, and ``import
fund.tui.stream`` drags in neither ``deepagents`` nor ``langgraph``.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    FreezeMidRunModel,
    ScriptedFundModel,
    expected_weights,
    make_constraint_set,
    make_mandate,
    seed_panel,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fund.agents.graph import stream_fund
from fund.audit import (
    AgentRunRepository,
    AutonomyRepository,
    AutonomyScope,
    AutonomyState,
    put_constraint_set,
)
from fund.config import settings
from fund.tui.stream import StepStatus, StreamEvent

# The four subagents report in fixed pipeline order, then the PM commits.
_EXPECTED_AGENTS = ["economist", "allocator", "risk", "executor", "orchestrator"]


@pytest.fixture
def stream_env():
    """An isolated SQLite engine + in-memory LangGraph handles + a scripted model."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one connection => one in-memory DB across threads
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    saver = MemorySaver()
    store = InMemoryStore()

    @contextmanager
    def get_session():
        """Yield a short SQLAlchemy session from the isolated engine, closed on exit."""
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    with get_session() as session:
        seed_panel(session)
        session.commit()
    put_constraint_set(
        store, make_constraint_set(), store_key=settings.constraint_set_store_key
    )
    with get_session() as session:
        expected = expected_weights(session)

    yield SimpleNamespace(
        get_session=get_session,
        saver=saver,
        store=store,
        expected=expected,
        model_factory=lambda: ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )
    engine.dispose()


def test_stream_fund_yields_ordered_tagged_events_then_pauses(stream_env) -> None:
    """Stream a scripted round: ordered tagged events ending in one paused gate."""
    with stream_env.get_session() as session:
        events = list(
            stream_fund(
                stream_env.model_factory(),
                make_mandate(),
                portfolio_id=PORTFOLIO_ID,
                asof=ASOF,
                session=session,
                checkpointer=stream_env.saver,
                store=stream_env.store,
            )
        )

        # Contiguous emission order, every row a StreamEvent.
        assert events, "stream yielded nothing"
        assert all(isinstance(event, StreamEvent) for event in events)
        assert [event.index for event in events] == list(range(len(events)))

        # Each pipeline step carries an agent tag + a step identity, in order.
        step_events = [event for event in events if event.kind == "step"]
        assert all(event.agent and event.step for event in step_events)
        assert [event.agent for event in step_events] == _EXPECTED_AGENTS
        assert step_events[-1].step == "place_orders"

        # Exactly one terminal event: the live place_orders gate.
        terminal = events[-1]
        assert [e for e in events if e.kind != "step"] == [terminal]
        assert terminal.kind == "paused"
        assert terminal.status is StepStatus.PENDING
        names = [
            request["name"]
            for request in (terminal.payload or {}).get("action_requests", [])
        ]
        assert "place_orders" in names

        # The run is flipped paused so the Decisions queue / resume driver find it.
        run = AgentRunRepository(session).get_run(terminal.run_id)
        assert run is not None
        assert run.status == "paused"


def test_stream_fund_pm_round_cap_end_records_distinct_reason(stream_env) -> None:
    """A PM-round-cap end finalises incomplete as ``"pm_round_cap"``, matching
    ``run_fund`` — not the generic ``"no_order"``.

    Regression guard for the streaming path: the updates stream never carries the final
    merged state, so ``stream_fund`` must latch ``pm_round_cap_reached`` as the cap's
    delta streams past, else it would mis-record the reason and diverge from
    ``run_fund`` on an otherwise identical end.
    """
    model = ScriptedFundModel(universe=UNIVERSE, weights={"AAA": 1.0}, overrun=True)
    # LOW pm-round cap under the default (high) recursion_limit, so the middleware trips
    # before the langgraph guard — an incomplete end, not a GraphRecursionError.
    capped = dataclasses.replace(settings, max_pm_rounds=3)

    with stream_env.get_session() as session:
        events = list(
            stream_fund(
                model,
                make_mandate(),
                portfolio_id=PORTFOLIO_ID,
                asof=ASOF,
                session=session,
                checkpointer=stream_env.saver,
                store=stream_env.store,
                config=capped,
            )
        )

        terminal = events[-1]
        assert terminal.kind == "incomplete"
        run = AgentRunRepository(session).get_run(terminal.run_id)
        assert run is not None
        assert run.status == "incomplete"
        reasons = [
            d.hitl_decision.get("reason")
            for d in session.query(AgentDecision)
            .filter(AgentDecision.run_id == terminal.run_id)
            .all()
            if d.hitl_decision
        ]
        assert "pm_round_cap" in reasons
        assert "no_order" not in reasons
        assert "round_cap" not in reasons


def test_stream_fund_halts_at_the_gate_when_frozen_while_in_flight(
    stream_env,
) -> None:
    """A freeze committed mid-stream ends the round ``incomplete``, never ``paused``.

    Otherwise the Decisions queue would list an ordinary gate under a FROZEN fund and
    one keypress would approve it.
    """

    def freeze() -> None:
        with stream_env.get_session() as other:
            AutonomyRepository(other).set_state(
                AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="killswitch"
            )
            other.commit()

    model = FreezeMidRunModel(
        universe=UNIVERSE,
        weights=stream_env.expected,
        on_first_call=freeze,
    )
    with stream_env.get_session() as session:
        events = list(
            stream_fund(
                model,
                make_mandate(),
                portfolio_id=PORTFOLIO_ID,
                asof=ASOF,
                session=session,
                checkpointer=stream_env.saver,
                store=stream_env.store,
            )
        )

        terminal = events[-1]
        assert terminal.kind == "incomplete"
        assert not [e for e in events if e.kind == "paused"]
        run = AgentRunRepository(session).get_run(terminal.run_id)
        assert run is not None
        assert run.status == "rejected"


@pytest.mark.parametrize(
    ("kind", "status", "expected"),
    [
        ("step", StepStatus.LIVE, "\u25cf live"),
        ("step", StepStatus.DONE, "\u2713 done"),
        ("paused", StepStatus.PENDING, "\u00b7 paused"),
        ("incomplete", StepStatus.PENDING, "\u00b7 incomplete"),
    ],
)
def test_status_text_pairs_the_glyph_with_a_word(
    kind: str, status: StepStatus, expected: str
) -> None:
    """A step states its liveness and a terminal event its outcome, never a bare dot."""
    event = StreamEvent(
        kind=kind,
        run_id=uuid.uuid4(),
        index=0,
        agent="orchestrator",
        step="place_orders",
        status=status,
        text="",
    )

    assert event.status_text == expected


def test_import_stream_is_agent_stack_light() -> None:
    """``import fund.tui.stream`` must pull neither ``deepagents`` nor ``langgraph``."""
    code = (
        "import sys, fund.tui.stream; "
        "assert 'deepagents' not in sys.modules, 'deepagents leaked at import'; "
        "assert 'langgraph' not in sys.modules, 'langgraph leaked at import'; "
        "print('ok')"
    )
    result = subprocess.run(  # noqa: S603 — fixed argv, our own interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
