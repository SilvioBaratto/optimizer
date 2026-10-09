"""``stream_fund`` — the live-deliberation producer under a scripted, no-network round.

Drives :func:`fund.agents.graph.stream_fund` over the same ``_fund_fakes`` stack the
TUI tests use (``ScriptedFundModel`` + ``MemorySaver`` + ``InMemoryStore`` + in-memory
SQLite) and asserts the T1 contract: events arrive in **emission order** carrying the
agent tag + step identity, the stream ends on a single terminal ``paused`` event, the
run is flipped ``paused`` so the Decisions queue finds it, and ``import
fund.tui.stream`` drags in neither ``deepagents`` nor ``langgraph``.
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    ScriptedFundModel,
    expected_weights,
    make_constraint_set,
    make_mandate,
    seed_panel,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fund.agents.graph import stream_fund
from fund.audit import AgentRunRepository, put_constraint_set
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
