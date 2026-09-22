"""Task 8 — ``fund.tui``: the four-panel observer under Textual's headless Pilot.

Drives :class:`fund.tui.app.FundTUI` end-to-end with the network-free fakes — an
isolated in-memory SQLite engine + ``MemorySaver`` + ``InMemoryStore`` + a
``ScriptedFundModel`` — exactly the stack the resume/CLI tests use, so the cockpit
is exercised with zero live LLM/network/PG (SPEC §5). A paused ``run_fund`` gate is
seeded once per test; the App then reads it through :mod:`fund.observe`, shows the
gate, and resumes it on a worker thread via ``resume_fund``.

The event loop is never blocked: ``Approve``/``Reject`` dispatch a ``@work(thread=
True)`` worker, and the tests wait on ``app.workers.wait_for_complete()`` before
asserting the terminal DB state. ``import fund.tui.app`` is checked (in a stripped
subprocess env) to need no ``OLLAMA_API_KEY``/``DATABASE_URL`` and to drag in
neither ``deepagents`` nor ``langgraph``.
"""

from __future__ import annotations

import os
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
from textual.widgets import DataTable

from fund import observe
from fund.agents.graph import run_fund
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import settings
from fund.tui import app as tui_app
from fund.tui.app import FundTUI
from fund.tui.widgets import (
    HistoryPanel,
    HitlQueuePanel,
    PortfolioStatePanel,
    TranscriptPanel,
)

_SIZE = (120, 40)  # roomy enough that the Approve/Reject buttons are clickable


class _FakePool:
    """Stand-in for the psycopg pool; records ``main`` releasing it."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def tui_env():
    """An isolated SQLite engine + in-memory LangGraph handles with a paused run."""
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
    with get_session() as session:
        run = run_fund(
            ScriptedFundModel(universe=UNIVERSE, weights=expected),
            make_mandate(),
            portfolio_id=PORTFOLIO_ID,
            asof=ASOF,
            session=session,
            checkpointer=saver,
            store=store,
        )
        session.commit()
        run_id = run.run_id

    pool = _FakePool()
    yield SimpleNamespace(
        get_session=get_session,
        persistence=SimpleNamespace(saver=saver, store=store, pool=pool),
        model_factory=lambda: ScriptedFundModel(universe=UNIVERSE, weights=expected),
        run_id=run_id,
        expected=expected,
        saver=saver,
        store=store,
        pool=pool,
    )
    engine.dispose()


def _make_app(env, *, poll_interval: float = 30.0, model_factory=None) -> FundTUI:
    """Build the App wired onto ``env`` (a big poll interval keeps ticks out of the
    way; the initial ``call_after_refresh`` still populates)."""
    return FundTUI(
        PORTFOLIO_ID,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=model_factory or env.model_factory,
        poll_interval=poll_interval,
    )


# --- mount + poll -----------------------------------------------------------


@pytest.mark.asyncio
async def test_four_panels_mount_and_poll_populates(tui_env) -> None:
    app = _make_app(tui_env, poll_interval=0.05)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert app.query_one(TranscriptPanel) is not None
        assert app.query_one(PortfolioStatePanel) is not None
        assert app.query_one(HitlQueuePanel) is not None
        assert app.query_one(HistoryPanel) is not None
        # the poll tick populated the queue (one paused run) and the history.
        assert app.query_one("#hitl-table", DataTable).row_count == 1
        assert app.query_one("#history-table", DataTable).row_count >= 1
        # target weights (the allocator proposal fallback) fill panel 2.
        assert app.query_one("#holdings-table", DataTable).row_count >= 1


# --- selection drives the gate ----------------------------------------------


@pytest.mark.asyncio
async def test_selecting_paused_run_shows_gate(tui_env) -> None:
    app = _make_app(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app.select_run(tui_env.run_id)
        await pilot.pause()
        gate = app.query_one(HitlQueuePanel).gate_text
        assert "place_orders" in gate
        # the transcript panel filled for the selected run.
        assert app.query_one(TranscriptPanel).entry_count >= 1


@pytest.mark.asyncio
async def test_row_selection_via_datatable_drives_gate(tui_env) -> None:
    app = _make_app(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        table = app.query_one("#hitl-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert "place_orders" in app.query_one(HitlQueuePanel).gate_text


# --- approve / reject on a worker thread ------------------------------------


@pytest.mark.asyncio
async def test_approve_resumes_on_worker_and_refreshes_queue(tui_env) -> None:
    app = _make_app(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app.select_run(tui_env.run_id)
        await pilot.pause()
        await pilot.click("#approve-btn")
        await pilot.pause()  # let the button handler spawn the worker
        # resume ran on a background worker (never on the event loop).
        worker = app.last_worker
        assert worker is not None
        assert worker.group == "resume"
        await worker.wait()  # its call_from_thread UI updates have landed once done
        await pilot.pause()
        assert "approved" in app.last_status
        # the completed run left the pending queue.
        assert app.query_one("#hitl-table", DataTable).row_count == 0
    with tui_env.get_session() as session:
        row = AgentRunRepository(session).get_run(tui_env.run_id)
        assert row.status == "completed"
        assert row.weights == tui_env.expected


@pytest.mark.asyncio
async def test_reject_resumes_and_marks_rejected(tui_env) -> None:
    app = _make_app(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app.select_run(tui_env.run_id)
        await pilot.pause()
        await pilot.click("#reject-btn")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "rejected" in app.last_status
    with tui_env.get_session() as session:
        row = AgentRunRepository(session).get_run(tui_env.run_id)
        assert row.status == "rejected"
        assert row.weights == {}


@pytest.mark.asyncio
async def test_resume_without_selection_reports_and_spawns_no_worker(tui_env) -> None:
    app = _make_app(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        await pilot.press("a")  # action_approve with nothing selected
        await pilot.pause()
        assert app.last_worker is None
        assert "select a paused run" in app.last_status
        app.action_reject()  # the reject binding is guarded the same way
        assert "select a paused run" in app.last_status


@pytest.mark.asyncio
async def test_model_unavailable_surfaces_and_run_stays_paused(tui_env) -> None:
    def _no_model():
        raise RuntimeError("OLLAMA_API_KEY missing")

    app = _make_app(tui_env, model_factory=_no_model)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app.select_run(tui_env.run_id)
        await pilot.pause()
        await pilot.click("#approve-btn")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "model unavailable" in app.last_status
        assert "OLLAMA_API_KEY" in app.last_status
    with tui_env.get_session() as session:
        # nothing committed: the gate is still open for a later, keyed resume.
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "paused"


# --- observe.interrupt_for (the gate read the panel renders) -----------------


def test_interrupt_for_returns_action_requests(tui_env) -> None:
    with tui_env.get_session() as session:
        run_row = AgentRunRepository(session).get_run(tui_env.run_id)
        payload = observe.interrupt_for(tui_env.saver, run_row)
    assert payload is not None
    names = [request["name"] for request in payload["action_requests"]]
    assert "place_orders" in names


def test_interrupt_for_none_when_run_has_no_thread() -> None:
    assert observe.interrupt_for(object(), SimpleNamespace(thread_id=None)) is None


def test_interrupt_value_unwraps_bare_and_rejects_non_dict() -> None:
    from fund.observe import _interrupt_value

    assert _interrupt_value([{"action_requests": []}]) == {"action_requests": []}
    assert _interrupt_value("not-a-dict") is None


# --- main() wiring ----------------------------------------------------------


def test_main_wires_app_and_always_closes_pool(monkeypatch) -> None:
    import fund.agents.model as model_mod
    import fund.audit as audit_mod
    import fund.database as db_mod

    pool = _FakePool()
    persistence = SimpleNamespace(saver=MemorySaver(), store=InMemoryStore(), pool=pool)
    monkeypatch.setattr(audit_mod, "setup_langgraph", lambda *a, **k: persistence)
    monkeypatch.setattr(model_mod, "build_primary", lambda *a, **k: object())
    monkeypatch.setattr(db_mod, "get_session", lambda: None)
    monkeypatch.setattr(FundTUI, "run", lambda self: None)
    monkeypatch.setattr(sys, "argv", ["fund-tui", str(PORTFOLIO_ID)])

    tui_app.main()

    assert pool.closed == 1


# --- import hygiene: light + keyless -----------------------------------------


def test_import_app_is_light_and_needs_no_keys() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("OLLAMA_API_KEY", "DATABASE_URL")
    }
    code = (
        "import sys, fund.tui.app; "
        "assert 'deepagents' not in sys.modules, 'deepagents leaked at import'; "
        "assert 'langgraph' not in sys.modules, 'langgraph leaked at import'; "
        "print('ok')"
    )
    result = subprocess.run(  # noqa: S603 — fixed argv, our own interpreter
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
