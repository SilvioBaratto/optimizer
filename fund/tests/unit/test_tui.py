"""``fund.tui``: the four-panel observer under Textual's headless Pilot.

Drives :class:`fund.tui.app.FundTUI` end-to-end with the network-free fakes — an
isolated in-memory SQLite engine + ``MemorySaver`` + ``InMemoryStore`` + a
``ScriptedFundModel`` — exactly the stack the resume/CLI tests use, so the cockpit
is exercised with zero live LLM/network/PG. A paused ``run_fund`` gate is
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
import uuid
from contextlib import contextmanager, suppress
from datetime import date
from decimal import Decimal
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
from portopt_db.repositories.orders.portfolio_journal_repository import (
    PortfolioJournalRepository,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from fund import observe
from fund.agents.graph import run_fund
from fund.audit import (
    AgentRunRepository,
    HumanAuditLogRepository,
    MandateRepository,
    MifidProfileRepository,
    put_constraint_set,
)
from fund.config import settings
from fund.tui import app as tui_app
from fund.tui.app import ConfirmScreen, FundTUI, ShellApp
from fund.tui.questionnaire import QuestionnaireScreen
from fund.tui.views import (
    AuditView,
    DecisionsView,
    MandateView,
    PostureView,
    ShellView,
)
from fund.tui.widgets import (
    HistoryPanel,
    HitlQueuePanel,
    PortfolioStatePanel,
    RecentEventsPanel,
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
async def test_five_panels_mount_and_refresh_populates(tui_env) -> None:
    # Seed one portfolio overlay so the Recent Events panel has a row to paint.
    with tui_env.get_session() as session:
        PortfolioJournalRepository(session).upsert(
            PORTFOLIO_ID,
            ASOF,
            run_id=None,
            trades={},
            allocation={"AAA": 1.0},
            drift={},
            narrative="prior rebalance",
        )
        session.commit()

    app = _make_app(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        # Drive the read tick explicitly rather than racing the poll timer: the
        # old poll_interval=0.05 + single pause was flaky under load (the worker
        # tick could land after the assertions). _refresh is synchronous.
        app._refresh()
        await pilot.pause()
        assert app.query_one(TranscriptPanel) is not None
        assert app.query_one(PortfolioStatePanel) is not None
        assert app.query_one(HitlQueuePanel) is not None
        assert app.query_one(HistoryPanel) is not None
        assert app.query_one(RecentEventsPanel) is not None
        # the read tick populated the queue (one paused run) and the history.
        assert app.query_one("#hitl-table", DataTable).row_count == 1
        assert app.query_one("#history-table", DataTable).row_count >= 1
        # target weights (the allocator proposal fallback) fill panel 2.
        assert app.query_one("#holdings-table", DataTable).row_count >= 1
        # the seeded overlay fills the recent-events panel.
        assert app.query_one("#events-table", DataTable).row_count >= 1


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


def _patch_main_deps(monkeypatch) -> tuple[_FakePool, list]:
    """Stub out main()'s persistence/model/session and record ShellApp's focus.

    Returns the fake pool (to assert it was closed) and a list the patched
    ``ShellApp.run`` appends the shell's ``_portfolio_id`` to (to assert wiring).
    """
    import fund.agents.model as model_mod
    import fund.audit as audit_mod
    import fund.database as db_mod

    pool = _FakePool()
    persistence = SimpleNamespace(saver=MemorySaver(), store=InMemoryStore(), pool=pool)
    focused: list = []

    def _record_focus(self) -> None:
        focused.append(self._portfolio_id)

    monkeypatch.setattr(audit_mod, "setup_langgraph", lambda *a, **k: persistence)
    monkeypatch.setattr(model_mod, "build_primary", lambda *a, **k: object())
    monkeypatch.setattr(db_mod, "get_session", lambda: None)
    monkeypatch.setattr(ShellApp, "run", _record_focus)
    return pool, focused


def test_main_wires_app_and_always_closes_pool(monkeypatch) -> None:
    pool, focused = _patch_main_deps(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["fund-tui", str(PORTFOLIO_ID)])

    with suppress(SystemExit):  # Typer/Click exits 0 after the command returns
        tui_app.main()

    assert pool.closed == 1
    assert focused == [PORTFOLIO_ID]  # the id arg focused the shell on that portfolio


def test_main_no_arg_opens_fund_wide_shell(monkeypatch) -> None:
    pool, focused = _patch_main_deps(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["fund-tui"])  # no portfolio id

    with suppress(SystemExit):
        tui_app.main()

    assert pool.closed == 1
    assert focused == [None]  # fund-wide landing (Posture)


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


# --- ShellApp: the multi-view shell (T3) ------------------------------------

_VIEW_KEYS = ("posture", "deliberation", "decisions", "mandate", "audit", "ask")


@pytest.fixture
def shell_env():
    """An empty isolated SQLite + in-memory LangGraph handles for the shell.

    Lighter than ``tui_env`` (no seeded paused run): the shell's Phase-0 slices
    only need a mountable frame and, for the picker, a session to read ids from.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def get_session():
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    pool = _FakePool()
    yield SimpleNamespace(
        get_session=get_session,
        persistence=SimpleNamespace(
            saver=MemorySaver(), store=InMemoryStore(), pool=pool
        ),
        model_factory=lambda: None,
        pool=pool,
    )
    engine.dispose()


def _make_shell(
    env, *, portfolio_id=None, poll_interval: float = 30.0, model_factory=None
) -> ShellApp:
    """Build the shell wired onto ``env`` (a big poll interval keeps ticks away)."""
    return ShellApp(
        portfolio_id,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=model_factory or env.model_factory,
        poll_interval=poll_interval,
    )


@pytest.mark.asyncio
async def test_shell_mounts_frame_and_posture_view(shell_env) -> None:
    # Fund-wide landing (portfolio_id=None): the shell must still mount its frame.
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        # Sidebar lists all six views; the ContentSwitcher opens on Posture.
        assert len(app.query(ListItem)) == len(_VIEW_KEYS)
        content = app.query_one("#content", ContentSwitcher)
        assert content.current == "view-posture"
        posture = app.query_one("#view-posture", ShellView)
        assert posture.view_title == "Posture"
        # The synchronous per-view refresh entry point drives the active view.
        app._refresh()
        assert posture.render_count >= 1


def test_shell_accepts_optional_portfolio_id() -> None:
    # The fund-wide landing seam: __init__ must accept portfolio_id=None without a
    # session (the launcher opens the shell before any portfolio is focused).
    shell = ShellApp(
        None,
        session_factory=lambda: None,
        persistence=SimpleNamespace(saver=None, store=None),
        model_factory=lambda: None,
    )
    assert shell._portfolio_id is None


@pytest.mark.asyncio
async def test_shell_view_context_seam_reads_collaborators(shell_env) -> None:
    # The view-context seam (T2): a ShellView reads the App's injected collaborators
    # through the narrowed `shell` accessor and opens a short read session.
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        view = app.query_one("#view-posture", ShellView)
        assert view.shell is app
        assert view.shell._persistence is app._persistence
        assert view.shell._portfolio_id == app._portfolio_id
        with view.read_session() as session:
            # Empty fund → []; proves the seam opened a usable read session.
            assert observe.list_portfolios(session) == []


@pytest.mark.asyncio
async def test_shell_sidebar_navigates_all_six_views(shell_env) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        sidebar = app.query_one("#sidebar", ListView)
        content = app.query_one("#content", ContentSwitcher)
        sidebar.focus()
        # Selecting each sidebar entry switches the content pane to that view and
        # repaints it (render_count proves the per-view refresh ran).
        for i, key in enumerate(_VIEW_KEYS):
            sidebar.index = i
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert content.current == f"view-{key}", f"nav to {key} did not switch"
            assert app.query_one(f"#view-{key}", ShellView).render_count >= 1


# A letter-bearing UUID avoids SQLite's numeric-affinity coercion of all-digit ids.
_PID_A = uuid.UUID("aaaa0001-0000-4000-8000-000000000001")
_PID_B = uuid.UUID("bbbb0002-0000-4000-8000-000000000002")


@pytest.mark.asyncio
async def test_shell_picker_lists_portfolios_and_selection_refocuses(shell_env) -> None:
    # Two portfolios surface via their runs → observe.list_portfolios returns both.
    with shell_env.get_session() as session:
        repo = AgentRunRepository(session)
        for seed, pid in enumerate((_PID_A, _PID_B), start=1):
            repo.create_run(
                portfolio_id=pid,
                asof=date(2026, 1, 2),
                seed=seed,
                universe=[],
                optimizer_config={},
            )
        session.commit()

    app = _make_shell(shell_env)  # fund-wide landing (no focus yet)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        # The picker lists the real portfolio ids.
        assert set(app._portfolio_ids) == {_PID_A, _PID_B}
        # Picking one re-focuses the shell and reflects that portfolio's run state
        # (a freshly created run is pending → running glyph).
        app.query_one("#hdr-portfolio", Select).value = str(_PID_A)
        await pilot.pause()
        assert app._portfolio_id == _PID_A
        assert "running" in app._runstate


@pytest.mark.asyncio
async def test_shell_focused_launch_preselects_without_spurious_refocus(
    shell_env, monkeypatch
) -> None:
    # Both ids surface in the picker via runs; the launch focuses _PID_A.
    with shell_env.get_session() as session:
        repo = AgentRunRepository(session)
        for seed, pid in enumerate((_PID_A, _PID_B), start=1):
            repo.create_run(
                portfolio_id=pid,
                asof=date(2026, 1, 2),
                seed=seed,
                universe=[],
                optimizer_config={},
            )
        session.commit()

    # Count real (past-the-guard) focus changes: on_mount paints the glyph once; the
    # mount-time Select.Changed echo must NOT paint it a second time. A mount-window
    # flag can't suppress that echo (Textual dispatches it after on_mount returns, by
    # which point the flag is already cleared), so only the value-equality guard does.
    focus_paints = []
    original = ShellApp._refresh_runstate

    def _counting(self) -> None:
        focus_paints.append(self._portfolio_id)
        original(self)

    monkeypatch.setattr(ShellApp, "_refresh_runstate", _counting)

    app = _make_shell(shell_env, portfolio_id=_PID_A)  # launched focused
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()  # drain the mount-time Select.Changed the seeding echoes
        # Focus is preserved end-to-end and pre-selected in the picker.
        assert app._portfolio_id == _PID_A
        assert app.query_one("#hdr-portfolio", Select).value == str(_PID_A)
        # The echo was dropped: exactly one glyph paint (on_mount), no spurious refocus.
        assert focus_paints == [_PID_A]
        # A genuine pick still moves focus, proving the guard only drops no-ops rather
        # than wedging the picker.
        app.query_one("#hdr-portfolio", Select).value = str(_PID_B)
        await pilot.pause()
        assert app._portfolio_id == _PID_B
        assert focus_paints == [_PID_A, _PID_B]


@pytest.mark.asyncio
async def test_shell_blank_pick_returns_fund_wide_without_crashing(shell_env) -> None:
    # Picking the blank "⟨fund-wide⟩" row must return to fund-wide (portfolio_id=None),
    # NOT crash. Its value is the NoSelection sentinel Select.NULL; the guard must test
    # Select.NULL, not Select.BLANK (== Widget.BLANK == False, which never matches and
    # would send the blank value down uuid.UUID("Select.NULL") → ValueError).
    with shell_env.get_session() as session:
        AgentRunRepository(session).create_run(
            portfolio_id=_PID_A,
            asof=date(2026, 1, 2),
            seed=1,
            universe=[],
            optimizer_config={},
        )
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)  # launched focused
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert app._portfolio_id == _PID_A
        # Pick the blank row (its value is the NoSelection sentinel).
        app.query_one("#hdr-portfolio", Select).value = Select.NULL
        await pilot.pause()
        assert app._portfolio_id is None  # fund-wide again, no ValueError
        assert "idle" in app._runstate  # fund-wide reads as idle


@pytest.mark.asyncio
async def test_shell_collapses_sidebar_to_rail_when_narrow(shell_env) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=(60, 40)) as pilot:  # below the rail threshold
        await pilot.pause()
        assert app._narrow is True
        assert app.query_one("#sidebar", ListView).has_class("narrow")
        # The labels collapse to a single-letter rail.
        label = app.query_one("#nav-posture", ListItem).query_one(Label)
        assert str(label.render()) == "P"


@pytest.mark.asyncio
async def test_shell_keeps_full_sidebar_when_wide(shell_env) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:  # (120, 40) — wide
        await pilot.pause()
        assert app._narrow is False
        assert not app.query_one("#sidebar", ListView).has_class("narrow")
        label = app.query_one("#nav-posture", ListItem).query_one(Label)
        assert str(label.render()) == "Posture"


# --- Decisions view: the cross-portfolio HITL inbox (T3) ---------------------


async def _open_decisions(app, pilot) -> DecisionsView:
    """Switch the shell to the Decisions view and paint its queue."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-decisions"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-decisions", DecisionsView)


@pytest.mark.asyncio
async def test_decisions_queue_is_cross_portfolio_and_gate_shows_why(tui_env) -> None:
    # Fund-wide landing (portfolio_id=None): the queue still lists the paused gate,
    # proving the read is cross-portfolio (pending_hitl(..., None)).
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        assert view.query_one("#decisions-table", DataTable).row_count == 1
        view.select_run(tui_env.run_id)
        await pilot.pause()
        gate = view.gate_text
        assert "place_orders" in gate  # the reviewable gate
        assert "why" in gate  # full gate context + "why"


@pytest.mark.asyncio
async def test_decisions_row_selection_via_datatable_shows_gate(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        table = view.query_one("#decisions-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert view.selected_run_id == tui_env.run_id
        assert "place_orders" in view.gate_text


@pytest.mark.asyncio
async def test_decisions_approve_resumes_on_worker_and_refreshes(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        assert worker.group == "action"
        await worker.wait()
        await pilot.pause()
        assert "approved" in app.last_status
        # the completed run left the cross-portfolio queue.
        assert view.query_one("#decisions-table", DataTable).row_count == 0
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "completed"
        assert run.weights == tui_env.expected
        rows = HumanAuditLogRepository(session).list_recent(action="approve")
        assert len(rows) == 1
        assert rows[0].run_id == tui_env.run_id
        assert rows[0].portfolio_id == PORTFOLIO_ID


@pytest.mark.asyncio
async def test_decisions_reject_marks_rejected_and_logs(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_reject()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "rejected" in app.last_status
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "rejected"
        assert run.weights == {}
        rows = HumanAuditLogRepository(session).list_recent(action="reject")
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_decisions_approve_without_selection_reports_no_worker(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_decisions(app, pilot)
        app.action_decide_approve()  # nothing selected
        await pilot.pause()
        assert app.last_worker is None
        assert "select a paused" in app.last_status


@pytest.mark.asyncio
async def test_decisions_queue_ignores_header_focus(tui_env) -> None:
    # Focus the shell on a DIFFERENT portfolio than the one owning the paused run:
    # the queue must still list it (fund-wide pending_hitl(..., None)). This fails if
    # the view scoped the read to self.shell._portfolio_id instead of None.
    assert _PID_B != PORTFOLIO_ID
    app = _make_shell(tui_env, portfolio_id=_PID_B)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        assert app._portfolio_id == _PID_B  # the shell is focused elsewhere
        assert view.query_one("#decisions-table", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_decisions_actions_inert_when_view_not_active(tui_env) -> None:
    # A real selection exists, but the active pane is not Decisions: the active-view
    # guard in _active_decisions_view must block the worker from firing.
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.query_one(ContentSwitcher).current = "view-posture"  # navigate away
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        assert app.last_worker is None
        assert "select a paused" in app.last_status


@pytest.mark.asyncio
async def test_decisions_model_unavailable_surfaces_and_run_stays_paused(
    tui_env,
) -> None:
    def _no_model():
        raise RuntimeError("OLLAMA_API_KEY missing")

    app = _make_shell(tui_env, model_factory=_no_model)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
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


@pytest.mark.asyncio
async def test_decisions_action_failure_surfaces_and_run_stays_paused(
    tui_env, monkeypatch
) -> None:
    import fund.actions as actions_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("resume exploded")

    # The worker lazy-imports apply_action from the module, so patching the attribute
    # is picked up at call time.
    monkeypatch.setattr(actions_mod, "apply_action", _boom)

    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "action failed" in app.last_status
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "paused"


@pytest.mark.asyncio
async def test_decisions_killswitch_confirm_finalizes_rejected_and_logs(
    tui_env,
) -> None:
    # Confirm the explicit-consequence dialog: the paused run finalises rejected and
    # logs a *distinct* killswitch row (not a plain reject), leaving the queue.
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_killswitch()
        await pilot.pause()  # let the confirm modal mount on top
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        assert worker.group == "action"
        await worker.wait()
        await pilot.pause()
        assert "killswitch" in app.last_status
        assert view.query_one("#decisions-table", DataTable).row_count == 0
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "rejected"
        assert run.weights == {}
        # logged distinctly: a killswitch row, never a plain reject row.
        assert HumanAuditLogRepository(session).list_recent(action="reject") == []
        rows = HumanAuditLogRepository(session).list_recent(action="killswitch")
        assert len(rows) == 1
        assert rows[0].run_id == tui_env.run_id
        assert rows[0].portfolio_id == PORTFOLIO_ID


@pytest.mark.asyncio
async def test_decisions_killswitch_cancel_is_a_noop_and_run_stays_paused(
    tui_env,
) -> None:
    # Cancel the confirm (escape): no worker fires and the gate stays paused + queued.
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_killswitch()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app.last_worker is None
        assert "cancelled" in app.last_status
        assert view.query_one("#decisions-table", DataTable).row_count == 1
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "paused"


@pytest.mark.asyncio
async def test_decisions_killswitch_without_selection_shows_no_confirm(tui_env) -> None:
    # Guard: nothing selected => no modal, no worker, just the status hint.
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_decisions(app, pilot)
        app.action_decide_killswitch()
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmScreen)
        assert app.last_worker is None
        assert "select a paused" in app.last_status


def test_format_decision_gate_renders_why_from_description_else_fallback() -> None:
    from fund.tui.views import _format_decision_gate

    run = SimpleNamespace(id=_PID_A, portfolio_id=_PID_B)
    with_desc = _format_decision_gate(
        {
            "action_requests": [
                {
                    "name": "place_orders",
                    "description": "trimmed energy on macro risk-off",
                    "args": {"weights": {"AAA": 0.5}},
                }
            ]
        },
        run,
    )
    assert "place_orders" in with_desc
    assert "why: trimmed energy on macro risk-off" in with_desc
    assert "AAA: 0.5000" in with_desc

    no_desc = _format_decision_gate(
        {"action_requests": [{"name": "place_orders", "args": {}}]}, run
    )
    assert "why: review before it executes" in no_desc


# --- Mandate view: run-policy editor + guardrail override (T5) ----------------


def _seed_mandate_and_profile(env, *, band: str = "balanced") -> None:
    """Seed one portfolio's run-policy mandate + an active MiFID profile."""
    with env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        MifidProfileRepository(session).add_version(
            portfolio_id=PORTFOLIO_ID,
            questionnaire={},
            constraint_set=make_constraint_set().model_dump(mode="json"),
            suitability={"band": band},
            store_key=settings.constraint_set_store_key,
        )
        session.commit()


async def _open_mandate(app, pilot) -> MandateView:
    """Switch the shell to the Mandate view and paint it."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-mandate"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-mandate", MandateView)


@pytest.mark.asyncio
async def test_mandate_renders_run_policy_and_guardrails(shell_env) -> None:
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        assert view.query_one("#mandate-capital", Input).value == "100000"
        assert view.query_one("#mandate-drift", Input).value == "0.1"
        assert view.suitability_band == "balanced"
        assert view.query_one("#mandate-band", Select).value == "balanced"
        caps = str(view.query_one("#mandate-caps", Static).render())
        assert "ESG exclude" in caps


@pytest.mark.asyncio
async def test_mandate_save_run_policy_persists_and_logs(shell_env) -> None:
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-drift", Input).value = "0.25"
        app.action_save_mandate()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "mandate updated" in app.last_status
    with shell_env.get_session() as session:
        reloaded = observe.get_mandate(session, PORTFOLIO_ID)
        assert reloaded.drift_l1_threshold == 0.25  # upsert persisted the edit
        rows = HumanAuditLogRepository(session).list_recent(action="mandate_edit")
        assert len(rows) == 1
        assert rows[0].override_flag is False
        assert rows[0].suitability_flag is False


@pytest.mark.asyncio
async def test_mandate_guardrail_override_out_of_band_flags_and_logs(shell_env) -> None:
    # Suitability ceiling is balanced; proposing aggressive sits out of band (flag it).
    _seed_mandate_and_profile(shell_env, band="balanced")
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-band", Select).value = "aggressive"
        app.action_override_guardrail()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "⚠" in app.last_status
    with shell_env.get_session() as session:
        rows = HumanAuditLogRepository(session).list_recent(action="guardrail_override")
        assert len(rows) == 1
        assert rows[0].prev_value == "balanced"
        assert rows[0].new_value == "aggressive"
        assert rows[0].override_flag is True
        assert rows[0].suitability_flag is True
        # audit-only: the guardrail edit did not touch the persisted mandate.
        assert observe.get_mandate(session, PORTFOLIO_ID).drift_l1_threshold == 0.1


@pytest.mark.asyncio
async def test_mandate_guardrail_in_band_logs_without_flag(shell_env) -> None:
    # Proposing a more conservative band than the ceiling is always suitable.
    _seed_mandate_and_profile(shell_env, band="balanced")
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-band", Select).value = "conservative"
        app.action_override_guardrail()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
    with shell_env.get_session() as session:
        rows = HumanAuditLogRepository(session).list_recent(action="guardrail_override")
        assert len(rows) == 1
        assert rows[0].override_flag is False
        assert rows[0].suitability_flag is False


@pytest.mark.asyncio
async def test_mandate_actions_inert_without_a_mandate(shell_env) -> None:
    # Fund-wide landing, nothing seeded: both mandate actions no-op with a hint.
    app = _make_shell(shell_env)  # fund-wide landing, no portfolio focused
    async with app.run_test(size=_SIZE) as pilot:
        await _open_mandate(app, pilot)
        app.action_save_mandate()
        await pilot.pause()
        assert app.last_worker is None
        assert "focus a portfolio" in app.last_status
        app.action_override_guardrail()
        await pilot.pause()
        assert app.last_worker is None
        assert "suitability band" in app.last_status


# --- Posture view: fund roster + fund strip + focused detail (T6) -------------

_PID_C = uuid.UUID("cccc0003-0000-4000-8000-000000000003")


async def _open_posture(app, pilot) -> PostureView:
    """Switch the shell to the Posture view and paint its roster + detail."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-posture"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-posture", PostureView)


def _seed_run(session, pid, *, seed: int, status: str | None = None):
    """Create one run for ``pid``; optionally flip it to a terminal/paused status."""
    repo = AgentRunRepository(session)
    run = repo.create_run(
        portfolio_id=pid,
        asof=date(2026, 1, 2),
        seed=seed,
        universe=[],
        optimizer_config={},
    )
    if status == "paused":
        repo.mark_paused(run.id)
    return run


def test_posture_health_tiers() -> None:
    from fund.tui.views import _posture_health

    # In-band drift → ok; at/above the mandate threshold → warn; a failed/blocked
    # latest run → breach (dot + label, so two tiers sharing ● still read apart).
    assert _posture_health(0.05, 0.1, "completed") == ("●", "ok")
    assert _posture_health(0.2, 0.1, "completed") == ("◐", "warn")
    assert _posture_health(0.0, None, "incomplete") == ("●", "breach")
    assert _posture_health(0.0, 0.1, "running") == ("●", "ok")


@pytest.mark.asyncio
async def test_posture_roster_lists_portfolios_and_aggregates(shell_env) -> None:
    # Two portfolios surface via their runs: one pending (running), one paused
    # (awaiting). The roster lists both; the fund strip aggregates the counts.
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)  # pending → running
        _seed_run(session, _PID_B, seed=2, status="paused")  # awaiting
        session.commit()

    app = _make_shell(shell_env)  # fund-wide landing
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert view.query_one("#posture-roster", DataTable).row_count == 2
        strip = str(view.query_one("#posture-strip", Static).render())
        assert "2 pf" in strip
        assert "1 awaiting" in strip
        assert "1 running" in strip


@pytest.mark.asyncio
async def test_posture_empty_portfolio_shows_profile_hint(shell_env) -> None:
    # A portfolio with a mandate but no runs: roster lists it; focusing shows the
    # "no runs yet · p profile to start" empty-state in the detail pane.
    with shell_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate(_PID_C))
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_C)  # launched focused
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert view.query_one("#posture-roster", DataTable).row_count == 1
        assert "no runs yet" in view.detail_text
        assert "p profile to start" in view.detail_text


@pytest.mark.asyncio
async def test_posture_roster_selection_focuses_and_reveals_detail(shell_env) -> None:
    # Selecting a roster row sets the header context (focuses that portfolio) and
    # reveals its detail — proving the launchpad wiring through the shell seam.
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)
        session.commit()

    app = _make_shell(shell_env)  # fund-wide landing (nothing focused yet)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert app._portfolio_id is None
        table = view.query_one("#posture-roster", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._portfolio_id == _PID_A  # header context set from the roster
        assert "ACTIVE RUN" in view.detail_text  # detail revealed for the focus


# --- Onboarding + drive flows: questionnaire → profiler → run → approve (T7) ---

_PID_NEW = uuid.UUID("dddd0004-0000-4000-8000-000000000004")


def _fill_questionnaire(screen, answers, *, capital: str) -> None:
    """Populate the questionnaire widgets from a ``MiFIDAnswers`` (+ capital)."""
    screen.query_one("#q-capital", Input).value = capital
    screen.query_one("#q-base-currency", Input).value = answers.base_currency
    screen.query_one("#q-knowledge", Select).value = answers.knowledge.level.value
    screen.query_one("#q-max-loss", Input).value = str(
        answers.capacity.max_1yr_loss_pct
    )
    screen.query_one("#q-buffer", Input).value = str(answers.capacity.buffer_months)
    screen.query_one("#q-goal", Select).value = answers.objectives.goal.value
    screen.query_one("#q-horizon", Select).value = answers.objectives.horizon.value
    screen.query_one("#q-likert", Input).value = ",".join(
        str(item) for item in answers.objectives.likert_items
    )
    screen.query_one(
        "#q-loss-reaction", Select
    ).value = answers.objectives.loss_reaction.value


@pytest.fixture
def onboard_env():
    """A seeded price panel + shared LangGraph handles for the full onboard cycle.

    Unlike ``tui_env`` nothing is pre-seeded beyond prices: the new portfolio, its
    mandate, and its active ``ConstraintSet`` are all created *through the TUI* (the
    profiler's approve caches the set the subsequent run resolves).
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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

    from _fund_fakes import seed_panel

    with get_session() as session:
        seed_panel(session)
        session.commit()
    with get_session() as session:
        expected = expected_weights(session)

    pool = _FakePool()
    yield SimpleNamespace(
        get_session=get_session,
        persistence=SimpleNamespace(saver=saver, store=store, pool=pool),
        model_factory=lambda: None,
        expected=expected,
        saver=saver,
        store=store,
        pool=pool,
    )
    engine.dispose()


@pytest.mark.asyncio
async def test_questionnaire_builds_typed_answers(shell_env) -> None:
    from _profiler_fakes import make_answers

    answers = make_answers()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        screen = QuestionnaireScreen(ask_capital=True)
        app.push_screen(screen)
        await pilot.pause()
        _fill_questionnaire(screen, answers, capital="100000")
        submission = screen.build_submission()
    assert submission.capital == Decimal("100000")
    assert submission.answers.base_currency == "EUR"
    assert submission.answers.objectives.goal == answers.objectives.goal
    assert submission.answers.objectives.likert_items == answers.objectives.likert_items


@pytest.mark.asyncio
async def test_drive_actions_inert_off_posture_or_without_focus(shell_env) -> None:
    app = _make_shell(shell_env)  # fund-wide landing
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        # Posture active but nothing focused → run/profile report a hint, no worker.
        app.action_run_portfolio()
        await pilot.pause()
        assert app.last_worker is None
        assert "focus a portfolio" in app.last_status
        # Off Posture → the drive launchpad is inert.
        app.query_one(ContentSwitcher).current = "view-decisions"
        await pilot.pause()
        app.action_new_portfolio()
        await pilot.pause()
        assert not isinstance(app.screen, QuestionnaireScreen)
        assert "open Posture" in app.last_status


@pytest.mark.asyncio
async def test_onboarding_cancel_is_a_noop(onboard_env) -> None:
    app = _make_shell(onboard_env)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        app.action_new_portfolio()
        await pilot.pause()
        assert isinstance(app.screen, QuestionnaireScreen)
        await pilot.press("escape")  # cancel the questionnaire
        await pilot.pause()
        assert app.last_worker is None
        assert "cancelled" in app.last_status
    with onboard_env.get_session() as session:
        # nothing created: no mandate, no runs.
        assert observe.list_portfolios(session) == []


@pytest.mark.asyncio
async def test_onboarding_full_cycle_profile_run_approve(
    onboard_env, monkeypatch
) -> None:
    # The phase-gate path: a new portfolio is created, profiled, approved, run, and
    # the rebalance approved — entirely from the TUI, no CLI / docker exec.
    from _profiler_fakes import (
        ScriptedProfilerModel,
        final_reply,
        make_answers,
        save_tool_call,
    )

    answers = make_answers()  # lands in the Balanced band, no suitability flags
    monkeypatch.setattr(tui_app.uuid, "uuid4", lambda: _PID_NEW)

    # Phase-1 model drives run_profiler to the save_profile gate.
    app = _make_shell(
        onboard_env,
        model_factory=lambda: ScriptedProfilerModel(
            structured_outcomes=[answers],
            chat_responses=[save_tool_call(str(_PID_NEW)), final_reply()],
        ),
    )
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)

        # n → questionnaire → submit → profiler drive worker.
        app.action_new_portfolio()
        await pilot.pause()
        assert isinstance(app.screen, QuestionnaireScreen)
        _fill_questionnaire(app.screen, answers, capital="100000")
        app.screen._submit()
        await pilot.pause()
        assert app.last_worker is not None
        await app.last_worker.wait()
        await pilot.pause()
        assert "awaiting approval" in app.last_status

        # The new portfolio now carries a paused profiler gate.
        with onboard_env.get_session() as session:
            paused = AgentRunRepository(session).list_paused_runs(_PID_NEW)
            assert len(paused) == 1
            profiler_run_id = paused[0].id

        # Approve the profiler gate (resume model = just the closing turn).
        app._model_factory = lambda: ScriptedProfilerModel(
            chat_responses=[final_reply(), final_reply()]
        )
        decisions = await _open_decisions(app, pilot)
        decisions.select_run(profiler_run_id)
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert "approved" in app.last_status

        # u → drive one rebalance for the focused new portfolio to the gate.
        app._model_factory = lambda: ScriptedFundModel(
            universe=UNIVERSE, weights=onboard_env.expected
        )
        await _open_posture(app, pilot)
        app.focus_portfolio(_PID_NEW)
        await pilot.pause()
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert "awaiting approval" in app.last_status

        with onboard_env.get_session() as session:
            paused = AgentRunRepository(session).list_paused_runs(_PID_NEW)
            assert len(paused) == 1
            fund_run_id = paused[0].id

        # Approve the rebalance gate (fund model is role-aware; a fresh one resumes).
        decisions = await _open_decisions(app, pilot)
        decisions.select_run(fund_run_id)
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert "approved" in app.last_status

    with onboard_env.get_session() as session:
        run = AgentRunRepository(session).get_run(fund_run_id)
        assert run.status == "completed"
        assert run.weights == onboard_env.expected  # optimizer output, not the LLM's
        actions = [row.action for row in HumanAuditLogRepository(session).list_recent()]
        assert "new_portfolio" in actions
        assert "profile" in actions
        assert "run" in actions
        assert actions.count("approve") == 2  # both gates approved through the path


# --- Audit view: fund-wide filterable timeline (agents ∪ humans) (T8) ---------


async def _open_audit(app, pilot) -> AuditView:
    """Switch the shell to the Audit view and paint its timeline."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-audit"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-audit", AuditView)


def _seed_audit_rows(session) -> None:
    """Seed one agent decision + two human rows (an approve + a flagged override)."""
    repo = AgentRunRepository(session)
    run = repo.create_run(
        portfolio_id=_PID_A,
        asof=date(2026, 1, 2),
        seed=1,
        universe=[],
        optimizer_config={},
    )
    repo.append_decision(
        run.id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response='{"weights": {"AAA": 1.0}}',
    )
    hlog = HumanAuditLogRepository(session)
    hlog.append(action="approve", portfolio_id=_PID_A, run_id=run.id, actor="human")
    hlog.append(
        action="guardrail_override",
        portfolio_id=_PID_B,
        actor="human",
        target="risk_band",
        prev_value="balanced",
        new_value="aggressive",
        source="mifid_profile",
        override_flag=True,
        suitability_flag=True,
        idempotency_key="idem-xyz",
    )


@pytest.mark.asyncio
async def test_audit_unions_agent_and_human_rows(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_audit_rows(session)
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        # 1 agent decision + 2 human rows merge into one timeline.
        assert view.query_one("#audit-table", DataTable).row_count == 3


@pytest.mark.asyncio
async def test_audit_filters_compose(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_audit_rows(session)
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        table = view.query_one("#audit-table", DataTable)
        view.cycle_kind()  # all → agent
        assert view.filter_kind == "agent"
        assert table.row_count == 1
        view.cycle_kind()  # agent → human
        assert view.filter_kind == "human"
        assert table.row_count == 2
        view.cycle_kind()  # human → all
        view.toggle_overrides()  # ⚑ only → the single flagged override row
        assert table.row_count == 1


@pytest.mark.asyncio
async def test_audit_portfolio_scope_filters_to_focus(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_audit_rows(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)  # focus _PID_A
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        table = view.query_one("#audit-table", DataTable)
        assert table.row_count == 3  # fund-wide
        view.toggle_portfolio_scope()  # scope to the focused _PID_A
        assert view.filter_portfolio is True
        # _PID_A owns the agent decision + the approve row; the override is _PID_B.
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_audit_detail_shows_provenance_and_flags(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_audit_rows(session)
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        view.cycle_kind()
        view.cycle_kind()  # → human
        view.toggle_overrides()  # isolate the flagged override row
        view.select_index(0)
        assert "⚑ override" in view.detail_text
        assert "aggressive" in view.detail_text  # new_value provenance
        assert "idem-xyz" in view.detail_text  # idempotency key
        assert "immutable" in view.detail_text


@pytest.mark.asyncio
async def test_audit_export_writes_csv(shell_env, tmp_path) -> None:
    with shell_env.get_session() as session:
        _seed_audit_rows(session)
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        out = tmp_path / "audit.csv"
        written = view.export_csv(str(out))
    assert written == str(out)
    text = out.read_text(encoding="utf-8")
    assert "when,kind,actor,action" in text  # header row
    assert "guardrail_override" in text
    assert "optimize_portfolio" in text  # the agent decision exported too
    assert text.strip().count("\n") >= 3  # header + 3 data rows
