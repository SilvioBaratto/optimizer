"""``fund.tui``: the ShellApp oversight cockpit under Textual's headless Pilot.

Exercises the unified ``apply_action`` path, the notification bus, and the full
onboarding→profile→run→approve cycle against the network-free stack — an isolated
in-memory SQLite engine + ``MemorySaver`` + ``InMemoryStore`` + a scripted model —
so zero live LLM/network/PG is required.

Covered views: Posture / Decisions / Mandate / Audit / Deliberation.  ``import
fund.tui.app`` is checked (in a stripped subprocess env) to need no
``OLLAMA_API_KEY``/``DATABASE_URL`` and to drag in neither ``deepagents`` nor
``langgraph``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

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
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import Base, PaperOrder
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from textual.coordinate import Coordinate
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Checkbox,
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
from fund.tui.app import (
    ConfirmScreen,
    HelpScreen,
    OrderEditScreen,
    RationaleConfirmScreen,
    ScopePickScreen,
    ShellApp,
)
from fund.tui.ask import ResolvedAction
from fund.tui.barge_in import BargeInResult, BargeInScreen
from fund.tui.questionnaire import QuestionnaireScreen
from fund.tui.stream import StepStatus, StreamEvent
from fund.tui.views import (
    AskView,
    AuditView,
    DailyBriefView,
    DecisionsView,
    DeliberationView,
    MandateView,
    PerformanceView,
    PostureView,
    ShellView,
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
        # The Performance/Ask views read through fund.observe, which lazy-imports the
        # optimizer/skfolio tracker inside its body — so a bare import must drag in
        # neither (they are allowed in fund, but keep fund.tui light).
        "assert 'optimizer' not in sys.modules, 'optimizer leaked at import'; "
        "assert 'skfolio' not in sys.modules, 'skfolio leaked at import'; "
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


def test_thread_workers_take_their_view_as_a_param() -> None:
    """Thread workers must receive their view, never query the DOM on the worker thread.

    The Ask/stream/readout workers run on ``@work(thread=True)`` and touch their view
    only through ``call_from_thread``. The view itself is resolved by the caller on the
    event loop and passed in — a refactor that moved the ``query_one`` resolution back
    inside a worker body would reintroduce a DOM-off-the-event-loop race, so this guards
    the contract at the signature level (``inspect.signature`` sees through ``@work``).
    """
    import inspect

    for name in ("_ask_worker", "_stream_worker", "_recompute_readout_worker"):
        params = inspect.signature(getattr(tui_app.ShellApp, name)).parameters
        assert "view" in params, name


# --- ShellApp: the multi-view shell (T3) ------------------------------------

_VIEW_KEYS = (
    "posture",
    "deliberation",
    "decisions",
    "mandate",
    "audit",
    "performance",
    "ask",
    "brief",
)


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
        # Sidebar lists every view; the ContentSwitcher opens on Posture.
        assert len(app.query(ListItem)) == len(_VIEW_KEYS)
        content = app.query_one("#content", ContentSwitcher)
        assert content.current == "view-posture"
        posture = app.query_one("#view-posture", ShellView)
        assert posture.view_title == "Posture"
        # The synchronous per-view refresh entry point drives the active view.
        app._refresh()
        assert posture.render_count >= 1


@pytest.mark.asyncio
async def test_poll_tick_after_teardown_is_a_noop(shell_env) -> None:
    """A timer tick racing the shutdown finds the DOM gone and must not raise.

    The headless harness re-raises an app-level exception on exit, so an unguarded
    tick made every fast-polling test flaky on a loaded machine.
    """
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
    app._poll()


@pytest.mark.asyncio
async def test_poll_tick_on_a_live_app_does_not_hide_a_missing_widget(
    shell_env, monkeypatch
) -> None:
    """Only the shutdown race is silent: a ``NoMatches`` from a view's repaint while
    the app is running is a real bug and must surface rather than leave the view
    stale on every tick."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        posture = app.query_one("#view-posture", ShellView)

        def renamed_widget() -> None:
            raise NoMatches("#renamed-widget")

        monkeypatch.setattr(posture, "refresh_view", renamed_widget)
        with pytest.raises(NoMatches):
            app._poll()


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
async def test_shell_sidebar_navigates_all_views(shell_env) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        sidebar = app.query_one("#sidebar", ListView)
        content = app.query_one("#content", ContentSwitcher)
        # Selecting each sidebar entry switches the content pane to that view and
        # repaints it (render_count proves the per-view refresh ran). Re-focus the
        # sidebar each step: a focusable view (Ask / Daily brief) grabs focus on
        # selection, so navigation must return focus to the rail before the next pick.
        for i, key in enumerate(_VIEW_KEYS):
            sidebar.focus()
            sidebar.index = i
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert content.current == f"view-{key}", f"nav to {key} did not switch"
            assert app.query_one(f"#view-{key}", ShellView).render_count >= 1


# A letter-bearing UUID avoids SQLite's numeric-affinity coercion of all-digit ids.
_PID_A = uuid.UUID("aaaa0001-0000-4000-8000-000000000001")
_PID_B = uuid.UUID("bbbb0002-0000-4000-8000-000000000002")
_BRIEF_RUN_ID = uuid.UUID("dddd0004-0000-4000-8000-000000000004")


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


def _seed_two_portfolio_runs(env) -> None:
    with env.get_session() as session:
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


@pytest.mark.asyncio
async def test_picker_reseed_keeps_focus_without_a_fund_wide_bounce(
    shell_env, monkeypatch
) -> None:
    # Re-seeding the options resets the Select to NULL; that intermediate NULL must not
    # reach on_select_changed, or the shell repaints fund-wide before snapping back.
    _seed_two_portfolio_runs(shell_env)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        seen = []
        original = ShellApp._refresh_runstate

        def _spy(self) -> None:
            seen.append(self._portfolio_id)
            original(self)

        monkeypatch.setattr(ShellApp, "_refresh_runstate", _spy)
        app._populate_picker()
        await pilot.pause()
        assert seen == []  # no focus change ever surfaced
        assert app._portfolio_id == _PID_A
        assert app.query_one("#hdr-portfolio", Select).value == str(_PID_A)


@pytest.mark.asyncio
async def test_picker_reseed_read_error_keeps_options_and_focus(
    shell_env, monkeypatch
) -> None:
    # A transient roster read error must not empty the picker nor drop the focus.
    _seed_two_portfolio_runs(shell_env)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert set(app._portfolio_ids) == {_PID_A, _PID_B}

        def _boom(session):
            raise ConnectionError("pool timeout")

        monkeypatch.setattr(observe, "list_portfolios", _boom)
        app._populate_picker()
        await pilot.pause()
        assert app._portfolio_id == _PID_A
        assert app.query_one("#hdr-portfolio", Select).value == str(_PID_A)
        assert set(app._portfolio_ids) == {_PID_A, _PID_B}


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


def _screen_lines(app: ShellApp) -> list[str]:
    """The text the compositor actually paints, one string per terminal row."""
    return [strip.text for strip in app.screen._compositor.render_strips()]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 30), (80, 24)])
async def test_status_message_is_painted_above_the_footer(shell_env, size) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._set_status("ACTION FAILED: boom")
        await pilot.pause()
        rows = _screen_lines(app)
        assert any("ACTION FAILED: boom" in row for row in rows)
        status_row = next(i for i, row in enumerate(rows) if "ACTION FAILED" in row)
        footer = app.query_one("Footer")
        assert status_row < footer.region.y  # never painted under the key hints


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 30), (80, 24)])
async def test_header_portfolio_picker_label_is_painted(shell_env, size) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        picker = app.query_one("#hdr-portfolio", Select)
        header = app.query_one("#shell-header")
        assert picker.region.height == header.region.height == 1
        # The prompt must sit on the header row itself, not on a row the body covers.
        assert "fund-wide" in _screen_lines(app)[header.region.y]
        # The run-state glyphs share that row and must not be pushed off it.
        assert "ARMED" in _screen_lines(app)[header.region.y]


def _seed_portfolio_run(env, pid, *, seed: int = 1) -> None:
    with env.get_session() as session:
        AgentRunRepository(session).create_run(
            portfolio_id=pid,
            asof=date(2026, 1, 2),
            seed=seed,
            universe=[],
            optimizer_config={},
        )
        session.commit()


@pytest.mark.asyncio
async def test_focusing_a_portfolio_created_after_launch_updates_the_header_picker(
    shell_env,
) -> None:
    _seed_portfolio_run(shell_env, _PID_A)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        picker = app.query_one("#hdr-portfolio", Select)
        assert picker.value == str(_PID_A)

        _seed_portfolio_run(shell_env, _PID_B, seed=2)  # created elsewhere (CLI)
        app.focus_portfolio(_PID_B)
        await pilot.pause()

        assert app._portfolio_id == _PID_B
        assert picker.value == str(_PID_B)  # header names the portfolio actions target
        assert _PID_B in app._portfolio_ids
        # The picker still works as a way back: picking A really moves the focus.
        picker.value = str(_PID_A)
        await pilot.pause()
        assert app._portfolio_id == _PID_A


@pytest.mark.asyncio
async def test_poll_tick_adds_a_new_portfolio_to_the_picker_without_refocusing(
    shell_env, monkeypatch
) -> None:
    _seed_portfolio_run(shell_env, _PID_A)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        picker = app.query_one("#hdr-portfolio", Select)
        assert app._portfolio_ids == [_PID_A]

        paints: list[object] = []
        original = ShellApp._refresh_runstate

        def _counting(self) -> None:
            paints.append(self._portfolio_id)
            original(self)

        monkeypatch.setattr(ShellApp, "_refresh_runstate", _counting)

        _seed_portfolio_run(shell_env, _PID_B, seed=2)
        app._poll()
        await pilot.pause()

        assert set(app._portfolio_ids) == {_PID_A, _PID_B}
        # Re-seeding the options must not echo a focus change through the picker:
        # one paint (the poll's own), still focused on A and still showing A.
        assert app._portfolio_id == _PID_A
        assert picker.value == str(_PID_A)
        assert paints == [_PID_A]

        # An unchanged roster leaves the options (and an open dropdown) alone.
        before = list(app._portfolio_ids)
        app._poll()
        await pilot.pause()
        assert app._portfolio_ids == before


@pytest.mark.asyncio
async def test_picker_recovers_when_the_mount_time_read_failed(
    shell_env, monkeypatch
) -> None:
    _seed_portfolio_run(shell_env, _PID_A)

    def _boom(*_a, **_k):
        raise RuntimeError("db not ready")

    monkeypatch.setattr(observe, "list_portfolios", _boom)
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert app._portfolio_ids == []

        monkeypatch.undo()
        app._poll()
        await pilot.pause()
        assert app._portfolio_ids == [_PID_A]


@pytest.mark.asyncio
async def test_picker_keeps_its_options_when_a_poll_time_read_fails(
    shell_env, monkeypatch
) -> None:
    _seed_portfolio_run(shell_env, _PID_A)
    _seed_portfolio_run(shell_env, _PID_B, seed=2)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        picker = app.query_one("#hdr-portfolio", Select)
        assert set(app._portfolio_ids) == {_PID_A, _PID_B}

        def _boom(*_a, **_k):
            raise RuntimeError("db hiccup")

        monkeypatch.setattr(observe, "list_portfolios", _boom)
        app._poll()
        await pilot.pause()

        # A transient read error is not "the fund has no portfolios": B must stay
        # pickable instead of vanishing until the next good tick.
        assert set(app._portfolio_ids) == {_PID_A, _PID_B}
        assert picker.value == str(_PID_A)


@pytest.mark.asyncio
async def test_launch_on_an_unlisted_portfolio_still_shows_it_in_the_picker(
    shell_env,
) -> None:
    _seed_portfolio_run(shell_env, _PID_A)
    app = _make_shell(shell_env, portfolio_id=_PID_B)  # B has no rows yet
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert app.query_one("#hdr-portfolio", Select).value == str(_PID_B)
        assert app._portfolio_id == _PID_B


@pytest.mark.asyncio
async def test_shell_collapses_sidebar_to_rail_when_narrow(shell_env) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=(60, 40)) as pilot:  # below the rail threshold
        await pilot.pause()
        assert app._narrow is True
        assert app.query_one("#sidebar", ListView).has_class("narrow")
        # The labels collapse to a short rail abbreviation.
        label = app.query_one("#nav-posture", ListItem).query_one(Label)
        assert str(label.render()) == "Po"


@pytest.mark.asyncio
async def test_narrow_rail_gives_every_view_a_distinct_painted_label(
    shell_env,
) -> None:
    # Leading letters collide (Deliberation / Decisions / Daily brief, Posture /
    # Performance, Audit / Ask), leaving the adviser to guess which rail entry is the
    # approve/kill gate.
    app = _make_shell(shell_env)
    async with app.run_test(size=(60, 40)) as pilot:
        await pilot.pause()
        sidebar = app.query_one("#sidebar", ListView)
        lines = _screen_lines(app)
        labels = []
        for key in _VIEW_KEYS:
            item = app.query_one(f"#nav-{key}", ListItem)
            label = str(item.query_one(Label).render())
            labels.append(label)
            row = lines[item.region.y][: sidebar.region.width]
            assert label in row, (key, label, row)
        assert len(set(labels)) == len(labels), labels


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
async def test_decisions_approve_survives_pm_failure_after_the_ticket(tui_env) -> None:
    """A PM follow-up LLM turn that fails after ``place_orders`` ran does not undo the
    approval: the ticket is kept, the run ends completed, and the status line reports
    the PM failure instead of an action failure."""
    from _fund_fakes import PmFailsAfterOrderModel

    def failing_model():
        return PmFailsAfterOrderModel(universe=UNIVERSE, weights=tui_env.expected)

    app = _make_shell(tui_env, model_factory=failing_model)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert "action failed" not in app.last_status
        assert "approved" in app.last_status
        assert "PM follow-up failed" in app.last_status
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "completed"
        assert session.query(PaperOrder).count() == 1


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
        await pilot.pause()  # let the scope picker mount on top
        assert isinstance(app.screen, ScopePickScreen)
        await pilot.click("#scope-run")  # run scope keeps the reject-finalise kill
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
        assert isinstance(app.screen, ScopePickScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app.last_worker is None
        assert "cancelled" in app.last_status
        assert view.query_one("#decisions-table", DataTable).row_count == 1
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "paused"


@pytest.mark.asyncio
async def test_confirm_screen_opens_on_cancel_so_a_stray_enter_is_not_a_confirm(
    shell_env,
) -> None:
    # The dialog opens right after the Enter that submitted the Ask turn, so a
    # double-tap or key repeat lands on it: the safe button must hold focus.
    app = _make_shell(shell_env)
    results: list[bool | None] = []
    async with app.run_test(size=_SIZE) as pilot:
        app.push_screen(ConfirmScreen("FREEZE the fund"), results.append)
        await pilot.pause()
        assert app.focused is not None
        assert app.focused.id == "confirm-no"
        await pilot.press("enter")
        await pilot.pause()
    assert results == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["k", "K"])
async def test_scope_pick_screen_opens_on_cancel_so_a_stray_enter_picks_nothing(
    tui_env, key
) -> None:
    # `k` (run killswitch) / `K` (portfolio re-arm) then an immediate Enter must not
    # fire the first, consequential scope button.
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()
        assert isinstance(app.screen, ScopePickScreen)
        assert app.focused is not None
        assert app.focused.id == "scope-cancel"
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, ScopePickScreen)
        assert app.last_worker is None
        assert "cancelled" in app.last_status
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
        assert not isinstance(app.screen, ScopePickScreen)
        assert app.last_worker is None
        assert "select a paused" in app.last_status


async def _wait_until(pilot, predicate, *, timeout: float = 5.0) -> None:
    """Pump the app until ``predicate()`` holds (a worker thread's side effect)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await pilot.pause(0.02)


# Every worker that mutates a gate or re-arms autonomy, keyed by a readable id, each
# dispatched exactly as the app's keybindings, pickers and Ask routing do. ``freeze`` is
# deliberately absent: the emergency stop is never refused (see the freeze tests below).
_GATE_ACTIONS = {
    "approve": lambda app, env: app._action_worker(env.run_id, PORTFOLIO_ID, "approve"),
    "reject": lambda app, env: app._action_worker(env.run_id, PORTFOLIO_ID, "reject"),
    "killswitch": lambda app, env: app._action_worker(
        env.run_id, PORTFOLIO_ID, "killswitch"
    ),
    "edit": lambda app, env: app._edit_worker(env.run_id, PORTFOLIO_ID, {"AAA": 1.0}),
    "steer": lambda app, env: app._barge_in_worker(
        env.run_id, PORTFOLIO_ID, "trim AAA", False
    ),
    "rearm": lambda app, env: app._rearm_worker("portfolio", PORTFOLIO_ID, None),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("second", sorted(_GATE_ACTIONS))
async def test_gate_action_is_refused_while_an_approve_is_in_flight(
    tui_env, monkeypatch, second
) -> None:
    # Resuming a gate takes seconds (graph resume + an LLM turn). A repeat or
    # conflicting keypress in that window used to start a second thread on the same
    # run, because Textual's ``exclusive`` only flags the older Worker cancelled.
    import fund.actions as actions_mod

    entered, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def _slow_apply(decision, **_kwargs):
        calls.append(decision)
        entered.set()
        release.wait(timeout=1.5)  # bounded so a regression fails instead of hanging
        return SimpleNamespace(message=f"{decision} done")

    monkeypatch.setattr(actions_mod, "apply_action", _slow_apply)

    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        first = app.last_worker
        await _wait_until(pilot, entered.is_set)

        _GATE_ACTIONS[second](app, tui_env)
        await pilot.pause(0.5)  # long enough for a second thread to start if allowed

        assert calls == ["approve"]
        assert "in progress" in app.last_status

        release.set()
        await first.wait()
        await pilot.pause()
        assert app.last_status == "approve done"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["portfolio", "fund"])
async def test_freeze_is_not_refused_while_a_gate_action_is_stuck_in_flight(
    tui_env, monkeypatch, scope
) -> None:
    # The freeze is the adviser's emergency stop. A hung graph resume (an LLM turn that
    # never returns) must not leave every freeze refused until the TUI restarts.
    import fund.actions as actions_mod

    entered, release = threading.Event(), threading.Event()
    calls: list[tuple[str, str | None]] = []

    def _apply(decision, **kwargs):
        calls.append((decision, kwargs.get("scope")))
        if decision == "approve":
            entered.set()
            release.wait(timeout=5)  # bounded so a regression fails instead of hanging
        return SimpleNamespace(message=f"{decision} done")

    monkeypatch.setattr(actions_mod, "apply_action", _apply)

    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        stuck = app.last_worker
        await _wait_until(pilot, entered.is_set)

        freeze = app._freeze_worker(scope, PORTFOLIO_ID)
        await freeze.wait()  # lands while the approve is still blocked
        await pilot.pause()

        assert ("freeze", scope) in calls
        assert not release.is_set()
        assert app.last_status == "freeze done"

        release.set()
        await stuck.wait()
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_stuck_freeze_does_not_strand_later_gate_actions(
    tui_env, monkeypatch
) -> None:
    # The exemption is one-way: a freeze never takes the single-flight lock, so a
    # freeze stuck on a slow DB cannot refuse the approve/reject gate either.
    import fund.actions as actions_mod

    entered, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def _apply(decision, **_kwargs):
        calls.append(decision)
        if decision == "freeze":
            entered.set()
            release.wait(timeout=5)
        return SimpleNamespace(message=f"{decision} done")

    monkeypatch.setattr(actions_mod, "apply_action", _apply)

    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        stuck = app._freeze_worker("fund", None)
        await _wait_until(pilot, entered.is_set)

        app.action_decide_approve()
        approve = app.last_worker
        await approve.wait()
        await pilot.pause()

        assert calls == ["freeze", "approve"]
        assert "in progress" not in app.last_status

        release.set()
        await stuck.wait()
        await pilot.pause()


@pytest.mark.asyncio
async def test_gate_action_shows_a_working_status_while_it_runs(
    tui_env, monkeypatch
) -> None:
    import fund.actions as actions_mod

    entered, release = threading.Event(), threading.Event()

    def _slow_apply(decision, **_kwargs):
        entered.set()
        release.wait(timeout=1.5)
        return SimpleNamespace(message=f"{decision} done")

    monkeypatch.setattr(actions_mod, "apply_action", _slow_apply)

    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        first = app.last_worker
        await _wait_until(pilot, entered.is_set)

        assert "resolving" in app.last_status  # no silent seconds-long wait

        release.set()
        await first.wait()
        await pilot.pause()


@pytest.mark.asyncio
async def test_gate_action_runs_once_the_previous_one_has_finished(
    tui_env, monkeypatch
) -> None:
    import fund.actions as actions_mod

    calls: list[str] = []

    def _apply(decision, **_kwargs):
        calls.append(decision)
        if len(calls) == 1:
            raise RuntimeError("resume exploded")  # a failure must free the latch too
        return SimpleNamespace(message=f"{decision} done")

    monkeypatch.setattr(actions_mod, "apply_action", _apply)

    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        await app.last_worker.wait()
        await pilot.pause()
        assert "action failed" in app.last_status

        app.action_decide_reject()
        await app.last_worker.wait()
        await pilot.pause()

        assert calls == ["approve", "reject"]
        assert app.last_status == "reject done"


@pytest.mark.asyncio
async def test_reject_pressed_mid_approve_does_not_resume_the_run_twice(
    tui_env, monkeypatch
) -> None:
    # End to end through the real action path: approve then reject on one gate must
    # leave a completed run and a single audit row, never a filled ticket on a run
    # recorded as rejected.
    import fund.agents.graph as graph_mod

    real_resume = graph_mod.resume_run
    entered = threading.Event()
    resumed: list[str] = []

    def _slow_resume(run_id, decision, **kwargs):
        resumed.append(decision)
        entered.set()
        time.sleep(0.6)  # stands in for the PM's post-approval LLM turn
        return real_resume(run_id, decision, **kwargs)

    monkeypatch.setattr(graph_mod, "resume_run", _slow_resume)

    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_approve()
        await _wait_until(pilot, entered.is_set)

        app.action_decide_reject()
        await _wait_until(pilot, lambda: "approved" in app.last_status, timeout=15)
        await pilot.pause(0.3)

        assert resumed == ["approve"]
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "completed"
        log = HumanAuditLogRepository(session)
        assert len(log.list_recent(action="approve")) == 1
        assert log.list_recent(action="reject") == []


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


def test_format_decision_gate_shows_the_risk_verdict() -> None:
    from fund.tui.views import _format_decision_gate

    run = SimpleNamespace(id=_PID_A, portfolio_id=_PID_B)

    def gate(risk_gate: dict) -> str:
        request = {"name": "place_orders", "args": {"risk_gate": risk_gate}}
        return _format_decision_gate({"action_requests": [request]}, run)

    assert "risk gate: passed" in gate({"status": "passed", "violations": []})
    failed = gate(
        {
            "status": "failed",
            "violations": [{"type": "max_weight", "asset": "AAA"}],
        }
    )
    assert "risk gate: FAILED" in failed
    assert "max_weight AAA" in failed
    assert "risk gate: NOT CHECKED" in gate({"status": "not_run"})
    stale = gate(
        {"status": "stale", "violations": [{"type": "max_weight", "asset": "AAA"}]}
    )
    assert "risk gate: STALE" in stale
    assert "max_weight" not in stale  # the breach belongs to a superseded vector
    no_gate = _format_decision_gate(
        {"action_requests": [{"name": "place_orders", "args": {}}]}, run
    )
    assert "risk gate" not in no_gate


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
async def test_mandate_save_rejects_an_unbreachable_drift_threshold(shell_env) -> None:
    # `5` typed for "5%" exceeds the L1 maximum of 2: the trigger would never fire.
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-drift", Input).value = "5"
        app.action_save_mandate()
        await pilot.pause()
        assert "invalid mandate" in app.last_status
    with shell_env.get_session() as session:
        assert observe.get_mandate(session, PORTFOLIO_ID).drift_l1_threshold == 0.1
        assert HumanAuditLogRepository(session).list_recent(action="mandate_edit") == []


@pytest.mark.asyncio
async def test_mandate_invalid_capital_reports_on_status_line_without_crashing(
    shell_env,
) -> None:
    # pydantic's error text carries "[type=..., input_value=..., input_type=...]",
    # which a markup-parsing status line rejects (MarkupError) and kills the cockpit.
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-capital", Input).value = "-5"
        app.action_save_mandate()
        await pilot.pause()
        assert app.last_status.startswith("invalid mandate")
        shown = str(app.query_one("#status-bar", Static).render())
        assert shown == app.last_status  # bracketed pydantic detail survives verbatim
        assert "[type=" in shown


@pytest.mark.asyncio
async def test_status_bar_renders_bracketed_text_verbatim(shell_env) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        message = "cash [5%] left; missing key [OLLAMA_API_KEY]; bad [/INST] tag"
        app._set_status(message)
        await pilot.pause()
        assert str(app.query_one("#status-bar", Static).render()) == message


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


# --- Mandate gate checkboxes + remembered-directives list (T10) --------------


@pytest.mark.asyncio
async def test_mandate_gate_checkbox_toggles_hitl_gates(shell_env) -> None:
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        place = view.query_one("#gate-place-orders", Checkbox)
        assert place.value is True  # the default mandate gates place_orders
        save_profile = view.query_one("#gate-save-profile", Checkbox)
        assert save_profile.disabled is True  # always-on indicator, not toggleable
        assert save_profile.value is True
        place.value = False  # toggle the PM gate off
        app.action_save_mandate()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "mandate updated" in app.last_status
    with shell_env.get_session() as session:
        reloaded = observe.get_mandate(session, PORTFOLIO_ID)
        assert reloaded.hitl_gates == ()  # the toggle persisted into the mandate


@pytest.mark.asyncio
async def test_mandate_remove_directive_drops_it_on_save(shell_env) -> None:
    with shell_env.get_session() as session:
        mandate = make_mandate().model_copy(
            update={"remembered_directives": ("Trim AAA", "Keep 5% cash")}
        )
        MandateRepository(session).upsert(mandate)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        listview = view.query_one("#mandate-directives", ListView)
        assert listview.children  # the two directives render
        assert view._directives == ["Trim AAA", "Keep 5% cash"]
        listview.index = 0
        view.action_remove_directive()
        await pilot.pause()
        assert view._directives == ["Keep 5% cash"]
        app.action_save_mandate()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
    with shell_env.get_session() as session:
        reloaded = observe.get_mandate(session, PORTFOLIO_ID)
        assert reloaded.remembered_directives == ("Keep 5% cash",)


@pytest.mark.asyncio
async def test_mandate_directive_with_bracket_sequences_renders_literally(
    shell_env,
) -> None:
    """A remembered steer typed by the adviser is data, never Textual markup."""
    directive = "cap energy [max=5%, hard] and never [/] trim"
    with shell_env.get_session() as session:
        mandate = make_mandate().model_copy(
            update={"remembered_directives": (directive,)}
        )
        MandateRepository(session).upsert(mandate)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        label = view.query_one("#mandate-directives", ListView).query_one(Label)
        assert str(label.render()) == directive


@pytest.mark.asyncio
async def test_poll_tick_keeps_unsaved_mandate_edits(shell_env) -> None:
    with shell_env.get_session() as session:
        mandate = make_mandate().model_copy(
            update={"remembered_directives": ("Trim AAA", "Keep 5% cash")}
        )
        MandateRepository(session).upsert(mandate)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-drift", Input).value = "0.25"
        view.query_one("#gate-place-orders", Checkbox).value = False
        view.query_one("#mandate-directives", ListView).index = 0
        view.action_remove_directive()
        await pilot.pause()

        app._poll()  # the timer tick must not repaint over what the adviser typed
        await pilot.pause()

        assert view.query_one("#mandate-drift", Input).value == "0.25"
        assert view.query_one("#gate-place-orders", Checkbox).value is False
        assert view._directives == ["Keep 5% cash"]
        app.action_save_mandate()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
    with shell_env.get_session() as session:
        reloaded = observe.get_mandate(session, PORTFOLIO_ID)
        assert reloaded.drift_l1_threshold == 0.25
        assert reloaded.hitl_gates == ()
        assert reloaded.remembered_directives == ("Keep 5% cash",)


@pytest.mark.asyncio
async def test_poll_tick_keeps_picked_guardrail_band_so_override_is_flagged(
    shell_env,
) -> None:
    # The ceiling is balanced; the adviser picks aggressive and a tick lands before
    # ``o``. Repainting the ceiling over the pick would log balanced -> balanced with
    # no override flag, silently dropping the MiFID audit signal.
    _seed_mandate_and_profile(shell_env, band="balanced")
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-band", Select).value = "aggressive"

        app._poll()
        await pilot.pause()

        assert view.proposed_band() == "aggressive"
        app.action_override_guardrail()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
    with shell_env.get_session() as session:
        rows = HumanAuditLogRepository(session).list_recent(action="guardrail_override")
        assert len(rows) == 1
        assert rows[0].new_value == "aggressive"
        assert rows[0].override_flag is True
        assert rows[0].suitability_flag is True


@pytest.mark.asyncio
async def test_poll_tick_still_repaints_an_untouched_mandate_view(shell_env) -> None:
    # The edit guard must not freeze the view: with nothing edited, an external change
    # to the stored mandate shows up on the next tick.
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        assert view.query_one("#mandate-drift", Input).value == "0.1"
        with shell_env.get_session() as session:
            MandateRepository(session).upsert(
                make_mandate().model_copy(update={"drift_l1_threshold": 0.3})
            )
            session.commit()

        app._poll()
        await pilot.pause()

        assert view.query_one("#mandate-drift", Input).value == "0.3"


@pytest.mark.asyncio
async def test_mandate_edits_reset_when_the_focused_portfolio_changes(
    shell_env,
) -> None:
    # An explicit refocus (not a timer tick) repaints even a dirty editor: stale
    # inputs from portfolio A must never be saved against portfolio B.
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-drift", Input).value = "0.25"

        app.focus_portfolio(_PID_B)
        await pilot.pause()

        assert view.query_one("#mandate-drift", Input).value == ""


# --- Mandate polling must not repaint over unsaved edits ----------------------

_FAST_POLL = 0.2
_SETTLE = 0.9  # several poll ticks at _FAST_POLL


@pytest.mark.asyncio
async def test_mandate_poll_keeps_unsaved_run_policy_edit(shell_env) -> None:
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID, poll_interval=_FAST_POLL)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-drift", Input).value = "0.25"
        view.query_one("#gate-place-orders", Checkbox).value = False
        await pilot.pause(_SETTLE)
        assert view.query_one("#mandate-drift", Input).value == "0.25"
        assert view.query_one("#gate-place-orders", Checkbox).value is False
        proposed = view.proposed_mandate()
        assert proposed.drift_l1_threshold == 0.25
        assert proposed.hitl_gates == ()
        app.action_save_mandate()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
    with shell_env.get_session() as session:
        assert observe.get_mandate(session, PORTFOLIO_ID).drift_l1_threshold == 0.25


@pytest.mark.asyncio
async def test_mandate_poll_keeps_unsaved_band_pick_for_override(shell_env) -> None:
    _seed_mandate_and_profile(shell_env, band="balanced")
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID, poll_interval=_FAST_POLL)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-band", Select).value = "aggressive"
        await pilot.pause(_SETTLE)
        assert view.proposed_band() == "aggressive"
        app.action_override_guardrail()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
    with shell_env.get_session() as session:
        rows = HumanAuditLogRepository(session).list_recent(action="guardrail_override")
        assert len(rows) == 1
        assert rows[0].prev_value == "balanced"
        assert rows[0].new_value == "aggressive"
        assert rows[0].override_flag is True


@pytest.mark.asyncio
async def test_mandate_poll_keeps_removed_directive_removed(shell_env) -> None:
    with shell_env.get_session() as session:
        mandate = make_mandate().model_copy(
            update={"remembered_directives": ("Trim AAA", "Keep 5% cash")}
        )
        MandateRepository(session).upsert(mandate)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID, poll_interval=_FAST_POLL)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-directives", ListView).index = 0
        view.action_remove_directive()
        await pilot.pause(_SETTLE)
        assert view._directives == ["Keep 5% cash"]
        assert view.proposed_mandate().remembered_directives == ("Keep 5% cash",)


@pytest.mark.asyncio
async def test_mandate_poll_still_repaints_fields_the_user_did_not_edit(
    shell_env,
) -> None:
    """Only a dirty field is protected: an external change to a clean field lands."""
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID, poll_interval=_FAST_POLL)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        view.query_one("#mandate-benchmark", Input).value = "SPY"  # unsaved edit
        with shell_env.get_session() as session:
            external = observe.get_mandate(session, PORTFOLIO_ID).model_copy(
                update={"drift_l1_threshold": 0.3}
            )
            MandateRepository(session).upsert(external)
            session.commit()
        await pilot.pause(_SETTLE)
        assert view.query_one("#mandate-drift", Input).value == "0.3"
        assert view.query_one("#mandate-benchmark", Input).value == "SPY"


@pytest.mark.asyncio
async def test_mandate_saved_field_is_released_even_if_typed_text_is_not_canonical(
    shell_env,
) -> None:
    """A save lands the typed value, but the persisted rendering can differ from the
    typed text (``0.20`` is stored as ``0.2``). The field must still be released, or
    an external change never reaches it and a later save overwrites that change."""
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID, poll_interval=_FAST_POLL)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        drift = view.query_one("#mandate-drift", Input)
        drift.value = "0.20"
        app.action_save_mandate()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause(_SETTLE)
        assert drift.value == "0.2"
        with shell_env.get_session() as session:
            external = observe.get_mandate(session, PORTFOLIO_ID).model_copy(
                update={"drift_l1_threshold": 0.3}
            )
            MandateRepository(session).upsert(external)
            session.commit()
        await pilot.pause(_SETTLE)
        assert drift.value == "0.3"


@pytest.mark.asyncio
async def test_mandate_save_ack_keeps_text_typed_while_the_save_was_in_flight(
    shell_env,
) -> None:
    """Only the fields still holding the submitted value are released: a keystroke
    typed between ctrl+s and the commit acknowledgement must not be repainted away."""
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID, poll_interval=_FAST_POLL)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        drift = view.query_one("#mandate-drift", Input)
        capital = view.query_one("#mandate-capital", Input)
        drift.value = "0.20"
        capital.value = "250000"
        submitted = view.proposed_mandate()
        drift.value = "0.205"
        view.mark_saved(submitted)
        assert "capital" not in view._painted  # still the submitted value: released
        assert "drift" in view._painted  # typed again since: still guarded
        await pilot.pause(_SETTLE)
        assert drift.value == "0.205"


@pytest.mark.asyncio
async def test_mandate_edit_made_after_a_save_is_still_protected(shell_env) -> None:
    """Releasing the saved fields must not make the next edit vulnerable again."""
    _seed_mandate_and_profile(shell_env)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID, poll_interval=_FAST_POLL)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_mandate(app, pilot)
        drift = view.query_one("#mandate-drift", Input)
        drift.value = "0.20"
        app.action_save_mandate()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause(_SETTLE)
        drift.value = "0.35"
        await pilot.pause(_SETTLE)
        assert drift.value == "0.35"


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
    elif status == "responded":
        repo.mark_responded(run.id)
    return run


def test_posture_health_tiers() -> None:
    from fund.tui.views import _posture_health

    # In-band drift → ok; at/above the mandate threshold → warn; a failed/blocked
    # latest run → breach (dot + label, so two tiers sharing ● still read apart).
    assert _posture_health(0.05, 0.1, "completed") == ("●", "ok")
    assert _posture_health(0.2, 0.1, "completed") == ("◐", "warn")
    assert _posture_health(0.0, None, "incomplete") == ("●", "breach")
    assert _posture_health(0.0, 0.1, "running") == ("●", "ok")


def test_responded_run_badges_as_awaiting() -> None:
    """A barge-in that re-gated (status ``"responded"``) reads as awaiting on both the
    roster glyph and the fund strip, exactly like a plain ``"paused"`` gate."""
    import uuid

    from fund.tui.views import _fund_strip, _RosterRow, _run_glyph

    assert _run_glyph("responded") == _run_glyph("paused") == "⏸"

    rows = [
        _RosterRow(uuid.uuid4(), "paused", "⏸", "●", "ok"),
        _RosterRow(uuid.uuid4(), "responded", "⏸", "●", "ok"),
        _RosterRow(uuid.uuid4(), "completed", "✓", "●", "ok"),
    ]
    assert "⏸2 awaiting" in _fund_strip(rows)


@pytest.mark.asyncio
async def test_header_glyph_reads_responded_run_as_awaiting(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1, status="responded")
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE):
        assert app._run_state_glyph() == "⏸ awaiting"


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
async def test_posture_poll_reads_only_the_latest_run(shell_env, monkeypatch) -> None:
    # The roster, the focused detail and the header glyph each need only the newest
    # run's status, so a 2s tick must not load a portfolio's whole run history.
    with shell_env.get_session() as session:
        for seed in range(3):
            _seed_run(session, _PID_A, seed=seed)
        session.commit()

    rows_read: list[int] = []
    real = observe.list_portfolio_runs

    def spy(session, portfolio_id, **kwargs):
        runs = real(session, portfolio_id, **kwargs)
        rows_read.append(len(runs))
        return runs

    monkeypatch.setattr(observe, "list_portfolio_runs", spy)

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert "ACTIVE RUN" in view.detail_text
        assert app._run_state_glyph() == "▶ running"

    assert rows_read
    assert max(rows_read) == 1


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
    from _profiler_fakes import make_answers

    from fund.agents.profiler import build_constraint_set

    with get_session() as session:
        seed_panel(session)
        session.commit()
    # The allocator optimises under the ConstraintSet the TUI's profiler step will
    # derive from these answers, not the stock one.
    onboarded = build_constraint_set(make_answers(), portfolio_id=str(_PID_NEW))
    with get_session() as session:
        expected = expected_weights(session, onboarded)

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
async def test_questionnaire_rejects_non_positive_capital(shell_env) -> None:
    """Non-positive capital is rejected inline, so the modal and its answers stay."""
    from _profiler_fakes import make_answers

    answers = make_answers()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        screen = QuestionnaireScreen(ask_capital=True)
        app.push_screen(screen)
        await pilot.pause()
        _fill_questionnaire(screen, answers, capital="-5000")
        with pytest.raises(ValueError, match="positive"):
            screen.build_submission()
        # _submit surfaces it inline and does NOT dismiss — the answers are preserved.
        screen._submit()
        await pilot.pause()
        assert isinstance(app.screen, QuestionnaireScreen)


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


def _seed_rearm_with_rationale(session) -> None:
    HumanAuditLogRepository(session).append(
        action="rearm",
        actor="human",
        target="fund",
        new_value={"state": "armed"},
        rationale="drawdown reviewed with the risk committee",
    )


@pytest.mark.asyncio
async def test_audit_detail_shows_rationale_and_target(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_rearm_with_rationale(session)
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        view.select_index(0)
        assert "target: fund" in view.detail_text
        assert "drawdown reviewed with the risk committee" in view.detail_text


@pytest.mark.asyncio
async def test_audit_detail_renders_bracketed_rationale_verbatim(shell_env) -> None:
    """A rationale is free text: brackets are literal, never Textual markup.

    ``[/ops]`` is an unmatched closing tag that makes markup parsing raise, and
    ``[CRO approved]`` would be swallowed as a style tag; either corrupts or crashes
    the pane that is labelled an immutable audit record.
    """
    rationale = "closed per [/ops] review [CRO approved]"
    with shell_env.get_session() as session:
        HumanAuditLogRepository(session).append(
            action="rearm", actor="human", target="fund", rationale=rationale
        )
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        view.select_index(0)
        await pilot.pause()
        rendered = view.query_one("#audit-detail", Static).visual.plain
    assert f"rationale: {rationale}" in rendered


@pytest.mark.asyncio
async def test_audit_export_includes_rationale_and_target(shell_env, tmp_path) -> None:
    import csv

    with shell_env.get_session() as session:
        _seed_rearm_with_rationale(session)
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        out = tmp_path / "audit.csv"
        view.export_csv(str(out))
    with out.open(newline="", encoding="utf-8") as handle:
        (row,) = csv.DictReader(handle)
    assert row["target"] == "fund"
    assert row["rationale"] == "drawdown reviewed with the risk committee"


def test_audit_csv_cells_neutralise_formula_injection() -> None:
    """Agent-generated cells starting with a spreadsheet trigger are de-fanged."""
    from fund.tui.views import _audit_cell, _csv_safe

    assert (
        _csv_safe("=HYPERLINK('http://evil','x')") == "'=HYPERLINK('http://evil','x')"
    )
    assert _csv_safe("+1") == "'+1"
    assert _csv_safe("-2") == "'-2"
    assert _csv_safe("@cmd") == "'@cmd"
    assert _csv_safe("safe") == "safe"
    assert _csv_safe("") == ""
    # Scalars route through the same guard; a JSON object renders as {...} (safe).
    assert _audit_cell("=evil") == "'=evil"
    assert _audit_cell({"k": "=v"}) == '{"k": "=v"}'


# --- Deliberation replay view: transcript + step detail (T9) ------------------


async def _open_deliberation(app, pilot) -> DeliberationView:
    """Switch the shell to the Deliberation view and paint it."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-deliberation"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-deliberation", DeliberationView)


def _seed_run_with_steps(session) -> uuid.UUID:
    """Seed one run for _PID_A with three ordered decisions; return its id."""
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
    repo.append_decision(run.id, agent="risk", step="risk_check")
    repo.append_decision(run.id, agent="executor", step="place_orders")
    return run.id


@pytest.mark.asyncio
async def test_deliberation_renders_run_steps_in_order_and_detail(shell_env) -> None:
    with shell_env.get_session() as session:
        run_id = _seed_run_with_steps(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        assert view.query_one("#delib-runs", DataTable).row_count == 1
        view.select_run(run_id)
        await pilot.pause()
        steps = view.query_one("#delib-steps", DataTable)
        assert steps.row_count == 3  # the three decisions, in order
        view.select_step(0)
        await pilot.pause()
        assert "allocator" in view.detail_text  # first step
        view.select_step(2)
        await pilot.pause()
        assert "executor" in view.detail_text  # last step


@pytest.mark.asyncio
async def test_deliberation_row_selection_drives_steps_then_detail(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_run_with_steps(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        runs_table = view.query_one("#delib-runs", DataTable)
        runs_table.focus()
        runs_table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert view.selected_run_id is not None
        steps_table = view.query_one("#delib-steps", DataTable)
        assert steps_table.row_count == 3
        steps_table.focus()
        steps_table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert view.selected_step == 0
        assert "allocator" in view.detail_text


@pytest.mark.asyncio
async def test_deliberation_empty_without_focus(shell_env) -> None:
    app = _make_shell(shell_env)  # fund-wide landing, nothing focused
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        assert view.query_one("#delib-runs", DataTable).row_count == 0
        assert "select a step" in view.detail_text


# --- Deliberation drill-down read-outs: build family (Phase 4 T1) -------------


def test_readout_build_renders_record_and_mock_labels() -> None:
    """The allocator build read-out shows recorded universe/objective/constraints and
    labels the aspirational sources (candidate universe, factors, views) as mock."""
    from fund.tui.readouts import render_readout

    payload = {
        "constraint_set": make_constraint_set().model_dump(mode="json"),
        "weights": {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
    }
    text = render_readout("allocator", "optimize_portfolio", payload)
    assert text is not None
    assert "3 selected" in text  # post-filter universe size from the weights keys
    assert "AAA" in text
    assert "objective" in text  # constraints/objective carry real recorded values
    assert "bounds" in text
    assert "mock — no live source yet" in text  # factors/signals/views labelled


def test_readout_returns_none_for_unmapped_step() -> None:
    """A step with no read-out returns None so the caller keeps the JSON dump."""
    from fund.tui.readouts import render_readout

    assert render_readout("economist", "report", {"note": "macro"}) is None
    assert render_readout("executor", "place_orders", None) is None


@pytest.mark.asyncio
async def test_deliberation_allocator_step_shows_build_readout(shell_env) -> None:
    """Selecting the allocator step paints the build read-out into the detail mirror
    (universe + constraints + mock labels), replacing the raw JSON dump."""
    constraint_set = make_constraint_set().model_dump(mode="json")
    with shell_env.get_session() as session:
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
            constraint_set=constraint_set,
            llm_response='{"weights": {"AAA": 0.6, "BBB": 0.4}}',
        )
        session.commit()
        run_id = run.id
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(0)
        await pilot.pause()
        detail = view.detail_text
        assert "2 selected" in detail  # post-filter universe from the weights keys
        assert "AAA" in detail
        assert "objective" in detail  # recorded constraints rendered, not dumped
        assert "mock — no live source yet" in detail
        assert "{" not in detail  # the structured read-out replaced the JSON dump


def test_readout_results_weights_ranked_and_pending_markers() -> None:
    """The allocator results read-out ranks the recorded weights by size and marks the
    frontier/backtest as pending (computed live on expand in T6), not as mock."""
    from fund.tui.readouts import render_readout

    payload = {"weights": {"BBB": 0.3, "AAA": 0.7}}
    text = render_readout("allocator", "optimize_portfolio", payload)
    assert text is not None
    results = text[text.index("results") :]
    assert "weights" in results
    assert results.index("AAA") < results.index("BBB")  # ranked by weight, descending
    assert results.count("compute to view") == 2  # frontier + backtest pending markers


def test_readout_risk_pending_verdict_and_exposures() -> None:
    """The risk read-out shows a dot+label verdict (pending until recomputed in T6) and
    an exposures section — sector pending, factor mock — without scraping LLM prose."""
    from fund.tui.readouts import render_readout

    text = render_readout("risk", "risk_check", None)
    assert text is not None
    assert "verdict" in text
    assert "○ pending" in text  # dot plus label, never the glyph alone
    assert "sector exposures" in text
    assert "compute to view" in text  # sector verdict/exposures pending, computed in T6
    assert "mock — no live source yet" in text  # factor exposures labelled mock


@pytest.mark.asyncio
async def test_deliberation_readout_risk_step_verdict(shell_env) -> None:
    """Selecting the risk step paints the risk read-out: verdict + exposures."""
    with shell_env.get_session() as session:
        run_id = _seed_run_with_steps(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(1)  # the risk step (allocator=0, risk=1, executor=2)
        await pilot.pause()
        detail = view.detail_text
        assert "verdict" in detail
        assert "○ pending" in detail
        assert "factor exposures" in detail


def test_readout_weights_tolerate_null_and_bool_values() -> None:
    """A null or boolean weight must not crash the sort nor render as a percentage."""
    from fund.tui.readouts import render_readout

    payload = {"weights": {"AAA": 0.6, "BBB": None, "CCC": True}}
    text = render_readout("allocator", "optimize_portfolio", payload)
    assert text is not None  # the sort tolerates None instead of raising TypeError
    results = text[text.index("results") :]
    assert "AAA · 60.0%" in results  # the genuine numeric weight
    assert "100.0%" not in results  # a JSON `true` is not a 100% weight
    assert "True" in results  # it renders raw, flagging the type anomaly
    assert results.index("AAA") < results.index("BBB")  # real weight ranks first


def test_readout_empty_weights_and_missing_constraints_degrade() -> None:
    """Empty weights and a missing constraint set degrade to labels, never a crash."""
    from fund.tui.readouts import render_readout

    text = render_readout("allocator", "optimize_portfolio", {"weights": {}})
    assert text is not None
    assert "0 selected" in text  # post-filter universe empty
    assert "weights · none recorded" in text
    assert "objective · unavailable (not recorded)" in text  # no constraint_set


def test_readout_truncates_long_universe_and_weights() -> None:
    """More than _MAX_NAMES holdings collapse the tail into a '… +k more' marker."""
    from fund.tui.readouts import render_readout

    weights = {f"T{i:02d}": round(1.0 - i * 0.01, 2) for i in range(10)}  # 10 > cap 8
    text = render_readout("allocator", "optimize_portfolio", {"weights": weights})
    assert text is not None
    assert text.count("… +2 more") == 2  # once in the universe block, once in weights


def test_readout_allocator_without_payload_returns_none() -> None:
    """The allocator read-out is gated on a non-empty payload; None/empty falls back."""
    from fund.tui.readouts import render_readout

    assert render_readout("allocator", "optimize_portfolio", None) is None
    assert render_readout("allocator", "optimize_portfolio", {}) is None


@pytest.mark.asyncio
async def test_deliberation_unmapped_step_with_payload_falls_back_to_json(
    shell_env,
) -> None:
    """An unmapped step that carries a payload still shows the raw JSON dump."""
    with shell_env.get_session() as session:
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=_PID_A,
            asof=date(2026, 1, 2),
            seed=1,
            universe=[],
            optimizer_config={},
        )
        repo.append_decision(
            run.id, agent="economist", step="report", views={"note": "macro"}
        )
        session.commit()
        run_id = run.id
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(0)
        await pilot.pause()
        detail = view.detail_text
        assert '"note"' in detail  # raw JSON dump, not a structured read-out
        assert "macro" in detail


# --- Deliberation drill-down recompute: expand → worker → render (Phase 4 T6) --


def _seed_allocator_run(session) -> uuid.UUID:
    """Seed a run with a constraint-set-bearing allocator step; return its id."""
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
        constraint_set=make_constraint_set().model_dump(mode="json"),
        llm_response='{"weights": {"AAA": 0.6, "BBB": 0.4}}',
    )
    repo.append_decision(run.id, agent="risk", step="risk_check")
    return run.id


@pytest.mark.asyncio
async def test_deliberation_expand_renders_recomputed_results(
    shell_env, monkeypatch
) -> None:
    """Pressing expand on the allocator step runs the worker and paints the real
    frontier + backtest numbers over the recompute, replacing the pending markers."""
    from fund import readouts_compute

    canned = readouts_compute.ResultsCompute(
        frontier=readouts_compute.FrontierPoint(
            mean=0.12, standard_deviation=0.2, sharpe_ratio=0.6, n_observations=40
        ),
        frontier_error=None,
        backtest=readouts_compute.BacktestResult(
            mean=0.1,
            standard_deviation=0.18,
            sharpe_ratio=0.55,
            max_drawdown=-0.08,
            n_observations=30,
            n_folds=2,
            out_of_sample=True,
            window_start="2026-01-01",
            window_end="2026-02-01",
        ),
        backtest_error=None,
    )
    monkeypatch.setattr(readouts_compute, "compute_results", lambda *_a, **_k: canned)
    with shell_env.get_session() as session:
        run_id = _seed_allocator_run(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(0)  # the allocator step
        await pilot.pause()
        assert "compute to view" in view.detail_text  # pending before expand
        app.action_expand_readout()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        detail = view.detail_text
        assert "compute to view" not in detail  # markers replaced by live numbers
        assert "sharpe 0.60" in detail  # the frontier point
        assert "sharpe 0.55" in detail  # the backtest


@pytest.mark.asyncio
async def test_deliberation_expand_renders_recomputed_risk_verdict(
    shell_env, monkeypatch
) -> None:
    """Expanding the risk step paints the live verdict (● breach from a max_weight
    violation) and the sector-exposure breakdown, replacing the pending placeholder."""
    from fund import readouts_compute

    canned = readouts_compute.RiskCompute(
        passed=False,
        violations=(
            {"type": "max_weight", "asset": "AAA", "value": 0.6, "limit": 0.4},
        ),
        sector_exposures=(("Tech", 0.8), ("Energy", 0.2)),
        error=None,
    )
    monkeypatch.setattr(readouts_compute, "compute_risk", lambda *_a, **_k: canned)
    with shell_env.get_session() as session:
        run_id = _seed_allocator_run(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(1)  # the risk step
        await pilot.pause()
        assert "○ pending" in view.detail_text  # pending before expand
        app.action_expand_readout()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        detail = view.detail_text
        assert "● breach" in detail  # live verdict tier, dot plus label
        assert "max_weight" in detail  # the driving violation
        assert "Tech" in detail  # sector-exposure breakdown


@pytest.mark.asyncio
async def test_deliberation_expand_replaces_recomputing_status_when_done(
    shell_env, monkeypatch
) -> None:
    """The ``recomputing read-out…`` status must not outlive the compute: left in
    place it reads as a hung computation long after the numbers painted in."""
    from fund import readouts_compute

    canned = readouts_compute.RiskCompute(
        passed=True, violations=(), sector_exposures=None, error=None
    )
    monkeypatch.setattr(readouts_compute, "compute_risk", lambda *_a, **_k: canned)
    with shell_env.get_session() as session:
        run_id = _seed_allocator_run(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(1)  # the risk step
        await pilot.pause()
        app.action_expand_readout()
        assert "recomputing" in app.last_status
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "recomputing" not in app.last_status
        assert "read-out ready" in app.last_status


@pytest.mark.asyncio
async def test_deliberation_expand_scopes_to_selected_allocator_step(
    shell_env, monkeypatch
) -> None:
    """In a re-optimized run (two allocator steps), expanding a step recomputes with
    THAT step's own weights and constraint set — so the frontier/backtest match the
    weights shown for it, never a different proposal's."""
    from fund import readouts_compute

    captured = {}

    def _capture(
        _session, _asof, _universe, constraint_set, weights, base_currency=None
    ):
        captured["constraint_set"] = constraint_set
        captured["weights"] = dict(weights)
        return readouts_compute.ResultsCompute(
            frontier=None, frontier_error="stub", backtest=None, backtest_error="stub"
        )

    monkeypatch.setattr(readouts_compute, "compute_results", _capture)
    first_cs = make_constraint_set().model_dump(mode="json")
    first_cs["bounds"] = {"min_weights": 0.0, "max_weights": 0.5, "budget": 1.0}
    second_cs = make_constraint_set().model_dump(mode="json")
    second_cs["bounds"] = {"min_weights": 0.0, "max_weights": 0.3, "budget": 1.0}
    with shell_env.get_session() as session:
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
            constraint_set=first_cs,
            llm_response='{"weights": {"AAA": 0.5, "BBB": 0.5}}',
        )
        repo.append_decision(
            run.id,
            agent="allocator",
            step="optimize_portfolio",
            constraint_set=second_cs,
            llm_response='{"weights": {"AAA": 0.7, "CCC": 0.3}}',
        )
        session.commit()
        run_id = run.id
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(0)  # the FIRST (earlier) allocator proposal
        await pilot.pause()
        app.action_expand_readout()
        await pilot.pause()
        first_worker = app.last_worker
        assert first_worker is not None
        await first_worker.wait()
        await pilot.pause()
        assert captured["constraint_set"] == first_cs  # step 0's own bounds
        assert captured["weights"] == {"AAA": 0.5, "BBB": 0.5}  # step 0's own weights

        view.select_step(1)  # the SECOND (re-optimized) allocator proposal
        await pilot.pause()
        app.action_expand_readout()
        await pilot.pause()
        second_worker = app.last_worker
        assert second_worker is not None
        await second_worker.wait()
        await pilot.pause()
    assert captured["constraint_set"] == second_cs  # step 1's own bounds
    assert captured["weights"] == {"AAA": 0.7, "CCC": 0.3}  # step 1's own weights


@pytest.mark.asyncio
async def test_deliberation_expand_recomputes_in_the_mandate_base_currency(
    shell_env, monkeypatch
) -> None:
    """The replay converts prices to the portfolio mandate's base currency, as the
    agent's allocator tools did, so the drill-down matches the decided numbers."""
    from fund import readouts_compute

    captured = {}

    def _capture(
        _session, _asof, _universe, _constraint_set, _weights, base_currency=None
    ):
        captured["base_currency"] = base_currency
        return readouts_compute.ResultsCompute(
            frontier=None, frontier_error="stub", backtest=None, backtest_error="stub"
        )

    monkeypatch.setattr(readouts_compute, "compute_results", _capture)
    with shell_env.get_session() as session:
        MandateRepository(session).upsert(
            make_mandate(_PID_A).model_copy(update={"base_currency": "CHF"})
        )
        run_id = _seed_allocator_run(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(0)
        await pilot.pause()
        app.action_expand_readout()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
    assert captured["base_currency"] == "CHF"


@pytest.mark.asyncio
async def test_deliberation_paint_readout_does_not_clobber_live_detail(
    shell_env,
) -> None:
    """A recompute landing after a live round has started must not overwrite the
    streamed detail with stale replay content (it is stored for when replay resumes)."""
    from fund import readouts_compute

    with shell_env.get_session() as session:
        run_id = _seed_allocator_run(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.select_step(0)
        await pilot.pause()
        assert "compute to view" in view.detail_text  # pending replay render
        view.start_live(run_id, _PID_A)  # a live round begins while a recompute runs
        await pilot.pause()
        canned = readouts_compute.ResultsCompute(
            frontier=readouts_compute.FrontierPoint(0.1, 0.2, 0.5, 40),
            frontier_error=None,
            backtest=None,
            backtest_error="too short",
        )
        view.paint_readout(run_id, "allocator", 0, canned)
        await pilot.pause()
        assert "sharpe 0.50" not in view.detail_text  # live detail was not clobbered
        assert view._readout is not None  # stored for when replay resumes


@pytest.mark.asyncio
async def test_deliberation_recompute_ignores_allocator_message_with_same_index(
    shell_env,
) -> None:
    """An allocator *report message* shares the numbering space of decision indices.

    A recompute is keyed by the decision's ``decision_index``; a message row that
    happens to carry the same number must not inherit that proposal's frontier and
    backtest read-out.
    """
    from fund import readouts_compute
    from fund.observe import TranscriptEntry

    canned = readouts_compute.ResultsCompute(
        frontier=readouts_compute.FrontierPoint(0.1, 0.2, 0.5, 40),
        frontier_error=None,
        backtest=None,
        backtest_error="too short",
    )
    with shell_env.get_session() as session:
        run_id = _seed_allocator_run(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view.select_run(run_id)
        await pilot.pause()
        view.paint_readout(run_id, "allocator", 4, canned)
        message = TranscriptEntry(
            source="message",
            agent="allocator",
            step=None,
            index=4,
            text="allocator report",
            payload=None,
        )
        decision = TranscriptEntry(
            source="decision",
            agent="allocator",
            step="optimize_portfolio",
            index=4,
            text="",
            payload={"weights": {"AAA": 1.0}},
        )
        assert view._compute_for(message) is None
        assert view._compute_for(decision) is canned


def test_compute_risk_empty_weights_with_mapping_degrades_to_none(db_session) -> None:
    """An empty roll-up (mapping present, no usable weights) is None, not an empty
    tuple — so the renderer shows 'unavailable', never a bare header with no rows."""
    from fund.readouts_compute import compute_risk

    constraint_set = make_constraint_set().model_dump(mode="json")
    result = compute_risk({}, constraint_set, {"AAA": "Tech"}, session=db_session)
    assert result.sector_exposures is None


def test_readout_allocator_renders_compute_without_payload() -> None:
    """A recompute result still renders even when the allocator step carries no payload
    (optimizer returned an error), rather than silently discarding it."""
    from fund import readouts_compute
    from fund.tui.readouts import render_readout

    compute = readouts_compute.ResultsCompute(
        frontier=readouts_compute.FrontierPoint(
            mean=0.1, standard_deviation=0.2, sharpe_ratio=0.5, n_observations=40
        ),
        frontier_error=None,
        backtest=None,
        backtest_error="too short",
    )
    assert render_readout("allocator", "optimize_portfolio", None) is None  # no compute
    text = render_readout("allocator", "optimize_portfolio", None, compute)
    assert text is not None
    assert "sharpe 0.50" in text  # the frontier point surfaces
    assert "weights · none recorded" in text  # empty payload rendered, not crashed


# --- Deliberation live mode: streamed timeline + follow/inspect (GP0-1) --------


def _scripted_stream(run_id: uuid.UUID) -> list[StreamEvent]:
    """Two live subagent steps, then the terminal paused-gate event, in order."""
    return [
        StreamEvent(
            kind="step",
            run_id=run_id,
            index=0,
            agent="economist",
            step="report",
            status=StepStatus.LIVE,
            text="macro read",
        ),
        StreamEvent(
            kind="step",
            run_id=run_id,
            index=1,
            agent="allocator",
            step="report",
            status=StepStatus.LIVE,
            text="proposed weights",
        ),
        StreamEvent(
            kind="paused",
            run_id=run_id,
            index=2,
            agent="orchestrator",
            step="place_orders",
            status=StepStatus.PENDING,
            text="awaiting the execution gate",
            payload={"action_requests": []},
        ),
    ]


@pytest.mark.asyncio
async def test_deliberation_live_renders_steps_and_flips_markers(shell_env) -> None:
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        rid = uuid.uuid4()
        view.start_live(rid, _PID_A)
        assert view.live_portfolio_id == _PID_A  # pinned so a barge-in targets this run
        await pilot.pause()
        for event in _scripted_stream(rid):
            view.ingest_event(event)
            await pilot.pause()
        live = view.query_one("#delib-live", DataTable)
        assert live.display is True
        assert live.row_count == 3  # the two steps + the terminal gate row
        # Each new event flips the prior LIVE step ●→✓; the terminal row stays ·.
        assert live.get_cell_at(Coordinate(0, 0)) == StepStatus.DONE.glyph
        assert live.get_cell_at(Coordinate(1, 0)) == StepStatus.DONE.glyph
        assert live.get_cell_at(Coordinate(2, 0)) == StepStatus.PENDING.glyph
        assert "awaiting the execution gate" in view.live_text


@pytest.mark.asyncio
async def test_deliberation_live_follow_then_inspect_then_resume(shell_env) -> None:
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        rid = uuid.uuid4()
        view.start_live(rid, _PID_A)
        for event in _scripted_stream(rid):
            view.ingest_event(event)
            await pilot.pause()
        assert view.following is True  # auto-follow by default
        live = view.query_one("#delib-live", DataTable)
        live.focus()
        live.move_cursor(row=1)  # inspect an earlier step → pause follow
        await pilot.pause()
        assert view.following is False
        view.action_follow_live()  # f → re-snap to the tail
        await pilot.pause()
        assert view.following is True


@pytest.mark.asyncio
async def test_deliberation_live_ends_back_to_replay(shell_env) -> None:
    with shell_env.get_session() as session:
        _seed_run_with_steps(session)
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        rid = uuid.uuid4()
        view.start_live(rid, _PID_A)
        view.ingest_event(_scripted_stream(rid)[0])
        await pilot.pause()
        assert view.query_one("#delib-live", DataTable).display is True
        assert view.query_one("#delib-runs", DataTable).display is False
        view.end_live()
        await pilot.pause()
        assert view.live_run_id is None
        # Replay is restored: the run picker returns, the live table is hidden.
        assert view.query_one("#delib-live", DataTable).display is False
        assert view.query_one("#delib-runs", DataTable).display is True
        assert view.query_one("#delib-runs", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_deliberation_live_drops_events_of_another_run(shell_env) -> None:
    """A superseded stream still draining its thread tags its events with its own
    run id and reuses the live run's step indexes; they must not enter the live
    timeline (a colliding row key aborts the stream that owns it)."""
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        rid, stale = uuid.uuid4(), uuid.uuid4()
        view.start_live(rid, _PID_A)
        events = _scripted_stream(rid)
        view.ingest_event(events[0])
        for event in _scripted_stream(stale)[:2]:
            view.ingest_event(event)
        view.ingest_event(events[1])
        await pilot.pause()
        live = view.query_one("#delib-live", DataTable)
        assert live.row_count == 2
        assert live.get_cell_at(Coordinate(1, 2)) == "report"
        assert view.live_run_id == rid


def test_deliberation_binds_follow_live() -> None:
    """``f`` is view-scoped to Deliberation (follow-live), not the global filter."""
    actions = {binding[1] for binding in DeliberationView.BINDINGS}
    assert "follow_live" in actions


@pytest.mark.asyncio
async def test_run_routes_stream_on_deliberation_blocking_on_posture(tui_env) -> None:
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    calls: list[str] = []
    app._stream_worker = lambda *_a, **_k: calls.append("stream")  # type: ignore[method-assign]
    app._run_worker = lambda *_a, **_k: calls.append("run")  # type: ignore[method-assign]
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        assert calls[-1] == "run"  # Posture → blocking _run_worker (unchanged)
        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        assert calls[-1] == "stream"  # Deliberation → live _stream_worker


@pytest.mark.asyncio
async def test_run_key_leaves_the_decision_bar_for_the_run_to_resolve(tui_env) -> None:
    """``u`` passes no bar: no close is stored after today, so nothing could fill."""
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    bars: list[object] = []
    app._run_worker = lambda _pid, asof: bars.append(asof)  # type: ignore[method-assign]
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        app.action_run_portfolio()
        await pilot.pause()

    assert bars == [None]


def _pin_wall_clock_to_asof(monkeypatch) -> None:
    """Make the unparameterised resolver read the seeded panel as of ``ASOF + 1d``.

    The real ``default_decision_bar`` runs over the seeded prices (daily bars through
    ``ASOF + 9d``); bounding it at ``ASOF + 1d`` makes it answer ``ASOF`` -- a date
    that differs from ``dt.date.today()``, so a path that skipped the resolver and
    fell back to the wall clock cannot pass for it.
    """
    real = observe.default_decision_bar
    monkeypatch.setattr(
        observe,
        "default_decision_bar",
        lambda session: real(session, today=ASOF + timedelta(days=1)),
    )


def _spy_decision_bar(monkeypatch, name: str) -> list[date]:
    """Record the ``asof`` each call to ``fund.agents.graph.<name>`` receives."""
    from fund.agents import graph

    real = getattr(graph, name)
    seen: list[date] = []

    def spy(*args, **kwargs):
        seen.append(kwargs["asof"])
        return real(*args, **kwargs)

    monkeypatch.setattr(graph, name, spy)
    return seen


def _persist_mandate(env) -> None:
    with env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()


@pytest.mark.asyncio
async def test_deliberation_u_resolves_the_stream_bar_from_stored_prices(
    tui_env, monkeypatch
) -> None:
    """``u`` on Deliberation streams at the stored fillable bar, not today's date."""
    _persist_mandate(tui_env)
    _pin_wall_clock_to_asof(monkeypatch)
    seen = _spy_decision_bar(monkeypatch, "stream_fund")
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_run_portfolio()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        assert worker.group == "stream"
        await worker.wait()
        await pilot.pause()

    assert seen == [ASOF]


@pytest.mark.asyncio
async def test_deliberation_explicit_bar_bypasses_the_resolver(
    tui_env, monkeypatch
) -> None:
    """A caller-supplied bar is streamed as given; the resolver is never consulted."""
    _persist_mandate(tui_env)
    consulted: list[object] = []

    def _resolver(session):
        consulted.append(session)
        return date(2031, 1, 2)

    monkeypatch.setattr(observe, "default_decision_bar", _resolver)
    seen = _spy_decision_bar(monkeypatch, "stream_fund")
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        assert app.last_worker is not None
        await app.last_worker.wait()
        await pilot.pause()

    assert seen == [ASOF]
    assert consulted == []


@pytest.mark.asyncio
async def test_posture_u_runs_at_the_stored_fillable_bar(tui_env, monkeypatch) -> None:
    """``u`` on Posture hands ``run_fund`` the resolved bar via the real worker."""
    _persist_mandate(tui_env)
    _pin_wall_clock_to_asof(monkeypatch)
    seen = _spy_decision_bar(monkeypatch, "run_fund")
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        app.action_run_portfolio()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        assert worker.group == "drive"
        await worker.wait()
        await pilot.pause()
        assert "awaiting approval" in app.last_status

    assert seen == [ASOF]


@pytest.mark.asyncio
async def test_ask_run_runs_at_the_stored_fillable_bar(tui_env, monkeypatch) -> None:
    """A confirmed Ask 'run' reaches ``run_fund`` with the resolved bar, not today."""
    _persist_mandate(tui_env)
    _pin_wall_clock_to_asof(monkeypatch)
    seen = _spy_decision_bar(monkeypatch, "run_fund")
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("run a rebalance")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        assert worker.group == "drive"
        await worker.wait()
        await pilot.pause()
        assert "awaiting approval" in app.last_status

    assert seen == [ASOF]


@pytest.mark.asyncio
async def test_deliberation_u_streams_live_round_to_gate(tui_env) -> None:
    # Persist a mandate so the stream worker's observe.get_mandate finds it.
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_run_portfolio(asof=ASOF)  # u from Deliberation → live stream
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        assert worker.group == "stream"
        await worker.wait()
        await pilot.pause()
        view = app.query_one("#view-deliberation", DeliberationView)
        live = view.query_one("#delib-live", DataTable)
        # The scripted round streamed its steps, ending at the place_orders gate; the
        # paused timeline stays up (the adviser just watched it build).
        assert live.display is True
        assert live.row_count >= 1
        assert live.get_cell_at(Coordinate(live.row_count - 1, 2)) == "place_orders"
        streamed_run_id = view.live_run_id
        assert streamed_run_id is not None
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(streamed_run_id)
        assert run.status == "paused"  # the run reached its HITL pause


@pytest.mark.asyncio
async def test_deliberation_leaves_live_mode_once_the_gate_is_resolved(tui_env) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        streamed = view.live_run_id
        assert streamed is not None  # the timeline stands at the gate

        decisions = await _open_decisions(app, pilot)
        decisions.select_run(streamed)
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()

        view = await _open_deliberation(app, pilot)
        assert view.live_run_id is None  # the round is over: back to replay
        assert view.query_one("#delib-runs", DataTable).display is True
        assert view.query_one("#delib-steps", DataTable).display is True
        assert view.query_one("#delib-live", DataTable).display is False
        app.action_barge_in()  # nothing live to steer any more
        await pilot.pause()
        assert not isinstance(app.screen, BargeInScreen)


# --- Run launch is single-flight ---------------------------------------------


class _HeldRound:
    """A rebalance stand-in that records each launch and blocks until released."""

    def __init__(self) -> None:
        self.launches: list[str] = []
        self.release = threading.Event()

    def enter(self, label: str) -> None:
        self.launches.append(label)
        self.release.wait(timeout=10)

    async def until_launched(self, pilot, count: int = 1) -> None:
        for _ in range(200):
            if len(self.launches) >= count:
                return
            await pilot.pause(0.05)
        raise AssertionError(f"expected {count} launches, saw {self.launches}")


def _hold_rounds(monkeypatch) -> _HeldRound:
    """Replace the stream and the blocking-run entry points with held stand-ins."""
    held = _HeldRound()

    def fake_stream_fund(*_args, **_kwargs):
        held.enter("stream")
        return
        yield  # a generator that ends without a step

    def fake_apply_action(action, **_kwargs):
        held.enter(action)
        return SimpleNamespace(message="run done")

    monkeypatch.setattr("fund.agents.graph.stream_fund", fake_stream_fund)
    monkeypatch.setattr("fund.actions.apply_action", fake_apply_action)
    return held


def _seed_mandate(env) -> None:
    with env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()


@pytest.mark.asyncio
async def test_second_u_is_refused_while_a_stream_is_in_flight(
    tui_env, monkeypatch
) -> None:
    """The live timeline stays empty until the first agent reports, so ``u`` is easy
    to press twice; a second concurrent round would double the LLM spend and open a
    competing gate."""
    _seed_mandate(tui_env)
    held = _hold_rounds(monkeypatch)
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_run_portfolio(asof=ASOF)
        await held.until_launched(pilot)
        first = app.last_worker

        app.action_run_portfolio(asof=ASOF)
        await pilot.pause(0.2)
        assert "already in progress" in app.last_status
        assert app.last_worker is first
        assert held.launches == ["stream"]

        held.release.set()
        await first.wait()
        await pilot.pause()

        app.action_run_portfolio(asof=ASOF)  # the guard clears once the round ends
        await app.last_worker.wait()
        assert held.launches == ["stream", "stream"]


@pytest.mark.asyncio
async def test_run_in_flight_from_posture_refuses_deliberation_and_ask_runs(
    tui_env, monkeypatch
) -> None:
    """The blocking and streamed runs sit in different worker groups, so only a
    shared guard stops one from starting while the other still runs."""
    _seed_mandate(tui_env)
    held = _hold_rounds(monkeypatch)
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await held.until_launched(pilot)
        first = app.last_worker

        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause(0.2)
        assert "already in progress" in app.last_status

        ask_view = SimpleNamespace(append_log=lambda *_a, **_k: None)
        ask_run = ResolvedAction("run", None, PORTFOLIO_ID, None)
        assert app._dispatch_ask_action(ask_view, ask_run) is None
        await pilot.pause(0.2)
        assert held.launches == ["run"]

        held.release.set()
        await first.wait()


@pytest.mark.asyncio
async def test_deliberation_u_stream_appends_a_run_audit_row(tui_env) -> None:
    """A streamed round leaves the same human ``run`` audit row a blocking one does."""
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        streamed_run_id = view.live_run_id
        assert streamed_run_id is not None
    with tui_env.get_session() as session:
        rows = HumanAuditLogRepository(session).list_recent(action="run")
        assert [(row.run_id, row.target, row.actor) for row in rows] == [
            (streamed_run_id, "rebalance", "human")
        ]
        assert rows[0].portfolio_id == PORTFOLIO_ID


def _held_stream(calls: list[uuid.UUID], release: threading.Event):
    """A ``stream_fund`` stand-in: one live step, then blocked until ``release``.

    Records every portfolio it is started for, so a test can count how many rounds
    the shell launched; the held step keeps the round in flight (no committed row yet).
    """

    def _stream(model, mandate, *, portfolio_id, asof, **_kwargs):
        calls.append(portfolio_id)
        run_id = uuid.uuid4()
        yield StreamEvent(
            kind="step",
            run_id=run_id,
            index=0,
            agent="economist",
            step="report",
            status=StepStatus.LIVE,
            text="macro read",
        )
        release.wait(timeout=10)
        yield StreamEvent(
            kind="incomplete",
            run_id=run_id,
            index=1,
            agent="orchestrator",
            step="place_orders",
            status=StepStatus.PENDING,
            text="pipeline ended before the gate",
        )

    return _stream


async def _wait_on_loop(predicate, *, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` on the loop until it holds or ``timeout`` elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.mark.asyncio
async def test_run_refused_while_a_round_is_already_in_flight(
    tui_env, monkeypatch
) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    calls: list[uuid.UUID] = []
    release = threading.Event()
    monkeypatch.setattr("fund.agents.graph.stream_fund", _held_stream(calls, release))
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    blocking_runs: list[str] = []
    app._run_worker = lambda *_a, **_k: blocking_runs.append("run")  # type: ignore[method-assign]
    try:
        async with app.run_test(size=_SIZE) as pilot:
            app.query_one(ContentSwitcher).current = "view-deliberation"
            app._refresh()
            await pilot.pause()
            app.action_run_portfolio(asof=ASOF)  # first u: the round starts streaming
            assert await _wait_on_loop(lambda: len(calls) == 1)
            app.action_run_portfolio(asof=ASOF)  # impatient second u
            await pilot.pause(0.3)
            # One round only: a second thread would interleave into the same live
            # table and race the first for the run / gate rows.
            assert len(calls) == 1
            assert "already in progress" in app.last_status
            # The blocking Posture launchpad is refused too while a round streams.
            await _open_posture(app, pilot)
            app.action_run_portfolio(asof=ASOF)
            await pilot.pause()
            assert blocking_runs == []
            assert "already in progress" in app.last_status
            release.set()
            assert await _wait_on_loop(
                lambda: not any(w.group == "stream" for w in app.workers)
            )
    finally:
        release.set()


@pytest.mark.asyncio
async def test_ask_run_refused_while_a_round_is_already_in_flight(
    tui_env, monkeypatch
) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    calls: list[uuid.UUID] = []
    release = threading.Event()
    monkeypatch.setattr("fund.agents.graph.stream_fund", _held_stream(calls, release))
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    blocking_runs: list[str] = []
    app._run_worker = lambda *_a, **_k: blocking_runs.append("run")  # type: ignore[method-assign]
    try:
        async with app.run_test(size=_SIZE) as pilot:
            app.query_one(ContentSwitcher).current = "view-deliberation"
            app._refresh()
            await pilot.pause()
            app.action_run_portfolio(asof=ASOF)  # a round is streaming
            assert await _wait_on_loop(lambda: len(calls) == 1)
            view = await _open_ask(app, pilot)
            view.submit_turn("run a rebalance")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.click("#confirm-yes")
            await pilot.pause(0.3)
            # The Ask verb is a second entry point to the same round: it must not open
            # a second run / gate for the bar next to the one still streaming.
            assert blocking_runs == []
            assert len(calls) == 1
            assert app.last_worker is None  # nothing dispatched
            assert "already in progress" in app.last_status
            release.set()
            assert await _wait_on_loop(
                lambda: not any(w.group == "stream" for w in app.workers)
            )
    finally:
        release.set()


@pytest.mark.asyncio
async def test_freeze_mid_stream_leaves_no_approvable_gate(
    shell_env, monkeypatch
) -> None:
    """A freeze pressed while a round streams is not outrun by the gate it reaches.

    The freeze rejects committed paused gates only, and the in-flight round's run row
    is not one of them; without a re-check at the gate the round would commit a
    ``paused`` run the adviser can still approve on a frozen portfolio.
    """
    from fund.actions import apply_action

    with shell_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    proceed = threading.Event()
    run_ids: list[uuid.UUID] = []

    def _stream(model, mandate, *, portfolio_id, asof, session, **_kwargs):
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=portfolio_id,
            asof=asof,
            seed=1,
            universe=[],
            optimizer_config={},
        )
        run_ids.append(run.id)
        yield StreamEvent(
            kind="step",
            run_id=run.id,
            index=0,
            agent="economist",
            step="report",
            status=StepStatus.LIVE,
            text="macro read",
        )
        proceed.wait(timeout=10)
        repo.mark_paused(run.id)
        yield StreamEvent(
            kind="paused",
            run_id=run.id,
            index=1,
            agent="orchestrator",
            step="place_orders",
            status=StepStatus.PENDING,
            text="awaiting the execution gate",
            payload={"action_requests": []},
        )

    monkeypatch.setattr("fund.agents.graph.stream_fund", _stream)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    try:
        async with app.run_test(size=_SIZE) as pilot:
            view = await _open_deliberation(app, pilot)
            app.action_run_portfolio(asof=ASOF)
            assert await _wait_on_loop(lambda: view.live_run_id is not None)
            worker = app.last_worker
            assert worker is not None
            with shell_env.get_session() as session:  # the adviser presses `k`
                apply_action(
                    "freeze",
                    session=session,
                    scope="portfolio",
                    portfolio_id=PORTFOLIO_ID,
                    rationale="drawdown breach",
                )
                session.commit()
            proceed.set()
            await worker.wait()
            await pilot.pause()
            assert "frozen" in app.last_status
            assert view.live_run_id is None  # nothing left to follow at a halted gate
    finally:
        proceed.set()
    with shell_env.get_session() as session:
        repo = AgentRunRepository(session)
        assert repo.get_run(run_ids[0]).status == "rejected"
        assert repo.list_paused_runs(PORTFOLIO_ID) == []


@pytest.mark.asyncio
async def test_stream_ending_incomplete_reports_why_on_the_status_bar(
    tui_env, monkeypatch
) -> None:
    """A round that ends with no ticket drops to replay but says why it ended."""
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    reason = "delegation round cap reached before a ticket"

    def _stream(model, mandate, *, portfolio_id, asof, session, **_kwargs):
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=portfolio_id,
            asof=asof,
            seed=1,
            universe=[],
            optimizer_config={},
        )
        repo.finalize_run(run.id, weights={}, status="incomplete")
        yield StreamEvent(
            kind="step",
            run_id=run.id,
            index=0,
            agent="economist",
            step="report",
            status=StepStatus.LIVE,
            text="macro read",
        )
        yield StreamEvent(
            kind="incomplete",
            run_id=run.id,
            index=1,
            agent="orchestrator",
            step="place_orders",
            status=StepStatus.PENDING,
            text=reason,
        )

    monkeypatch.setattr("fund.agents.graph.stream_fund", _stream)
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert view.live_run_id is None  # nothing to act on: back to replay
        assert reason in app.last_status
        assert reason in str(app.query_one("#status-bar", Static).render())


@pytest.mark.asyncio
async def test_quit_with_barge_in_open_releases_the_stream_thread(
    shell_env, monkeypatch
) -> None:
    """Quitting while a barge-in holds the stream must not strand its worker thread.

    The overlay's resolve callback is what re-arms the stream, and it never runs when
    the app quits with the overlay open. A thread worker cannot be killed and executor
    threads are joined at exit, so a hold that only the overlay can release would leave
    the process alive after the UI is gone.
    """
    with shell_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    resumed = threading.Event()
    proceed = threading.Event()
    stream_closed = threading.Event()

    def _stream(model, mandate, *, portfolio_id, asof, **_kwargs):
        run_id = uuid.uuid4()
        try:
            for index in range(3):
                if index == 1:
                    resumed.set()  # the worker cleared the first step boundary
                    proceed.wait(timeout=10)
                yield StreamEvent(
                    kind="step",
                    run_id=run_id,
                    index=index,
                    agent="economist",
                    step="report",
                    status=StepStatus.LIVE,
                    text=f"step {index}",
                )
        finally:
            stream_closed.set()

    monkeypatch.setattr("fund.agents.graph.stream_fund", _stream)
    app = _make_shell(shell_env, portfolio_id=PORTFOLIO_ID)
    try:
        async with app.run_test(size=_SIZE) as pilot:
            view = await _open_deliberation(app, pilot)
            app.action_run_portfolio(asof=ASOF)
            assert await _wait_on_loop(resumed.is_set)
            app.action_barge_in()  # hold the stream at its next step boundary
            await pilot.pause()
            assert isinstance(app.screen, BargeInScreen)
            proceed.set()
            live = view.query_one("#delib-live", DataTable)
            assert await _wait_on_loop(lambda: live.row_count == 2)
            await pilot.press("ctrl+q")  # a priority binding: fires under the overlay
            await pilot.pause()
        assert await asyncio.to_thread(stream_closed.wait, 5.0), (
            "the stream worker is still blocked after the app quit"
        )
    finally:
        proceed.set()
        app._stream_resume.set()  # never leave the blocked thread to hang pytest


@pytest.mark.asyncio
async def test_deliberation_leaves_live_mode_once_the_gate_is_decided(tui_env) -> None:
    """A streamed round's timeline stands at its gate, then yields to replay.

    While the run still awaits approval the tail stays up (barge-in needs it); once the
    adviser approves, the view must not keep claiming LIVE for a finalised run.
    """
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        run_id = view.live_run_id
        assert run_id is not None
        app._refresh()  # a poll tick while the gate still stands keeps the tail
        assert view.live_run_id == run_id

        decisions = await _open_decisions(app, pilot)
        decisions.select_run(run_id)
        await pilot.pause()
        app.action_decide_approve()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert "approved" in app.last_status

        await _open_deliberation(app, pilot)  # the adviser returns to Deliberation
        assert view.live_run_id is None
        assert view.live_portfolio_id is None  # barge-in has no finalised run to target
        assert view.query_one("#delib-live", DataTable).display is False
        assert view.query_one("#delib-runs", DataTable).display is True
        assert view.selected_run_id == run_id  # the run just watched stays in view


@pytest.mark.asyncio
async def test_deliberation_drops_a_finished_tail_when_focus_moves_away(
    tui_env,
) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert view.live_run_id is not None  # standing at the gate for PORTFOLIO_ID

        app.focus_portfolio(_PID_B)
        await pilot.pause()

        assert view.live_run_id is None  # another portfolio's tail must not linger
        assert view.selected_run_id is None  # and its run is not carried over


@pytest.mark.asyncio
async def test_deliberation_keeps_the_tail_while_its_run_is_still_in_flight(
    tui_env,
) -> None:
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        with tui_env.get_session() as session:
            run = AgentRunRepository(session).create_run(
                portfolio_id=PORTFOLIO_ID,
                asof=ASOF,
                seed=1,
                universe=[],
                optimizer_config={},
            )
            session.commit()
            run_id = run.id
        assert app._streaming is False  # the window before the stream latches LIVE
        view.start_live(run_id, PORTFOLIO_ID)

        app._refresh()  # a poll tick lands while the run row reads as pending

        assert view.live_run_id == run_id


@pytest.mark.asyncio
async def test_barge_in_is_inert_once_the_live_run_was_decided_elsewhere(
    tui_env,
) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        run_id = view.live_run_id
        assert run_id is not None
        with tui_env.get_session() as session:  # decided by another session
            AgentRunRepository(session).get_run(run_id).status = "completed"
            session.commit()

        app.action_barge_in()  # before any poll tick repainted the view
        await pilot.pause()

        assert not isinstance(app.screen, BargeInScreen)
        assert view.live_run_id is None
        assert "live Deliberation" in app.last_status


# --- Barge-in: PM directive overlay + `b` hotkey (T8) -------------------------


def test_import_bargein_is_light() -> None:
    code = (
        "import sys, fund.tui.barge_in; "
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


@pytest.mark.asyncio
async def test_bargein_screen_sends_directive_and_remember_flag(shell_env) -> None:
    app = _make_shell(shell_env)
    results: list[BargeInResult | None] = []
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app.push_screen(BargeInScreen("economist · report · #0"), results.append)
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, BargeInScreen)
        screen.query_one("#bargein-text", Input).value = "Trim AAA before committing."
        screen.query_one("#bargein-remember", Checkbox).value = True
        screen._submit()
        await pilot.pause()
        assert results[-1] == BargeInResult(
            text="Trim AAA before committing.", remember=True
        )
        # An empty directive is a no-op: dismiss ``None`` (resume unchanged).
        app.push_screen(BargeInScreen("gate"), results.append)
        await pilot.pause()
        app.screen._submit()
        await pilot.pause()
        assert results[-1] is None


@pytest.mark.asyncio
async def test_barge_in_inert_without_a_live_round(tui_env) -> None:
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)  # not Deliberation → inert
        app.action_barge_in()
        await pilot.pause()
        assert app.last_worker is None
        assert not isinstance(app.screen, BargeInScreen)
        assert "live Deliberation" in app.last_status
        # On Deliberation but with no live round → still inert (nothing to barge into).
        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_barge_in()
        await pilot.pause()
        assert not isinstance(app.screen, BargeInScreen)


@pytest.mark.asyncio
async def test_barge_in_at_gate_responds_and_regates(tui_env) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(
        tui_env,
        portfolio_id=PORTFOLIO_ID,
        model_factory=lambda: ScriptedFundModel(
            universe=UNIVERSE, weights=tui_env.expected, max_orders=2
        ),
    )
    async with app.run_test(size=_SIZE) as pilot:
        app.query_one(ContentSwitcher).current = "view-deliberation"
        app._refresh()
        await pilot.pause()
        app.action_run_portfolio(asof=ASOF)  # stream the round to the gate
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        view = app.query_one("#view-deliberation", DeliberationView)
        run_id = view.live_run_id
        assert run_id is not None
        assert app._stream_resume.is_set()  # not paused — it reached the gate

        app.action_barge_in()  # b
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, BargeInScreen)
        assert "place_orders" in screen._paused_at  # overlay shows where it paused
        assert (
            not app._stream_resume.is_set()
        )  # the live stream is held during barge-in
        screen.query_one("#bargein-text", Input).value = "Trim AAA before committing."
        screen._submit()
        await pilot.pause()
        assert app._stream_resume.is_set()  # resumed on resolve
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "responded" in app.last_status
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(run_id)
        assert run.status == "responded"  # alive, still awaiting approval
        rows = HumanAuditLogRepository(session).list_recent(action="respond")
        assert len(rows) == 1
        assert rows[0].run_id == run_id


def _stream_held_after_first_step(
    monkeypatch, run_id: uuid.UUID, *, reaches_gate: bool
) -> threading.Event:
    """Stand in for ``stream_fund``: one step, then block until the returned event."""
    release = threading.Event()
    events = _scripted_stream(run_id)

    def fake_stream_fund(*_args, **_kwargs):
        yield events[0]
        release.wait(timeout=10)
        yield from events[1:] if reaches_gate else events[1:2]

    monkeypatch.setattr("fund.agents.graph.stream_fund", fake_stream_fund)
    return release


def _record_actions(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_apply_action(action, **kwargs):
        calls.append({"action": action, **kwargs})
        return SimpleNamespace(message=f"{action} done")

    monkeypatch.setattr("fund.actions.apply_action", fake_apply_action)
    return calls


async def _send_barge_in_mid_stream(app, pilot, run_id: uuid.UUID, directive: str):
    view = app.query_one("#view-deliberation", DeliberationView)
    for _ in range(200):
        if view.live_run_id == run_id:
            break
        await pilot.pause(0.05)
    assert view.live_run_id == run_id
    app.action_barge_in()
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, BargeInScreen)
    screen.query_one("#bargein-text", Input).value = directive
    screen._submit()
    await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_barge_in_mid_stream_is_delivered_after_the_gate_commits(
    tui_env, monkeypatch
) -> None:
    """The run row is only flushed until the stream worker commits at the end of the
    round, so a ``respond`` dispatched mid-stream cannot see it (``no agent_run``) and
    the directive is lost; it has to wait for the committed gate."""
    _seed_mandate(tui_env)
    run_id = uuid.uuid4()
    release = _stream_held_after_first_step(monkeypatch, run_id, reaches_gate=True)
    calls = _record_actions(monkeypatch)
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        stream = app.last_worker
        try:
            await _send_barge_in_mid_stream(app, pilot, run_id, "Trim AAA.")
            assert calls == []
            assert "queued" in app.last_status
        finally:
            release.set()
        await stream.wait()
        await pilot.pause()
        await app.last_worker.wait()
        assert [c["action"] for c in calls] == ["respond"]
        assert calls[0]["run_id"] == run_id
        assert calls[0]["portfolio_id"] == PORTFOLIO_ID
        assert calls[0]["pm_directive"] == "Trim AAA."


@pytest.mark.asyncio
async def test_barge_in_queued_mid_stream_is_dropped_when_no_gate_is_reached(
    tui_env, monkeypatch
) -> None:
    _seed_mandate(tui_env)
    run_id = uuid.uuid4()
    release = _stream_held_after_first_step(monkeypatch, run_id, reaches_gate=False)
    calls = _record_actions(monkeypatch)
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        stream = app.last_worker
        try:
            await _send_barge_in_mid_stream(app, pilot, run_id, "Trim AAA.")
            assert calls == []
        finally:
            release.set()
        await stream.wait()
        await pilot.pause()
        assert calls == []
        assert "dropped" in app.last_status


@pytest.mark.asyncio
async def test_barge_in_queued_mid_stream_keeps_the_stream_failure_reason(
    tui_env, monkeypatch
) -> None:
    """A stream that dies drops the queued directive, but the one-line status must
    still say why the run failed, not only that the directive was dropped."""
    _seed_mandate(tui_env)
    run_id = uuid.uuid4()
    release = threading.Event()
    events = _scripted_stream(run_id)

    def failing_stream_fund(*_args, **_kwargs):
        yield events[0]
        release.wait(timeout=10)
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("fund.agents.graph.stream_fund", failing_stream_fund)
    calls = _record_actions(monkeypatch)
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        stream = app.last_worker
        try:
            await _send_barge_in_mid_stream(app, pilot, run_id, "Trim AAA.")
        finally:
            release.set()
        await stream.wait()
        await pilot.pause()
        assert calls == []
        assert "provider exploded" in app.last_status
        assert "dropped" in app.last_status


@pytest.mark.asyncio
async def test_second_barge_in_mid_stream_says_it_replaces_the_queued_one(
    tui_env, monkeypatch
) -> None:
    """Only one directive can wait for the gate, so a second one replaces the first;
    the status says so and only the latest text is delivered."""
    _seed_mandate(tui_env)
    run_id = uuid.uuid4()
    release = _stream_held_after_first_step(monkeypatch, run_id, reaches_gate=True)
    calls = _record_actions(monkeypatch)
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_deliberation(app, pilot)
        app.action_run_portfolio(asof=ASOF)
        stream = app.last_worker
        try:
            await _send_barge_in_mid_stream(app, pilot, run_id, "Trim AAA.")
            assert "replac" not in app.last_status
            await _send_barge_in_mid_stream(app, pilot, run_id, "Keep 5% cash.")
            assert "replac" in app.last_status
        finally:
            release.set()
        await stream.wait()
        await pilot.pause()
        await app.last_worker.wait()
        assert [c["pm_directive"] for c in calls] == ["Keep 5% cash."]


# --- Decisions e/edit + s/steer (view-scoped bindings) (T9) -------------------


@pytest.mark.asyncio
async def test_decisions_edit_commits_edited_weights(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_edit_decision()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, OrderEditScreen)
        screen.query_one("#edit-weights", Input).value = "AAA:0.5, BBB:0.5"
        screen._submit()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "edited" in app.last_status
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "completed"
        assert run.weights == {
            "AAA": 0.5,
            "BBB": 0.5,
        }  # the human override, not the LLM
        rows = HumanAuditLogRepository(session).list_recent(action="edit")
        assert len(rows) == 1
        assert rows[0].override_flag is True


@pytest.mark.asyncio
async def test_decisions_e_binding_is_edit_not_export(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        table = view.query_one("#decisions-table", DataTable)
        table.focus()
        await pilot.pause()
        await pilot.press("e")  # view-scoped → edit, not the global Audit export
        await pilot.pause()
        assert isinstance(app.screen, OrderEditScreen)


@pytest.mark.asyncio
async def test_decisions_steer_responds_and_regates(tui_env) -> None:
    app = _make_shell(
        tui_env,
        model_factory=lambda: ScriptedFundModel(
            universe=UNIVERSE, weights=tui_env.expected, max_orders=2
        ),
    )
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_steer_decision()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, BargeInScreen)
        screen.query_one("#bargein-text", Input).value = "Trim AAA before committing."
        screen._submit()
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "responded" in app.last_status
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "responded"


@pytest.mark.asyncio
async def test_decisions_edit_steer_inert_without_selection(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_decisions(app, pilot)  # nothing selected
        app.action_edit_decision()
        await pilot.pause()
        assert not isinstance(app.screen, OrderEditScreen)
        assert "select a paused gate" in app.last_status
        app.action_steer_decision()
        await pilot.pause()
        assert not isinstance(app.screen, BargeInScreen)


@pytest.mark.asyncio
async def test_decisions_edit_error_shows_typed_brackets_verbatim(tui_env) -> None:
    # _parse_weights echoes the adviser's text via !r; a markup-parsing error line
    # rejects "[/x]" (MarkupError) and drops "[AAPL]", killing the weight editor.
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_edit_decision()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, OrderEditScreen)
        for typed in ("[/x]", "[AAPL]"):
            screen.query_one("#edit-weights", Input).value = typed
            screen._submit()
            await pilot.pause()
            assert isinstance(app.screen, OrderEditScreen)  # editor stays open
            shown = str(screen.query_one("#edit-error", Static).render())
            assert typed in shown


def test_parse_weights_rejects_garbage_and_parses_pairs() -> None:
    from fund.tui.app import _parse_weights

    assert _parse_weights("AAA:0.6, BBB:0.4") == {"AAA": 0.6, "BBB": 0.4}
    with pytest.raises(ValueError, match="ticker:weight"):
        _parse_weights("AAA 0.6")
    with pytest.raises(ValueError, match="weight"):
        _parse_weights("AAA:oops")


def test_parse_weights_resolves_tickers_against_the_proposed_tickers() -> None:
    from fund.tui.app import _parse_weights

    allowed = {"AAA", "BBB", "CCC"}
    # A lowercase entry resolves to the canonical ticker instead of a phantom name.
    assert _parse_weights("aaa:0.6, Bbb:0.4", allowed=allowed) == {
        "AAA": 0.6,
        "BBB": 0.4,
    }
    with pytest.raises(
        ValueError, match="unknown ticker 'ZZZ'; not among the proposed tickers"
    ):
        _parse_weights("AAA:0.8, ZZZ:0.2", allowed=allowed)
    # The same name twice would let the second weight silently replace the first.
    with pytest.raises(ValueError, match="duplicate ticker 'AAA'"):
        _parse_weights("AAA:0.5, aaa:0.5", allowed=allowed)


@pytest.mark.asyncio
async def test_order_edit_prefill_committed_unchanged_stays_within_budget(
    shell_env,
) -> None:
    # Thirds are the worst case for a rounded pre-fill: 3 x 0.3333 sums to 0.9999,
    # outside risk_check's 1e-6 budget tolerance.
    weights = {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}
    committed: list[dict[str, float] | None] = []
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        app.push_screen(OrderEditScreen(weights), committed.append)
        await pilot.pause()
        app.screen._submit()
        await pilot.pause()
    assert committed and committed[0] is not None
    assert abs(sum(committed[0].values()) - 1.0) <= 1e-6


@pytest.mark.asyncio
async def test_decisions_edit_prefills_the_audited_optimizer_weights(tui_env) -> None:
    # The gate shows the model's own place_orders args, which may be rounded; the
    # ticket is placed from the audited optimizer output, so that is what to edit.
    from fund.tui.app import _parse_weights

    audited = {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}
    with tui_env.get_session() as session:
        AgentRunRepository(session).append_decision(
            tui_env.run_id,
            agent="allocator",
            step="optimize_portfolio",
            llm_response=json.dumps({"weights": audited}),
        )
        session.commit()
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_edit_decision()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, OrderEditScreen)
        seeded = _parse_weights(screen.query_one("#edit-weights", Input).value)
    assert seeded.keys() == audited.keys()
    assert all(abs(seeded[k] - audited[k]) <= 1e-9 for k in audited)


@pytest.mark.asyncio
async def test_decisions_edit_with_unknown_ticker_keeps_editor_open(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_edit_decision()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, OrderEditScreen)
        # Sums to 1 and is long-only, so risk_check alone would wave it through.
        screen.query_one("#edit-weights", Input).value = "AAA:0.5, BBB:0.3, ZZZ:0.2"
        screen._submit()
        await pilot.pause()
        assert app.screen is screen  # the editor stays open on the typo
        error = str(screen.query_one("#edit-error", Static).render())
        assert "ZZZ" in error
        assert "proposed tickers" in error
        assert app.last_worker is None  # nothing was dispatched
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "paused"


# --- Remember persistence: barge-in/steer remember flag (T11) -----------------


@pytest.mark.asyncio
async def test_steer_remember_persists_directive_to_mandate(tui_env) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(
        tui_env,
        portfolio_id=PORTFOLIO_ID,
        model_factory=lambda: ScriptedFundModel(
            universe=UNIVERSE, weights=tui_env.expected, max_orders=2
        ),
    )
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_steer_decision()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, BargeInScreen)
        screen.query_one("#bargein-text", Input).value = "Always keep 5% cash."
        screen.query_one("#bargein-remember", Checkbox).value = True  # remember
        screen._submit()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        # It shows in the (focused) Mandate list, where it is removable (T10).
        mandate_view = await _open_mandate(app, pilot)
        assert "Always keep 5% cash." in mandate_view._directives
    with tui_env.get_session() as session:
        mandate = observe.get_mandate(session, PORTFOLIO_ID)
        assert "Always keep 5% cash." in mandate.remembered_directives
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "responded"  # the directive still steered this round


@pytest.mark.asyncio
async def test_steer_one_shot_directive_is_not_persisted(tui_env) -> None:
    with tui_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    app = _make_shell(
        tui_env,
        portfolio_id=PORTFOLIO_ID,
        model_factory=lambda: ScriptedFundModel(
            universe=UNIVERSE, weights=tui_env.expected, max_orders=2
        ),
    )
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_steer_decision()
        await pilot.pause()
        screen = app.screen
        screen.query_one("#bargein-text", Input).value = "Just this once."
        # remember left unchecked (one-shot)
        screen._submit()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
    with tui_env.get_session() as session:
        mandate = observe.get_mandate(session, PORTFOLIO_ID)
        assert mandate.remembered_directives == ()  # nothing persisted
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "responded"  # but the round was still steered


# --- Autonomy TUI: killswitch scope + re-arm + ARMED/FROZEN header (T12) ------


@pytest.mark.asyncio
async def test_killswitch_portfolio_scope_freezes(tui_env) -> None:
    from fund.audit.autonomy_repository import AutonomyRepository

    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_killswitch()
        await pilot.pause()
        assert isinstance(app.screen, ScopePickScreen)
        await pilot.click("#scope-portfolio")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert str(app.query_one("#hdr-mode", Static).render()) == "❄ FROZEN"
    with tui_env.get_session() as session:
        repo = AutonomyRepository(session)
        assert repo.portfolio_state(PORTFOLIO_ID).state == "frozen"
        # the live run in scope was rejected by the freeze (DB-level halt)
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "rejected"


@pytest.mark.asyncio
async def test_freezing_a_non_focused_portfolio_names_it_and_flags_the_header(
    tui_env,
) -> None:
    # The Decisions queue is cross-portfolio, so the gate's portfolio need not be the
    # header-focused one (here: fund-wide).
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_killswitch()
        await pilot.pause()
        await pilot.click("#scope-portfolio")
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()

        mode = app.query_one("#hdr-mode", Static)
        assert str(PORTFOLIO_ID) in app.last_status  # says which portfolio froze
        assert str(mode.render()) == "● ARMED · ❄ 1 frozen"

        app.focus_portfolio(PORTFOLIO_ID)  # the frozen one: no longer "other"
        await pilot.pause()
        assert str(mode.render()) == "❄ FROZEN"

        app.focus_portfolio(_PID_B)  # a third, armed portfolio
        await pilot.pause()
        assert str(mode.render()) == "● ARMED · ❄ 1 other frozen"


@pytest.mark.asyncio
async def test_killswitch_fund_scope_freezes(tui_env) -> None:
    from fund.audit.autonomy_repository import AutonomyRepository

    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_decide_killswitch()
        await pilot.pause()
        await pilot.click("#scope-fund")
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert str(app.query_one("#hdr-mode", Static).render()) == "❄ FROZEN"
    with tui_env.get_session() as session:
        assert AutonomyRepository(session).fund_state().state == "frozen"


@pytest.mark.asyncio
async def test_rearm_portfolio_restores_armed_without_rationale(tui_env) -> None:
    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    with tui_env.get_session() as session:
        AutonomyRepository(session).set_state(
            AutonomyScope.PORTFOLIO, PORTFOLIO_ID, AutonomyState.FROZEN, reason="test"
        )
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        assert str(app.query_one("#hdr-mode", Static).render()) == "❄ FROZEN"
        app.action_rearm()
        await pilot.pause()
        assert isinstance(app.screen, ScopePickScreen)
        await pilot.click("#scope-portfolio")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)  # portfolio re-arm: plain confirm
        await pilot.click("#confirm-yes")
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert str(app.query_one("#hdr-mode", Static).render()) == "● ARMED"
    with tui_env.get_session() as session:
        state = AutonomyRepository(session).portfolio_state(PORTFOLIO_ID).state
        assert state == "armed"


@pytest.mark.asyncio
async def test_header_glyphs_show_unknown_not_armed_idle_on_read_error(
    shell_env,
) -> None:
    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    with shell_env.get_session() as session:
        AutonomyRepository(session).set_state(
            AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="test"
        )
        _seed_run(session, _PID_A, seed=1, status="paused")
        session.commit()
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        mode = app.query_one("#hdr-mode", Static)
        assert str(mode.render()) == "❄ FROZEN"
        assert "awaiting" in app._runstate

        @contextmanager
        def _failing_session():
            raise ConnectionError("pool timeout")
            yield  # pragma: no cover

        healthy = app._session_factory
        app._session_factory = _failing_session
        app._refresh_runstate()
        await pilot.pause()
        # A blind poll must not read as a permissive "armed"/"idle" on a frozen fund.
        assert "ARMED" not in str(mode.render())
        assert "unknown" in str(mode.render()).lower()
        assert "idle" not in app._runstate
        assert "unknown" in app._runstate.lower()

        app._session_factory = healthy
        app._refresh_runstate()
        await pilot.pause()
        assert str(mode.render()) == "❄ FROZEN"  # self-heals on the next good read
        assert "awaiting" in app._runstate


@pytest.mark.asyncio
async def test_rearm_fund_requires_rationale(tui_env) -> None:
    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    with tui_env.get_session() as session:
        AutonomyRepository(session).set_state(
            AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="test"
        )
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)
        app.action_rearm()
        await pilot.pause()
        await pilot.click("#scope-fund")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RationaleConfirmScreen)
        confirm = screen.query_one("#rationale-yes", Button)
        assert confirm.disabled is True  # gated until a rationale is typed
        screen.query_one("#rationale-text", Input).value = "Risk reviewed; resuming."
        await pilot.pause()
        assert confirm.disabled is False  # enabled once non-empty
        screen._submit()
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
    with tui_env.get_session() as session:
        assert AutonomyRepository(session).fund_state().state == "armed"
        rows = HumanAuditLogRepository(session).list_recent(action="rearm")
        assert len(rows) == 1
        assert rows[0].rationale == "Risk reviewed; resuming."


@pytest.mark.asyncio
async def test_posture_strip_mirrors_fund_autonomy(tui_env) -> None:
    from fund.audit.autonomy_repository import (
        AutonomyRepository,
        AutonomyScope,
        AutonomyState,
    )

    with tui_env.get_session() as session:
        AutonomyRepository(session).set_state(
            AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="test"
        )
        session.commit()
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert "FROZEN" in str(view.query_one("#posture-strip", Static).render())


# --- Notification bus: poll-diff → persistent toasts + counters (T10) ---------


@pytest.mark.asyncio
async def test_notification_bus_toasts_new_gate_then_no_retoast(shell_env) -> None:
    from _fund_fakes import seed_paused_fund_run

    app = _make_shell(shell_env)  # empty fund: nothing awaiting yet
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app._refresh()  # a tick with no gates
        await pilot.pause()
        assert app._awaiting_count == 0
        before = len(app.last_attention)

        # Inject a new paused gate between two ticks (cross-portfolio pending_hitl).
        seed_paused_fund_run(
            shell_env.get_session,
            shell_env.persistence.saver,
            shell_env.persistence.store,
        )
        app._refresh()  # the new gate appears → persistent toast + counter bump
        await pilot.pause()
        assert app._awaiting_count == 1
        assert len(app.last_attention) == before + 1
        assert str(app.query_one("#hdr-attention", Static).render()) == "⏸1 awaiting"

        # An already-seen gate must NOT re-toast on the following tick.
        app._refresh()
        await pilot.pause()
        assert app._awaiting_count == 1
        assert len(app.last_attention) == before + 1  # unchanged — no re-toast


@pytest.mark.asyncio
async def test_attention_toast_outlives_textuals_default_timeout(shell_env) -> None:
    # ``notify(timeout=None)`` means "the default" (5s), not "never": a gate raised
    # while the adviser is away would vanish and, being in the seen-set, never re-toast.
    from _fund_fakes import seed_paused_fund_run

    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        seed_paused_fund_run(
            shell_env.get_session,
            shell_env.persistence.saver,
            shell_env.persistence.store,
        )
        app._refresh()
        await pilot.pause()

        toasts = list(app._notifications)
        assert len(toasts) == 1
        assert toasts[0].timeout > app.NOTIFICATION_TIMEOUT
        assert toasts[0].time_left > 3600  # effectively until the adviser dismisses it


@pytest.mark.asyncio
async def test_notification_bus_toasts_drift_breach_once(shell_env) -> None:
    # A portfolio whose L1 drift breaches its mandate threshold raises a toast. With
    # no positions, drift = Σ target weights = 1.0 ≥ the 0.1 mandate threshold.
    with shell_env.get_session() as session:
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=_PID_A,
            asof=date(2026, 1, 2),
            seed=1,
            universe=[],
            optimizer_config={},
        )
        repo.finalize_run(run.id, weights={"AAA": 1.0}, status="completed")
        MandateRepository(session).upsert(make_mandate(_PID_A))
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        assert any("drift breach" in message for message in app.last_attention)
        # The same breach does not re-toast on the next tick.
        seen = len(app.last_attention)
        app._refresh()
        await pilot.pause()
        assert len(app.last_attention) == seen


@pytest.mark.asyncio
async def test_notification_bus_preserves_state_on_read_error(
    shell_env, monkeypatch
) -> None:
    """A transient read error leaves the seen-set and counter intact (no re-toast)."""
    from _fund_fakes import seed_paused_fund_run

    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        seed_paused_fund_run(
            shell_env.get_session,
            shell_env.persistence.saver,
            shell_env.persistence.store,
        )
        app._poll_attention()  # the gate is seen → persistent toast + counter = 1
        await pilot.pause()
        assert app._awaiting_count == 1
        seen = len(app.last_attention)

        # A read error on the next tick must neither clear the seen-set nor zero the
        # counter, so the still-pending gate is not re-toasted when the read recovers.
        def _boom(*_a, **_k):
            raise RuntimeError("transient DB hiccup")

        monkeypatch.setattr(observe, "pending_hitl", _boom)
        app._poll_attention()
        await pilot.pause()
        assert app._awaiting_count == 1  # preserved, not flickered to zero
        assert len(app.last_attention) == seen  # no re-toast
        assert str(app.query_one("#hdr-attention", Static).render()) == "⏸1 awaiting"

        monkeypatch.undo()
        app._poll_attention()  # recovery: gate already in the seen-set → still no toast
        await pilot.pause()
        assert app._awaiting_count == 1
        assert len(app.last_attention) == seen


def _seed_breaching_portfolio(env, pid) -> None:
    # No positions, so drift = sum of target weights = 1.0, over the 0.1 threshold.
    with env.get_session() as session:
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=pid,
            asof=date(2026, 1, 2),
            seed=1,
            universe=[],
            optimizer_config={},
        )
        repo.finalize_run(run.id, weights={"AAA": 1.0}, status="completed")
        MandateRepository(session).upsert(make_mandate(pid))
        session.commit()


def _corrupt_mandate(env, pid) -> None:
    with env.get_session() as session:
        repo = MandateRepository(session)
        repo.upsert(make_mandate(pid))
        row = repo.get(pid)
        row.mandate = {**row.mandate, "capital": "not-a-number"}
        session.commit()


@pytest.mark.asyncio
async def test_unreadable_mandate_does_not_blind_the_gate_alerts(shell_env) -> None:
    from _fund_fakes import seed_paused_fund_run

    _corrupt_mandate(shell_env, _PID_B)
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        seed_paused_fund_run(
            shell_env.get_session,
            shell_env.persistence.saver,
            shell_env.persistence.store,
        )
        app._poll_attention()
        await pilot.pause()

        assert app._awaiting_count == 1
        assert any("gate awaiting" in message for message in app.last_attention)
        assert str(app.query_one("#hdr-attention", Static).render()) == "⏸1 awaiting"


@pytest.mark.asyncio
async def test_unreadable_mandate_does_not_blind_other_portfolios_breaches(
    shell_env,
) -> None:
    _seed_breaching_portfolio(shell_env, _PID_A)
    _corrupt_mandate(shell_env, _PID_B)
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app._poll_attention()
        await pilot.pause()

        assert [m for m in app.last_attention if "drift breach" in m] == [
            f"◐ drift breach · {str(_PID_A)[:8]}"
        ]


@pytest.mark.asyncio
async def test_gate_read_failure_does_not_blind_the_breach_alerts(
    shell_env, monkeypatch
) -> None:
    _seed_breaching_portfolio(shell_env, _PID_A)

    def _boom(*_a, **_k):
        raise RuntimeError("gate read failed")

    monkeypatch.setattr(observe, "pending_hitl", _boom)
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app._poll_attention()
        await pilot.pause()

        assert any("drift breach" in message for message in app.last_attention)


@pytest.mark.asyncio
async def test_attention_feed_flags_itself_degraded_until_every_read_succeeds(
    shell_env,
) -> None:
    _corrupt_mandate(shell_env, _PID_B)
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        feed = app.query_one("#hdr-feed", Static)
        app._poll_attention()
        await pilot.pause()
        assert feed.display
        assert "degraded" in str(feed.render())

        with shell_env.get_session() as session:
            MandateRepository(session).upsert(make_mandate(_PID_B))  # repaired
            session.commit()
        app._poll_attention()
        await pilot.pause()
        assert not feed.display


@pytest.mark.asyncio
async def test_unreadable_portfolio_keeps_its_alerted_breach_quiet(
    shell_env, monkeypatch
) -> None:
    _seed_breaching_portfolio(shell_env, _PID_A)
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app._poll_attention()
        await pilot.pause()
        alerted = len(app.last_attention)
        assert alerted == 1

        original = ShellApp._is_drift_breach

        def _flaky(self, session, portfolio_id):
            if portfolio_id == _PID_A:
                raise RuntimeError("drift read failed")
            return original(self, session, portfolio_id)

        monkeypatch.setattr(ShellApp, "_is_drift_breach", _flaky)
        app._poll_attention()  # A's state is unknown, not "cleared"
        monkeypatch.undo()
        app._poll_attention()  # recovery: the breach is still the one already alerted
        await pilot.pause()

        assert len(app.last_attention) == alerted


def _attention_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "fund.tui.app" and r.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
async def test_unreadable_portfolio_is_logged_once_per_failure_episode(
    shell_env, caplog
) -> None:
    # The header only says "feed degraded"; the log is how an operator learns which
    # portfolio is failing. A persistent failure must not write a line every tick.
    with shell_env.get_session() as session:
        MandateRepository(session).upsert(make_mandate(_PID_B))
        session.commit()
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        caplog.clear()

        _corrupt_mandate(shell_env, _PID_B)
        app._poll_attention()
        app._poll_attention()  # still failing: same episode, no second line
        await pilot.pause()

        records = _attention_warnings(caplog)
        assert len(records) == 1
        assert str(_PID_B) in records[0].getMessage()
        assert records[0].exc_info is not None  # the traceback says why it failed

        with shell_env.get_session() as session:
            MandateRepository(session).upsert(make_mandate(_PID_B))  # repaired
            session.commit()
        app._poll_attention()
        assert len(_attention_warnings(caplog)) == 1  # recovery is quiet

        _corrupt_mandate(shell_env, _PID_B)  # a recurrence is a new episode
        app._poll_attention()
        assert len(_attention_warnings(caplog)) == 2


@pytest.mark.asyncio
async def test_gate_and_roster_read_failures_are_logged(
    shell_env, monkeypatch, caplog
) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        caplog.clear()

        def _gate_boom(*_a, **_k):
            raise RuntimeError("gate read failed")

        def _roster_boom(*_a, **_k):
            raise RuntimeError("roster read failed")

        monkeypatch.setattr(observe, "pending_hitl", _gate_boom)
        monkeypatch.setattr(observe, "list_portfolios", _roster_boom)
        app._poll_attention()
        await pilot.pause()

        messages = [r.getMessage() for r in _attention_warnings(caplog)]
        assert len(messages) == 2
        assert any("gate" in m for m in messages)
        assert any("roster" in m for m in messages)


# --- Daily brief (Phase 07 T3) ----------------------------------------------


async def _open_brief(app, pilot) -> DailyBriefView:
    """Switch the shell to the Daily brief view and paint it."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-brief"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-brief", DailyBriefView)


def _full_brief(
    pid,
    *,
    asof: date = date(2026, 10, 6),
    prev_day: date | None = date(2026, 10, 5),
    next_day: date | None = None,
    run_id: uuid.UUID = _BRIEF_RUN_ID,
):
    """A fully-populated DailyBrief fixture — every section carries real data."""
    from fund.attribution import AttributionResult
    from fund.performance import PerformanceSummary

    perf = PerformanceSummary(
        portfolio_id=pid,
        returns={"1D": 0.0042, "1M": 0.012, "YTD": 0.05, "1Y": 0.09, "max": 0.12},
        excess={"1D": 0.0011, "1M": None, "YTD": None, "1Y": None, "max": None},
        tracking_error=0.01,
        max_drawdown=-0.03,
        volatility=0.1,
        sharpe=1.2,
        beta=0.95,
    )
    attr = AttributionResult(
        total_return=0.0042,
        contributions={
            "equity": 0.0031,
            "credit": 0.0005,
            "usd": -0.0002,
            "rates": -0.0008,
        },
        residual=0.0016,
        factor_names=("equity", "credit", "usd", "rates"),
        insufficient_history=False,
    )
    return observe.DailyBrief(
        portfolio_id=pid,
        asof=asof,
        prev_day=prev_day,
        next_day=next_day,
        performance=perf,
        attribution=attr,
        market_events=({"as_of": "2026-10-06", "narrative": "CPI < expected"},),
        economist_read="risk-on; your tech tilt benefited from the soft CPI",
        economist_read_date=date(2026, 10, 3),  # carried forward (!= asof)
        deliberation_run_id=run_id,
        fund_events=(
            {
                "as_of": "2026-10-03",
                "allocation": {"AAA": 1.0},
                "narrative": "rebalance ✓",
            },
        ),
    )


def _empty_brief(pid):
    """A fully-degraded DailyBrief fixture — every section is missing data."""
    from fund.attribution import AttributionResult

    attr = AttributionResult(
        total_return=0.0,
        contributions={},
        residual=0.0,
        factor_names=(),
        insufficient_history=True,
    )
    return observe.DailyBrief(
        portfolio_id=pid,
        asof=date(2026, 10, 6),
        prev_day=None,
        next_day=None,
        performance=None,
        attribution=attr,
        market_events=(),
        economist_read=None,
        economist_read_date=None,
        deliberation_run_id=None,
        fund_events=(),
    )


@pytest.mark.asyncio
async def test_daily_brief_registers_in_sidebar_and_opens(shell_env) -> None:
    """The sidebar lists Daily brief; selecting it switches to the brief pane."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert len(app.query(ListItem)) == len(_VIEW_KEYS)
        content = app.query_one(ContentSwitcher)
        content.current = "view-brief"
        app._refresh()
        await pilot.pause()
        view = app.query_one("#view-brief", DailyBriefView)
        assert view.view_title == "Daily brief"


@pytest.mark.asyncio
async def test_daily_brief_view_renders_five_sections(shell_env, monkeypatch) -> None:
    """Over an assembler fixture the brief paints all five sections."""
    brief = _full_brief(_PID_A)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: brief)

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_brief(app, pilot)
        text = view.detail_text
        assert view.brief is not None and view.brief.asof == date(2026, 10, 6)
        assert "TODAY" in text and "up" in text  # dot+label P&L direction
        assert "MACRO ATTRIBUTION" in text
        assert any(cell in text for cell in "▏▎▍▌▋▊▉█")  # a meter rendered
        assert "(-)" in text  # a negative contribution marked, not colour alone
        assert "WHAT HAPPENED" in text and "CPI < expected" in text
        assert "ECONOMIST READ" in text and "tech tilt" in text
        assert "(from 10-03)" in text  # carried-forward read is dated
        assert "FUND DID" in text and "rebalance" in text


@pytest.mark.asyncio
async def test_daily_brief_degrades_missing_sections(shell_env, monkeypatch) -> None:
    """Each missing section renders an explicit no-data line rather than crashing."""
    brief = _empty_brief(_PID_A)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: brief)

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_brief(app, pilot)
        assert view.detail_text.count("no data") >= 4


def _short_history_brief(pid):
    """A brief whose attribution is populated but flagged as a short-window OLS fit."""
    import dataclasses

    brief = _full_brief(pid)
    return dataclasses.replace(
        brief,
        attribution=dataclasses.replace(brief.attribution, insufficient_history=True),
    )


def test_daily_brief_flags_short_history_attribution_in_detail() -> None:
    """A populated but short-window fit carries a caveat, not a normal-looking read."""
    from fund.tui.views import _render_brief_detail

    flagged = _render_brief_detail(_short_history_brief(_PID_A))
    normal = _render_brief_detail(_full_brief(_PID_A))

    assert "short history" in flagged
    assert "short history" not in normal
    # The bars still render: the caveat qualifies the fit, it does not hide it.
    assert "equity" in flagged and "total" in flagged


def test_daily_brief_flags_short_history_attribution_in_markdown() -> None:
    """The exported markdown recap keeps the short-history caveat with the table."""
    from fund.tui.views import _render_brief_markdown

    flagged = _render_brief_markdown(_short_history_brief(_PID_A))
    normal = _render_brief_markdown(_full_brief(_PID_A))

    assert "short history" in flagged
    assert "short history" not in normal
    assert "| equity |" in flagged


def test_daily_brief_empty_attribution_shows_no_short_history_caveat() -> None:
    """A fully degraded estimate keeps its plain no-data line, with no extra caveat."""
    from fund.tui.views import _render_brief_detail

    assert "short history" not in _render_brief_detail(_empty_brief(_PID_A))


def _nav_grid(pid):
    """A three-day NAV grid keyed by the asof the assembler is asked for."""
    latest = _full_brief(
        pid, asof=date(2026, 10, 6), prev_day=date(2026, 10, 5), next_day=None
    )
    mid = _full_brief(
        pid,
        asof=date(2026, 10, 5),
        prev_day=date(2026, 10, 4),
        next_day=date(2026, 10, 6),
    )
    earliest = _full_brief(
        pid, asof=date(2026, 10, 4), prev_day=None, next_day=date(2026, 10, 5)
    )
    return {
        None: latest,
        date(2026, 10, 6): latest,
        date(2026, 10, 5): mid,
        date(2026, 10, 4): earliest,
    }


@pytest.mark.asyncio
async def test_daily_brief_nav_steps_the_grid(shell_env, monkeypatch) -> None:
    """←/→ step to the prev/next NAV-marked day; arrows no-op at the grid ends."""
    grid = _nav_grid(_PID_A)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, asof=None: grid[asof])

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_brief(app, pilot)
        assert view.brief is not None and view.brief.asof == date(2026, 10, 6)

        view.action_prev_day()
        assert view.asof == date(2026, 10, 5)
        assert view.brief.asof == date(2026, 10, 5)

        view.action_prev_day()
        assert view.asof == date(2026, 10, 4)

        rc = view.render_count
        view.action_prev_day()  # earliest mark → prev is None → no-op, no re-render
        assert view.asof == date(2026, 10, 4)
        assert view.render_count == rc

        view.action_next_day()
        assert view.asof == date(2026, 10, 5)


@pytest.mark.asyncio
async def test_daily_brief_arrow_keys_navigate_days(shell_env, monkeypatch) -> None:
    """The ←/→ view bindings are wired to the day-step actions when focused."""
    grid = _nav_grid(_PID_A)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, asof=None: grid[asof])

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_brief(app, pilot)
        view.focus()
        await pilot.pause()
        await pilot.press("left")
        await pilot.pause()
        assert view.asof == date(2026, 10, 5)


@pytest.mark.asyncio
async def test_daily_brief_posture_tile_opens_the_brief(shell_env, monkeypatch) -> None:
    """The Posture 'Daily brief' tile jumps to the brief, portfolio still focused."""
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)  # a run so the posture detail renders
        session.commit()
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: _full_brief(_PID_A))

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert "Daily brief" in view.detail_text  # the tile affordance is present

        app.action_open_brief()
        await pilot.pause()
        assert app.query_one(ContentSwitcher).current == "view-brief"
        assert app._portfolio_id == _PID_A  # same portfolio stays focused


@pytest.mark.asyncio
async def test_daily_brief_enter_opens_deliberation(shell_env, monkeypatch) -> None:
    """↵ on the brief opens the day's deliberation and preselects its run."""
    with shell_env.get_session() as session:
        run = _seed_run(session, _PID_A, seed=1)
        session.commit()
        run_id = run.id
    monkeypatch.setattr(
        observe, "daily_brief", lambda *a, **k: _full_brief(_PID_A, run_id=run_id)
    )

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_brief(app, pilot)
        app.action_open_deliberation()
        await pilot.pause()
        assert app.query_one(ContentSwitcher).current == "view-deliberation"
        delib = app.query_one("#view-deliberation", DeliberationView)
        assert delib.selected_run_id == run_id


@pytest.mark.asyncio
async def test_daily_brief_enter_noop_when_never_run(shell_env, monkeypatch) -> None:
    """↵ is a no-op (stays on the brief) when the day resolves to no run."""
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: _empty_brief(_PID_A))

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_brief(app, pilot)
        app.action_open_deliberation()
        await pilot.pause()
        assert app.query_one(ContentSwitcher).current == "view-brief"


@pytest.mark.asyncio
async def test_daily_brief_export_markdown(shell_env, monkeypatch, tmp_path) -> None:
    """e writes daily-brief-<pid>-<asof>.md to CWD and reports the path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: _full_brief(_PID_A))

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_brief(app, pilot)
        app.action_export_brief()
        await pilot.pause()
        expected = tmp_path / f"daily-brief-{_PID_A}-2026-10-06.md"
        assert expected.exists()
        assert "daily-brief" in app.last_status
        content = expected.read_text(encoding="utf-8")
        assert content.startswith("# Daily brief")
        assert "## Macro attribution" in content
        # Re-export overwrites — idempotent.
        app.action_export_brief()
        await pilot.pause()
        assert len(list(tmp_path.glob("daily-brief-*.md"))) == 1


@pytest.mark.asyncio
async def test_daily_brief_export_off_pane(shell_env, monkeypatch, tmp_path) -> None:
    """e off the brief pane writes nothing and sets no export status."""
    monkeypatch.chdir(tmp_path)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)  # the brief is not the active pane
        app.action_export_brief()
        await pilot.pause()
        assert not list(tmp_path.glob("daily-brief-*.md"))
        assert "exported brief" not in app.last_status


@pytest.mark.asyncio
async def test_daily_brief_export_degraded_markdown(
    shell_env, monkeypatch, tmp_path
) -> None:
    """A degraded day exports markdown with explicit no-data sections, no crash."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: _empty_brief(_PID_A))

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_brief(app, pilot)
        app.action_export_brief()
        await pilot.pause()
        content = (tmp_path / f"daily-brief-{_PID_A}-2026-10-06.md").read_text(
            encoding="utf-8"
        )
        assert content.startswith("# Daily brief")
        assert content.count("_no data_") >= 3  # attribution + events + read empty


@pytest.mark.asyncio
async def test_daily_brief_export_reports_oserror(
    shell_env, monkeypatch, tmp_path
) -> None:
    """A write failure surfaces on the status line, never a crash."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: _full_brief(_PID_A))

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_brief(app, pilot)

        def _boom(_path):
            raise OSError("disk full")

        monkeypatch.setattr(view, "export_markdown", _boom)
        app.action_export_brief()
        await pilot.pause()
        assert "export failed" in app.last_status
        assert not list(tmp_path.glob("daily-brief-*.md"))


def test_daily_brief_status_tiers_carry_labels() -> None:
    """Every P&L tier is (dot, label) with a word, so direction never rides on colour.

    A theme-independent contrast guarantee: the label disambiguates the two ``●`` tiers,
    so the brief reads the same on light and dark terminals.
    """
    from fund.tui.views import _BRIEF_DOWN, _BRIEF_FLAT, _BRIEF_UP

    for dot, label in (_BRIEF_UP, _BRIEF_FLAT, _BRIEF_DOWN):
        assert dot and label
    assert _BRIEF_UP[0] == _BRIEF_DOWN[0]  # both ●
    assert _BRIEF_UP[1] != _BRIEF_DOWN[1]  # told apart by the label


def test_bar_meter_negative_marker_is_ascii() -> None:
    """The negative marker is ASCII ``(-)`` — a sign glyph, not colour alone."""
    from fund.tui.readouts import attribution_bar

    assert attribution_bar(-0.5, 1.0, width=8).endswith(" (-)")
    assert attribution_bar(0.5, 1.0, width=8).endswith("█")  # positive: no marker


@pytest.mark.asyncio
async def test_daily_brief_opens_focused_so_keys_work(shell_env, monkeypatch) -> None:
    """The Posture tile jump focuses the brief, so ← navigates with no focus()."""
    grid = _nav_grid(_PID_A)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, asof=None: grid[asof])

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app.action_open_brief()  # the Posture 'd' jump — no manual focus() here
        await pilot.pause()
        view = app.query_one("#view-brief", DailyBriefView)
        assert app.focused is view  # the jump leaves the brief focused
        await pilot.press("left")  # so its bindings are reachable
        await pilot.pause()
        assert view.asof == date(2026, 10, 5)


@pytest.mark.asyncio
async def test_daily_brief_sidebar_pick_focuses(shell_env, monkeypatch) -> None:
    """Picking 'Daily brief' in the sidebar focuses the brief pane."""
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: _full_brief(_PID_A))

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        sidebar = app.query_one("#sidebar", ListView)
        sidebar.focus()
        sidebar.index = _VIEW_KEYS.index("brief")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        view = app.query_one("#view-brief", DailyBriefView)
        assert app.focused is view


# --- Performance view (Phase 08 T2/T3) --------------------------------------

_PERF_D1 = date(2026, 10, 2)
_PERF_D2 = date(2026, 10, 3)


def _seed_perf(session, pid, *, with_fx: bool = True) -> None:
    """Seed a two-day NAV series + a one-line USD paper book (EUR base) for ``pid``.

    ``with_fx=False`` omits the EUR→USD rates so marking the USD book raises
    ``FxCoverageError`` while the NAV-only summary still reads — the FX-gap case.
    """
    from portopt_db.models.market_data.yfinance_data import PriceHistory
    from portopt_db.models.orders.paper_order import PaperOrder
    from portopt_db.models.universe.universe import Exchange, Instrument
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    from fund.audit.nav_history_repository import NavHistoryRepository

    NavHistoryRepository(session).upsert_navs(
        [
            {
                "portfolio_id": pid,
                "date": _PERF_D1,
                "nav": Decimal("100000"),
                "benchmark": Decimal("200"),
            },
            {
                "portfolio_id": pid,
                "date": _PERF_D2,
                "nav": Decimal("101200"),
                "benchmark": Decimal("202"),
            },
        ]
    )
    ex = Exchange(name=f"EXP-{pid}")
    session.add(ex)
    session.flush()
    inst = Instrument(
        ticker="US1", short_name="US1", exchange=ex, yfinance_ticker="US1"
    )
    session.add(inst)
    session.flush()
    session.add_all(
        [
            PriceHistory(
                instrument_id=inst.id,
                date=_PERF_D1,
                close=Decimal("100"),
                price_unit="USD",
            ),
            PriceHistory(
                instrument_id=inst.id,
                date=_PERF_D2,
                close=Decimal("110"),
                price_unit="USD",
            ),
        ]
    )
    if with_fx:
        FxRatesRepository(session).upsert_rates(
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "date": _PERF_D1,
                    "rate": 0.90,
                },
                {
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "date": _PERF_D2,
                    "rate": 0.92,
                },
            ]
        )
    session.add(
        PaperOrder(
            portfolio_id=pid,
            asof=_PERF_D1,
            weights_hash="h1",
            weights={"US1": 1.0},
            fill_date=_PERF_D1,
            lines=[{"ticker": "US1", "notional": 9000.0, "fill_price": 100.0}],
            notional=9000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )


async def _open_performance(app, pilot) -> PerformanceView:
    """Switch the shell to the Performance view and paint its strip + holdings."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-performance"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-performance", PerformanceView)


@pytest.mark.asyncio
async def test_performance_registers_in_sidebar_and_opens(shell_env) -> None:
    """The sidebar lists Performance; selecting it switches to the performance pane."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert len(app.query(ListItem)) == len(_VIEW_KEYS)
        content = app.query_one(ContentSwitcher)
        content.current = "view-performance"
        app._refresh()
        await pilot.pause()
        view = app.query_one("#view-performance", PerformanceView)
        assert view.view_title == "Performance"


@pytest.mark.asyncio
async def test_performance_strip_and_holdings_render(shell_env) -> None:
    """A marked portfolio paints a real metrics strip + a per-holding table row."""
    with shell_env.get_session() as session:
        _seed_perf(session, _PID_A)
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_performance(app, pilot)
        assert view.summary is not None
        # nav 100000 -> 101200 is +1.20% over the window (not a fabricated figure).
        assert "+1.20%" in view.summary_text
        assert "RETURNS" in view.summary_text and "RISK" in view.summary_text
        # The one USD holding is marked into the holdings table.
        assert view.query_one("#perf-holdings", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_performance_holding_drilldown(shell_env) -> None:
    """Selecting a holdings row opens its drill-down detail."""
    with shell_env.get_session() as session:
        _seed_perf(session, _PID_A)
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_performance(app, pilot)
        table = view.query_one("#perf-holdings", DataTable)
        table.focus()
        table.move_cursor(row=0)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert view.selected_ticker == "US1"
        assert "US1" in view.detail_text
        assert "unrealized P&L" in view.detail_text


@pytest.mark.asyncio
async def test_performance_unmarked_degrades(shell_env) -> None:
    """A focused portfolio with no NAV/book shows — placeholders + no holdings rows."""
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)  # in the fund, but unmarked
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_performance(app, pilot)
        assert "—" in view.summary_text  # no fabricated return
        assert view.query_one("#perf-holdings", DataTable).row_count == 0


@pytest.mark.asyncio
async def test_performance_no_focus_shows_hint(shell_env) -> None:
    """Fund-wide (no focus) the Performance view invites a pick, never crashes."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_performance(app, pilot)
        assert "no portfolio focused" in str(
            view.query_one("#perf-returns", Static).render()
        )


@pytest.mark.asyncio
async def test_performance_fx_gap_keeps_returns_strip(shell_env) -> None:
    """A holdings FX gap blanks only the table, never the NAV-based returns strip."""
    with shell_env.get_session() as session:
        _seed_perf(session, _PID_A, with_fx=False)  # USD book, no EUR→USD rates
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_performance(app, pilot)
        # The FX-independent returns strip still reads the real +1.20% NAV move...
        assert view.summary is not None
        assert "+1.20%" in view.summary_text
        # ...while the holdings mark needs the missing FX rate, so it degrades to empty.
        assert view.query_one("#perf-holdings", DataTable).row_count == 0


# --- Posture NAV tile (Phase 08 T3) -----------------------------------------


@pytest.mark.asyncio
async def test_posture_detail_nav_tile_shows_real_nav(shell_env) -> None:
    """The Posture detail NAV tile shows a real period return, not the placeholder."""
    from fund.audit.nav_history_repository import NavHistoryRepository

    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)
        NavHistoryRepository(session).upsert_navs(
            [
                {
                    "portfolio_id": _PID_A,
                    "date": date(2026, 10, 2),
                    "nav": Decimal("100000"),
                    "benchmark": None,
                },
                {
                    "portfolio_id": _PID_A,
                    "date": date(2026, 10, 3),
                    "nav": Decimal("101200"),
                    "benchmark": None,
                },
            ]
        )
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert "NAV" in view.detail_text
        assert "+1.20%" in view.detail_text  # real max return (100000 -> 101200)


@pytest.mark.asyncio
async def test_posture_detail_nav_tile_degrades_when_unmarked(shell_env) -> None:
    """A run-but-unmarked portfolio shows a no-marks line, never a fabricated NAV."""
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_posture(app, pilot)
        assert "no marks yet" in view.detail_text
        assert "ACTIVE RUN" in view.detail_text  # the rest of the detail still renders


# --- Ask view (Phase 08 T5) -------------------------------------------------


class _CannedAskModel(BaseChatModel):
    """A network-free chat model returning one fixed answer, for the Ask query path."""

    answer: str = "grounded: the fund looks healthy"

    @property
    def _llm_type(self) -> str:
        return "canned-ask"

    def _generate(self, messages, stop=None, run_manager=None, **_):
        """Return the canned answer regardless of the prompt (no network)."""
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )


async def _open_ask(app, pilot) -> AskView:
    """Switch the shell to the Ask view and paint it."""
    await pilot.pause()
    app.query_one(ContentSwitcher).current = "view-ask"
    app._refresh()
    await pilot.pause()
    return app.query_one("#view-ask", AskView)


@pytest.mark.asyncio
async def test_ask_registers_in_sidebar_and_opens(shell_env) -> None:
    """The sidebar lists Ask; selecting it switches to the ask pane."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert len(app.query(ListItem)) == len(_VIEW_KEYS)
        content = app.query_one(ContentSwitcher)
        content.current = "view-ask"
        app._refresh()
        await pilot.pause()
        view = app.query_one("#view-ask", AskView)
        assert view.view_title == "Ask"


@pytest.mark.asyncio
async def test_ask_query_returns_grounded_answer(shell_env) -> None:
    """A query turn builds grounded context, invokes the lazy model, logs the answer."""
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)
        session.commit()

    app = _make_shell(
        shell_env, portfolio_id=_PID_A, model_factory=lambda: _CannedAskModel()
    )
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        inp = view.query_one("#ask-input", Input)
        inp.focus()
        inp.value = "how are we doing?"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert view.last_answer == "grounded: the fund looks healthy"
        assert "grounded" in str(view.query_one("#ask-log", Static).render())
        assert "how are we doing?" in str(view.query_one("#ask-log", Static).render())


class _GatedAskModel(BaseChatModel):
    """A chat model that can park inside ``invoke`` until released, then answer or fail.

    ``entered`` is set as soon as the call starts and ``finished`` once it is about to
    return, so a test can overlap two queries deterministically.
    """

    answer: str = ""
    fail: bool = False
    entered: threading.Event | None = None
    release: threading.Event | None = None
    finished: threading.Event | None = None

    @property
    def _llm_type(self) -> str:
        return "gated-ask"

    def _generate(self, messages, stop=None, run_manager=None, **_):
        """Wait for ``release`` (when gated), then answer or raise."""
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=10)
        try:
            if self.fail:
                raise RuntimeError("model blew up")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=self.answer))]
            )
        finally:
            if self.finished is not None:
                self.finished.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_fails", [False, True], ids=["answer", "error"])
async def test_ask_superseded_query_never_paints_its_stale_reply(
    shell_env, stale_fails
) -> None:
    """A newer question cancels the older one; its late reply must not reach the log.

    A cancelled thread worker keeps running, so without a guard its answer (or error
    note) lands after the newer question's answer and reads as the reply to it.
    """
    entered, release, finished = (threading.Event() for _ in range(3))
    models = iter(
        [
            _GatedAskModel(
                answer="ANSWER-1 (slow)",
                fail=stale_fails,
                entered=entered,
                release=release,
                finished=finished,
            ),
            _GatedAskModel(answer="ANSWER-2 (fast)"),
        ]
    )
    app = _make_shell(
        shell_env, portfolio_id=_PID_A, model_factory=lambda: next(models)
    )
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("how are we doing?")
        assert await asyncio.to_thread(entered.wait, 5)  # Q1 is parked in the model
        view.submit_turn("what about drift?")
        second = app.last_worker
        assert second is not None
        await second.wait()
        await pilot.pause()
        assert view.last_answer == "ANSWER-2 (fast)"
        status_after_answer = app.last_status

        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.sleep(0.2)  # let a stale paint through if the guard is missing
        await pilot.pause()

        log = _rendered(view.query_one("#ask-log", Static))
        assert "ANSWER-2 (fast)" in log
        assert "ANSWER-1" not in log
        assert "ask failed" not in log
        assert view.last_answer == "ANSWER-2 (fast)"
        assert app.last_status == status_after_answer


@pytest.mark.asyncio
async def test_ask_log_renders_bracketed_answer_verbatim(shell_env) -> None:
    """Bracketed tickers/values and a stray ``[/x]`` in an answer neither vanish nor
    crash the log; the screen shows exactly what the markdown export keeps."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        answer = "top holdings: [AAPL], [MSFT]; cash [5%]; stray [/INST] tag"
        view.set_answer(answer)
        await pilot.pause()
        shown = str(view.query_one("#ask-log", Static).render())
        assert shown == f"ask: {answer}"


@pytest.mark.asyncio
async def test_ask_query_without_model_degrades_to_status(shell_env) -> None:
    """No model configured → a status line + log note, never a crash, no answer."""
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)  # model_factory=lambda: None
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("how are we doing?")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert view.last_answer == ""  # nothing fabricated
        assert "model" in app.last_status.lower()


# --- Ask action router (Phase 08 T6) ----------------------------------------


@pytest.mark.asyncio
async def test_ask_action_unambiguous_confirms_then_approves(tui_env) -> None:
    """'approve the gate' with one gate pushes a confirm; yes runs apply_action."""
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)  # no mutation without a confirm
        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "approved" in app.last_status
    with tui_env.get_session() as session:
        run = AgentRunRepository(session).get_run(tui_env.run_id)
        assert run.status == "completed"
        rows = HumanAuditLogRepository(session).list_recent(action="approve")
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_ask_action_cancel_is_a_noop(tui_env) -> None:
    """Cancelling the confirm fires no worker and leaves the gate paused."""
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app.last_worker is None  # nothing dispatched
        assert "cancelled" in app.last_status
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "paused"


@pytest.mark.asyncio
async def test_ask_action_reject_confirms_then_rejects(tui_env) -> None:
    """'reject the gate' confirms, then finalises the run rejected via apply_action."""
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("reject the gate")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert "rejected" in app.last_status
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "rejected"


@pytest.mark.asyncio
async def test_ask_action_freeze_fund_confirms_then_freezes(shell_env) -> None:
    """'freeze the fund' confirms, then flips fund autonomy FROZEN via apply_action."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("freeze the fund")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
    with shell_env.get_session() as session:
        from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState

        assert AutonomyRepository(session).fund_posture() == AutonomyState.FROZEN.value


@pytest.mark.parametrize(
    ("turn", "expected"),
    [
        ("freeze it", "FREEZE autonomy (halts runs) portfolio aaaa0001?"),
        ("re-arm it", "Re-arm autonomy portfolio aaaa0001?"),
        ("run a rebalance", "Run a paper rebalance for portfolio aaaa0001?"),
    ],
)
@pytest.mark.asyncio
async def test_ask_confirm_names_the_focused_portfolio_not_a_gate(
    shell_env, turn, expected
) -> None:
    """A focused freeze / re-arm / run names the portfolio in its confirm; only a
    resume verb (approve / reject / killswitch) talks about a gate."""
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn(turn)
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        prompt = str(app.screen.query_one(".confirm-prompt", Static).render())
        assert prompt == expected
        assert "gate" not in prompt


def test_ask_confirm_prompt_keys_the_target_on_the_verb() -> None:
    from fund.tui.ask import ResolvedAction

    run_id = uuid.UUID("dddd0004-0000-4000-8000-000000000004")
    # A resume verb names its gate and the gate's portfolio; a run verb never names a
    # gate, even if run_id is populated.
    assert (
        tui_app._ask_confirm_prompt(ResolvedAction("approve", run_id, _PID_A, None))
        == "Approve and execute gate #dddd0004 (portfolio aaaa0001)?"
    )
    assert (
        tui_app._ask_confirm_prompt(ResolvedAction("run", _PID_A, _PID_A, None))
        == "Run a paper rebalance for portfolio aaaa0001?"
    )


@pytest.mark.asyncio
async def test_ask_killswitch_the_fund_confirms_then_freezes_the_fund(
    tui_env,
) -> None:
    """'killswitch the fund' with a gate paused is a fund freeze, not a run reject."""
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("killswitch the fund")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        prompt = str(app.screen.query_one(".confirm-prompt", Static).render())
        assert "whole fund" in prompt and "gate" not in prompt
        await pilot.click("#confirm-yes")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
    with tui_env.get_session() as session:
        from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState

        assert AutonomyRepository(session).fund_posture() == AutonomyState.FROZEN.value


@pytest.mark.asyncio
async def test_ask_gate_confirm_names_the_gates_portfolio(tui_env) -> None:
    """An approve confirm names the gate's portfolio, so a fund-wide 'approve' never
    offers orders of an unnamed portfolio."""
    app = _make_shell(tui_env)  # fund-wide, nothing focused
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        prompt = str(app.screen.query_one(".confirm-prompt", Static).render())
        assert f"gate #{str(tui_env.run_id)[:8]}" in prompt
        assert f"portfolio {str(PORTFOLIO_ID)[:8]}" in prompt
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_ask_portfolio_scope_confirm_names_the_portfolio_not_a_gate(
    shell_env,
) -> None:
    """A focused 'freeze it' confirm spells out the portfolio — there is no gate."""
    with shell_env.get_session() as session:
        _seed_run(session, _PID_A, seed=1)
        session.commit()

    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("freeze it")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        prompt = str(app.screen.query_one(".confirm-prompt", Static).render())
        assert "portfolio aaaa0001" in prompt
        assert "gate" not in prompt


# --- Ask disambiguation (Phase 08 T7) ---------------------------------------


def _seed_fake_gate(env, pid, *, thread_id, seed) -> uuid.UUID:
    """Seed a second pending-HITL gate (paused run + live interrupt); return its id."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.types import Interrupt

    with env.get_session() as session:
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=pid,
            asof=date(2026, 1, 3),
            seed=seed,
            universe=[],
            optimizer_config={},
            status="paused",
        )
        repo.set_thread_id(run.id, thread_id)
        session.commit()
        run_id = run.id
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["messages"] = [HumanMessage(content="go")]
    checkpoint["channel_versions"]["messages"] = 1
    saved = env.saver.put(
        cfg, checkpoint, {"source": "loop", "step": 1}, {"messages": 1}
    )
    env.saver.put_writes(
        saved,
        [
            (
                "__interrupt__",
                Interrupt(value={"action_requests": [{"name": "place_orders"}]}),
            )
        ],
        "task-2",
    )
    return run_id


@pytest.mark.asyncio
async def test_ask_action_scope_ambiguous_asks_then_resolves(shell_env) -> None:
    """'freeze it' with no focus holds a pending action; 'the fund' resolves it."""
    app = _make_shell(shell_env)  # fund-wide, nothing focused
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("freeze it")
        await pilot.pause()
        assert view.pending_action is not None  # carried across turns
        assert app.last_worker is None  # did NOT execute
        assert not isinstance(app.screen, ConfirmScreen)  # no confirm yet

        view.submit_turn("the fund")
        await pilot.pause()
        assert view.pending_action is None  # cleared on resolution
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await pilot.pause()
        await app.last_worker.wait()
    with shell_env.get_session() as session:
        from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState

        assert AutonomyRepository(session).fund_posture() == AutonomyState.FROZEN.value


@pytest.mark.asyncio
async def test_ask_action_unrelated_turn_clears_pending(shell_env) -> None:
    """An unrelated next turn clears the pending action and is processed fresh."""
    app = _make_shell(shell_env)  # model_factory=lambda: None
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("freeze it")
        await pilot.pause()
        assert view.pending_action is not None

        view.submit_turn("how are we doing?")  # a query, unrelated to the freeze
        await pilot.pause()
        assert view.pending_action is None  # cleared, never executed
        assert not isinstance(app.screen, ConfirmScreen)
        worker = app.last_worker  # routed fresh → a query worker (model is None)
        if worker is not None:
            await worker.wait()
        await pilot.pause()
        assert "model" in app.last_status.lower()


@pytest.mark.asyncio
async def test_ask_action_two_gates_disambiguate_to_named(tui_env) -> None:
    """Two pending gates ask which; naming one reaches the confirm, then approves."""
    _seed_fake_gate(tui_env, PORTFOLIO_ID, thread_id="thread-gate2", seed=2)
    short = str(tui_env.run_id).split("-", 1)[0]

    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert view.pending_action is not None  # ambiguous: two gates
        assert app.last_worker is None
        assert not isinstance(app.screen, ConfirmScreen)

        view.submit_turn(f"#{short}")  # name the real gate
        await pilot.pause()
        assert view.pending_action is None
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert "approved" in app.last_status
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "completed"


@pytest.mark.asyncio
async def test_ask_pending_gate_answered_with_new_verb_runs_that_verb(tui_env) -> None:
    """'approve' asks which gate; answering 'reject #id' must reject it, not approve."""
    _seed_fake_gate(tui_env, PORTFOLIO_ID, thread_id="thread-gate2", seed=2)
    short = str(tui_env.run_id).split("-", 1)[0]

    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert view.pending_action is not None  # ambiguous: two gates

        view.submit_turn(f"reject #{short}")
        await pilot.pause()
        assert view.pending_action is None
        assert isinstance(app.screen, ConfirmScreen)
        assert "approve" not in app.screen._prompt.lower()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        await app.last_worker.wait()
        await pilot.pause()
        assert "rejected" in app.last_status
    with tui_env.get_session() as session:
        assert AgentRunRepository(session).get_run(tui_env.run_id).status == "rejected"
        assert HumanAuditLogRepository(session).list_recent(action="approve") == []


def _fail_gate_read(*_args, **_kwargs):
    raise RuntimeError("transient db error")


@pytest.mark.asyncio
async def test_ask_action_survives_a_gate_read_error(tui_env, monkeypatch) -> None:
    """A read failure while resolving an action turn reports on the shell, never exits.

    The resolve runs on the event loop inside the ``Input.Submitted`` handler, so an
    escaped exception would tear the whole cockpit down mid-decision.
    """
    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        monkeypatch.setattr(observe, "pending_hitl", _fail_gate_read)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert app.is_running
        assert "could not read gates" in app.last_status
        assert "could not read gates" in _rendered(view.query_one("#ask-log", Static))
        assert not isinstance(app.screen, ConfirmScreen)
        assert app.last_worker is None  # nothing dispatched on an unread gate list


@pytest.mark.asyncio
async def test_ask_pending_follow_up_survives_a_gate_read_error(
    tui_env, monkeypatch
) -> None:
    """A read failure resolving a disambiguation follow-up reports, never exits."""
    _seed_fake_gate(tui_env, PORTFOLIO_ID, thread_id="thread-gate2", seed=2)
    short = str(tui_env.run_id).split("-", 1)[0]

    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert view.pending_action is not None  # ambiguous: two gates

        monkeypatch.setattr(observe, "pending_hitl", _fail_gate_read)
        view.submit_turn(f"approve #{short}")
        await pilot.pause()
        assert app.is_running
        assert view.pending_action is None  # a pending action stays single-shot
        assert "could not read gates" in _rendered(view.query_one("#ask-log", Static))
        assert not isinstance(app.screen, ConfirmScreen)


# --- Agent-originated text is data, never Textual markup ---------------------

# Bracketed key=value runs and a stray closing tag all raise ``MarkupError`` when a
# ``Static`` parses them as markup — realistic in finance prose from a model.
_BRACKETED = "drift [a=1, b=2] vs [ECB=hawkish, BoE=dovish] and a stray [/]"


def _rendered(widget) -> str:
    return str(widget.render())


@pytest.mark.asyncio
async def test_ask_answer_with_bracket_sequences_renders_literally(shell_env) -> None:
    """A model answer holding ``[k=v, k=v]`` reaches the log verbatim, not as markup."""
    app = _make_shell(
        shell_env,
        portfolio_id=_PID_A,
        model_factory=lambda: _CannedAskModel(answer=_BRACKETED),
    )
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("how are we doing?")
        await pilot.pause()
        worker = app.last_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()
        assert view.last_answer == _BRACKETED
        assert _BRACKETED in _rendered(view.query_one("#ask-log", Static))
        # A later repaint re-renders every logged line, so it must stay healthy too.
        view.append_log("you", "and now?")
        assert _BRACKETED in _rendered(view.query_one("#ask-log", Static))


@pytest.mark.asyncio
async def test_deliberation_live_detail_renders_bracketed_text_literally(
    shell_env,
) -> None:
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        rid = uuid.uuid4()
        view.start_live(rid, _PID_A)
        view.ingest_event(
            StreamEvent(
                kind="step",
                run_id=rid,
                index=0,
                agent="economist",
                step="report",
                status=StepStatus.LIVE,
                text=_BRACKETED,
            )
        )
        await pilot.pause()
        assert _BRACKETED in _rendered(view.query_one("#delib-detail", Static))


@pytest.mark.asyncio
async def test_deliberation_replay_detail_renders_bracketed_text_literally(
    shell_env,
) -> None:
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_deliberation(app, pilot)
        view._entries = [
            SimpleNamespace(
                agent="economist",
                step="report",
                source="message",
                index=0,
                text=_BRACKETED,
                payload=None,
            )
        ]
        view.selected_step = 0
        view._paint_detail()
        await pilot.pause()
        assert _BRACKETED in _rendered(view.query_one("#delib-detail", Static))


@pytest.mark.asyncio
async def test_decisions_gate_renders_bracketed_text_literally(tui_env) -> None:
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view._paint_gate(f"place_orders — {_BRACKETED}")
        await pilot.pause()
        assert _BRACKETED in _rendered(view.query_one("#decisions-gate", Static))


@pytest.mark.asyncio
async def test_audit_detail_renders_bracketed_text_literally(shell_env) -> None:
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_audit(app, pilot)
        view._entries = [
            SimpleNamespace(
                kind="agent",
                actor="allocator",
                action="optimize_portfolio",
                portfolio_id=None,
                run_id=None,
                target=None,
                prev_value=None,
                new_value=_BRACKETED,
                source=None,
                idempotency_key=None,
                rationale=None,
                override_flag=False,
                suitability_flag=False,
            )
        ]
        view.selected_index = 0
        view._paint_detail()
        await pilot.pause()
        assert _BRACKETED in _rendered(view.query_one("#audit-detail", Static))


@pytest.mark.asyncio
async def test_status_bar_renders_bracketed_error_text_literally(shell_env) -> None:
    """Exception text (pydantic's ``[type=..., input_value=...]``) is not markup."""
    app = _make_shell(shell_env)
    message = "invalid mandate: [type=greater_than, input_value=-1, input_type=int]"
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        app._set_status(message)
        await pilot.pause()
        assert message in _rendered(app.query_one("#status-bar", Static))


@pytest.mark.asyncio
async def test_weights_editor_error_renders_typed_brackets_literally(tui_env) -> None:
    """A typo echoed back inline (``[/]``) must not be parsed as a closing tag."""
    app = _make_shell(tui_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_decisions(app, pilot)
        view.select_run(tui_env.run_id)
        await pilot.pause()
        app.action_edit_decision()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, OrderEditScreen)
        screen.query_one("#edit-weights", Input).value = "[/]"
        screen._submit()
        await pilot.pause()
        assert isinstance(app.screen, OrderEditScreen)
        assert "bad entry '[/]'" in _rendered(screen.query_one("#edit-error", Static))


@pytest.mark.asyncio
async def test_questionnaire_validation_error_renders_literally(shell_env) -> None:
    """Pydantic's ``[type=..., input_value=...]`` suffix must not reach the markup
    parser: an out-of-range answer keeps the form open instead of killing the app."""
    from _profiler_fakes import make_answers

    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        screen = QuestionnaireScreen(ask_capital=True)
        app.push_screen(screen)
        await pilot.pause()
        _fill_questionnaire(screen, make_answers(), capital="100000")
        screen.query_one("#q-max-loss", Input).value = "150"
        screen.query_one("#q-submit", Button).press()
        await pilot.pause()
        assert app.is_running
        assert isinstance(app.screen, QuestionnaireScreen)
        error = _rendered(screen.query_one("#q-error", Static))
        assert error.startswith("invalid:")
        assert "input_value=150" in error


@pytest.mark.asyncio
async def test_daily_brief_renders_bracketed_economist_read_literally(
    shell_env, monkeypatch
) -> None:
    import dataclasses

    brief = dataclasses.replace(_full_brief(_PID_A), economist_read=_BRACKETED)
    monkeypatch.setattr(observe, "daily_brief", lambda *a, **k: brief)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_brief(app, pilot)
        assert _BRACKETED in _rendered(view.query_one("#brief-detail", Static))


@pytest.mark.asyncio
async def test_ask_pending_follow_up_with_another_verb_runs_that_verb(tui_env) -> None:
    """After 'approve' asks which gate, 'reject #id' is confirmed as a reject."""
    _seed_fake_gate(tui_env, PORTFOLIO_ID, thread_id="thread-gate2", seed=2)
    short = str(tui_env.run_id).split("-", 1)[0]

    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert view.pending_action is not None

        view.submit_turn(f"reject #{short}")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        prompt = str(app.screen.query_one(".confirm-prompt", Static).render())
        assert "Reject" in prompt and "Approve" not in prompt
        await pilot.press("escape")
        await pilot.pause()
        assert app.last_worker is None  # nothing was dispatched


@pytest.mark.asyncio
async def test_ask_pending_negated_follow_up_never_confirms_the_excluded_gate(
    tui_env,
) -> None:
    """'approve the other one, not #id' after 'approve which?' asks again; it must not
    confirm an approve of the gate the adviser just excluded."""
    _seed_fake_gate(tui_env, PORTFOLIO_ID, thread_id="thread-gate2", seed=2)
    short = str(tui_env.run_id).split("-", 1)[0]

    app = _make_shell(tui_env, portfolio_id=PORTFOLIO_ID)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("approve the gate")
        await pilot.pause()
        assert view.pending_action is not None

        view.submit_turn(f"approve the other one, not #{short}")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmScreen)
        assert view.pending_action is not None  # asked which gate, outright
        assert app.last_worker is None  # nothing was dispatched


@pytest.mark.asyncio
async def test_ask_pending_scope_question_about_the_fund_is_not_a_freeze(
    shell_env,
) -> None:
    """A question mentioning 'fund' after 'freeze what?' never reaches a confirm."""
    app = _make_shell(shell_env)  # model_factory=lambda: None
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.submit_turn("freeze it")
        await pilot.pause()
        assert view.pending_action is not None

        view.submit_turn("how is the fund doing?")
        await pilot.pause()
        assert view.pending_action is None
        assert not isinstance(app.screen, ConfirmScreen)
        worker = app.last_worker  # routed fresh as a query (model is None)
        if worker is not None:
            await worker.wait()


# --- Ask markdown export (Phase 08 T8) --------------------------------------


@pytest.mark.asyncio
async def test_ask_export_markdown(shell_env, monkeypatch, tmp_path) -> None:
    """e on the focused Ask view writes the conversation to ask-<pf>-<ts>.md."""
    monkeypatch.chdir(tmp_path)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.append_log("you", "how are we doing?")
        view.append_log("ask", "all good")
        view.focus()  # the view (not the input) holds the view-scoped e binding
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        files = list(tmp_path.glob("ask-*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "how are we doing?" in content and "all good" in content
        assert "exported ask" in app.last_status


@pytest.mark.asyncio
async def test_ask_export_off_pane_is_noop(shell_env, monkeypatch, tmp_path) -> None:
    """e off the Ask pane writes no ask file and sets no ask-export status."""
    monkeypatch.chdir(tmp_path)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        await _open_posture(app, pilot)  # Ask is not the active pane
        app.action_export_ask()
        await pilot.pause()
        assert not list(tmp_path.glob("ask-*.md"))
        assert "exported ask" not in app.last_status


@pytest.mark.asyncio
async def test_ask_export_reports_oserror(shell_env, monkeypatch, tmp_path) -> None:
    """A write failure surfaces on the status line, never crashing the App."""
    monkeypatch.chdir(tmp_path)
    app = _make_shell(shell_env, portfolio_id=_PID_A)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.append_log("you", "hi")

        def _boom(_path):
            raise OSError("disk full")

        monkeypatch.setattr(view, "export_markdown", _boom)
        app.action_export_ask()
        await pilot.pause()
        assert "export failed" in app.last_status


# --- Help overlay (Phase 08 T9) ---------------------------------------------


@pytest.mark.asyncio
async def test_help_overlay_opens_on_question_and_f1_and_closes(shell_env) -> None:
    """? and F1 both open the help overlay (listing the keymap); escape closes it."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        body = str(app.screen.query_one("#help-body", Static).render())
        assert "approve" in body and "Ask" in body  # lists the keymap
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)

        await pilot.press("f1")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


@pytest.mark.asyncio
async def test_help_f1_opens_even_with_ask_input_focused(shell_env) -> None:
    """F1 opens help even while the Ask input holds focus (a function key)."""
    app = _make_shell(shell_env)
    async with app.run_test(size=_SIZE) as pilot:
        view = await _open_ask(app, pilot)
        view.query_one("#ask-input", Input).focus()
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


# --- theming + contrast (Phase 08 T10) --------------------------------------


def _wcag_contrast(fg, bg) -> float:
    """WCAG 2.x contrast ratio between two Textual Colors (1.0 = identical)."""

    def _relative_luminance(color) -> float:
        def _channel(value: int) -> float:
            srgb = value / 255.0
            return srgb / 12.92 if srgb <= 0.03928 else ((srgb + 0.055) / 1.055) ** 2.4

        red, green, blue = color.rgb
        return (
            0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)
        )

    hi, lo = sorted((_relative_luminance(fg), _relative_luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_theme_status_tokens_are_dot_plus_label_never_colour_alone():
    """Every status token carries a glyph AND a word, so meaning never rides on colour.

    Covers the health roster, the daily-brief P&L, and the Performance view tokens —
    the ● / ◐ tiers that share a dot are disambiguated by their label.
    """
    from fund.tui.views import (
        _BRIEF_DOWN,
        _BRIEF_FLAT,
        _BRIEF_UP,
        _HEALTH_BREACH,
        _HEALTH_OK,
        _HEALTH_WARN,
        _PERF_DOWN,
        _PERF_FLAT,
        _PERF_UP,
    )

    tokens = [
        _HEALTH_OK,
        _HEALTH_WARN,
        _HEALTH_BREACH,
        _BRIEF_UP,
        _BRIEF_FLAT,
        _BRIEF_DOWN,
        _PERF_UP,
        _PERF_FLAT,
        _PERF_DOWN,
    ]
    for dot, label in tokens:
        assert dot in {"●", "◐"}  # a status glyph
        assert label and label.isalpha()  # a word carries the meaning, not colour


def test_contrast_destructive_colour_legible_on_light_and_dark():
    """The destructive/breach ($error) signal meets the WCAG UI floor on both themes.

    warning/success/accent sit lower on the light theme by design — meaning is carried
    by the dot + label — but the safety-critical destructive signal must stay legible by
    colour too, so it holds the 3:1 UI-contrast floor on both the light and dark themes.
    """
    from textual.app import App
    from textual.color import Color

    app = App()
    for theme_name in ("textual-dark", "textual-light"):
        palette = app.get_theme(theme_name).to_color_system().generate()
        surface = Color.parse(palette["surface"])
        error = Color.parse(palette["error"])
        assert _wcag_contrast(error, surface) >= 3.0, theme_name
