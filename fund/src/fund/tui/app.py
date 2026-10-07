"""``fund-tui`` — the five-panel Textual observer over ``fund.observe``.

A read-mostly "Bloomberg" cockpit for one portfolio: a transcript, the current-vs-
target state, the HITL approval queue, the run history, and the recent market/
portfolio events. All five panels are thin over the model-free read model
(``fund.observe``); the App owns every side effect:

* **Polling, not pushing.** ``FundTUI._refresh`` runs on a ``set_interval`` timer
  (``poll_interval``, ~2s), opening a *short read-only* session, extracting plain
  dataclasses, and rolling back — the UI never holds a transaction or an ORM row
  across ticks.
* **Resume off the event loop.** Approve/Reject dispatch ``_resume_worker``, a
  ``@work(thread=True)`` worker: the synchronous ``resume_run`` (which dispatches to
  the profiler or rebalance resumer by gate and drives ``agent.invoke``) runs on a
  background thread so the event loop never blocks, then marshals its UI updates back
  with ``call_from_thread``.
* **Lazy model, light import.** ``import fund.tui.app`` pulls only Textual + the
  agent-stack-free read model — no ``deepagents`` / ``langgraph`` / model, so it needs
  no ``OLLAMA_API_KEY`` and no ``DATABASE_URL``. The chat model is built lazily via the
  injected ``model_factory`` only when the adviser approves/rejects; the persistence
  handles (checkpointer + store, needing ``DATABASE_URL``) are built in ``main``.

Every collaborator is injected (``session_factory`` / ``persistence`` /
``model_factory``), so ``App.run_test`` can drive the whole cockpit headlessly
over SQLite + ``MemorySaver`` + a scripted model with zero network.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.events import Resize
from textual.widgets import (
    ContentSwitcher,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from fund import observe
from fund.config import FundConfig, settings
from fund.tui.views import VIEW_SPECS, ShellView, build_views
from fund.tui.widgets import (
    HistoryPanel,
    HitlQueuePanel,
    PortfolioStatePanel,
    RecentEventsPanel,
    TranscriptPanel,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType
    from textual.widgets import DataTable
    from textual.worker import Worker

_PAST = {"approve": "approved", "reject": "rejected"}


class FundTUI(App):
    """The five-panel observer App for a single portfolio."""

    CSS = """
    Screen { layout: grid; grid-size: 2 3; grid-gutter: 1; }
    .panel { border: round $accent; padding: 0 1; height: 1fr; }
    .panel-title { text-style: bold; color: $accent; }
    #transcript-log { height: 1fr; }
    #gate-detail { height: auto; color: $warning; }
    #hitl-buttons { height: auto; align-horizontal: left; }
    Button { margin: 0 1; }
    #status-bar { dock: bottom; height: 1; background: $panel; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("a", "approve", "Approve"),
        ("r", "reject", "Reject"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        portfolio_id: uuid.UUID,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]],
        persistence: Any,
        model_factory: Callable[[], Any],
        config: FundConfig = settings,
        poll_interval: float = 2.0,
    ) -> None:
        """Initialise the cockpit for a single portfolio.

        All collaborators are injected so the App can be driven headlessly in
        tests over SQLite + MemorySaver without any network dependencies.

        Args:
            portfolio_id: Portfolio to observe and act on.
            session_factory: Callable returning a context-manager-managed
                SQLAlchemy Session used for read ticks and resume commits.
            persistence: Langgraph persistence bundle exposing ``.saver``
                (checkpointer) and ``.store`` for transcript and interrupt
                reads.
            model_factory: Called lazily on the first approve/reject to build
                the chat model; failures surface as a status-bar message rather
                than crashing the App.
            config: Application settings; defaults to the process-level
                singleton.
            poll_interval: Refresh timer cadence in seconds.
        """
        super().__init__()
        self._portfolio_id = portfolio_id
        self._session_factory = session_factory
        self._persistence = persistence
        self._model_factory = model_factory
        self._config = config
        self._poll_interval = poll_interval
        self._selected_run_id: uuid.UUID | None = None
        self.last_worker: Worker[None] | None = None
        # Mirror of the last status line (the Static's content is not publicly
        # readable); lets headless tests assert what the cockpit reported.
        self.last_status: str = ""

    # --- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield TranscriptPanel(classes="panel", id="transcript-panel")
        yield PortfolioStatePanel(classes="panel", id="state-panel")
        yield HitlQueuePanel(classes="panel", id="hitl-panel")
        yield HistoryPanel(classes="panel", id="history-panel")
        yield RecentEventsPanel(classes="panel", id="events-panel")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        # Populate once the child panels have mounted their columns, then poll.
        self.call_after_refresh(self._refresh)
        self.set_interval(self._poll_interval, self._refresh)

    # --- polling read loop --------------------------------------------------

    def _refresh(self) -> None:
        """One poll tick: read the model-free state and repaint every panel.

        Opens a short read-only session, extracts plain dataclasses (safe to hold
        past the transaction), rolls back, and paints. A read error is surfaced on
        the status bar and swallowed so the timer keeps the cockpit alive.
        """
        try:
            with self._session_factory() as session:
                runs = observe.list_portfolio_runs(session, self._portfolio_id)
                queue = observe.pending_hitl(
                    session, self._persistence.saver, self._portfolio_id
                )
                state = observe.portfolio_state(session, self._portfolio_id)
                events = observe.recent_events(session, self._portfolio_id)
                transcript: list[observe.TranscriptEntry] = []
                interrupt: dict[str, Any] | None = None
                if self._selected_run_id is not None:
                    # Lazy repo import (fund.audit bootstraps langgraph): the ORM row
                    # carries thread_id + decisions the read model needs.
                    from fund.audit.repository import AgentRunRepository

                    run_row = AgentRunRepository(session).get_run(self._selected_run_id)
                    if run_row is not None:
                        transcript = observe.load_run_transcript(
                            session, self._persistence.saver, run_row
                        )
                        interrupt = observe.interrupt_for(
                            self._persistence.saver, run_row
                        )
                session.rollback()
        except Exception as exc:  # keep the poll loop alive on any read error
            self._set_status(f"refresh error: {exc}")
            return

        self.query_one(HistoryPanel).show_runs(runs)
        hitl = self.query_one(HitlQueuePanel)
        hitl.show_runs(queue)
        hitl.show_gate(interrupt)
        self.query_one(PortfolioStatePanel).show(state)
        self.query_one(TranscriptPanel).show(transcript)
        self.query_one(RecentEventsPanel).show(events)

    # --- selection ----------------------------------------------------------

    def select_run(self, run_id: uuid.UUID) -> None:
        """Focus a run: its transcript, state, and (if paused) gate now render."""
        self._selected_run_id = run_id
        self._refresh()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A queue/history row selection drives panels 1-2 (and the gate)."""
        key = event.row_key.value
        if key is not None:
            self.select_run(uuid.UUID(key))

    # --- approve / reject (worker thread) -----------------------------------

    def on_button_pressed(self, event: Any) -> None:
        if event.button.id == "approve-btn":
            self._resume("approve")
        elif event.button.id == "reject-btn":
            self._resume("reject")

    def action_approve(self) -> None:
        self._resume("approve")

    def action_reject(self) -> None:
        self._resume("reject")

    def _resume(self, decision: str) -> None:
        """Dispatch a rebuild-to-resume for the selected run on a worker thread."""
        if self._selected_run_id is None:
            self._set_status("select a paused run first")
            return
        self.last_worker = self._resume_worker(self._selected_run_id, decision)

    @work(thread=True, exclusive=True, group="resume")
    def _resume_worker(self, run_id: uuid.UUID, decision: str) -> None:
        """Rebuild the paused run's agent and resume its gate — off the event loop.

        The model is built lazily here (a missing ``OLLAMA_API_KEY`` surfaces as a
        clear status line, not a crash). All UI touches marshal back to the main
        thread via ``call_from_thread``.
        """
        from fund.agents.graph import resume_run

        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"model unavailable: {exc}")
            return
        try:
            with self._session_factory() as session:
                resume_run(
                    run_id,
                    decision,
                    session=session,
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    model=model,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"resume failed: {exc}")
            return
        self.call_from_thread(self._after_resume, run_id, decision)

    def _after_resume(self, run_id: uuid.UUID, decision: str) -> None:
        """On the main thread once the worker committed: report + repaint."""
        self._set_status(f"run {run_id} {_PAST[decision]}")
        self._refresh()

    # --- helpers ------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self.last_status = message
        self.query_one("#status-bar", Static).update(message)


_NARROW_WIDTH = 80  # below this terminal width the sidebar collapses to a rail


class ShellApp(App):
    """Mission-control shell: a sidebar of six views over one ContentSwitcher.

    The multi-view frame the redesign lands on — a left nav rail (Posture ·
    Deliberation · Decisions · Mandate · Audit · Ask) switching a single content
    pane, under a status header (focused portfolio · mode · run state) and a key
    hint bar. It lands **fund-wide** (``portfolio_id=None``) and can focus one
    portfolio; Phase-0 views are placeholders (:class:`~fund.tui.views.ShellView`),
    filled by later phases.

    Shares ``FundTUI``'s discipline: the same injected ``__init__`` seam (so
    ``run_test`` drives it headlessly over SQLite + ``MemorySaver`` with zero
    network), a synchronous per-view ``_refresh``, and an agent-stack-light import
    (Textual + the model-free read model only — no ``deepagents`` / ``langgraph``).
    """

    CSS = """
    .shell-header { height: 1; background: $panel; color: $text; }
    .shell-header Static { width: auto; padding: 0 2; }
    .shell-body { height: 1fr; }
    .sidebar { width: 22; border-right: solid $accent; }
    .sidebar.narrow { width: 5; }
    .content { width: 1fr; padding: 0 1; }
    .view-title { text-style: bold; color: $accent; }
    .status-bar { dock: bottom; height: 1; background: $panel; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        portfolio_id: uuid.UUID | None = None,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]],
        persistence: Any,
        model_factory: Callable[[], Any],
        config: FundConfig = settings,
        poll_interval: float = 2.0,
    ) -> None:
        """Initialise the shell, optionally focused on one portfolio.

        Args:
            portfolio_id: Portfolio to focus, or ``None`` for the fund-wide
                landing (the launcher opens the shell before a pick is made).
            session_factory: Callable returning a context-manager-managed
                SQLAlchemy Session for read ticks and the portfolio picker.
            persistence: Langgraph persistence bundle exposing ``.saver`` and
                ``.store`` (used by the data-bearing views in later phases).
            model_factory: Called lazily when the adviser acts; unused by the
                Phase-0 placeholder views.
            config: Application settings; defaults to the process singleton.
            poll_interval: Refresh timer cadence in seconds.
        """
        super().__init__()
        self._portfolio_id = portfolio_id
        self._session_factory = session_factory
        self._persistence = persistence
        self._model_factory = model_factory
        self._config = config
        self._poll_interval = poll_interval
        self.last_status: str = ""
        # ``_portfolio_ids`` / ``_runstate`` are headless-test hooks mirroring the
        # views' ``render_count``.
        self._portfolio_ids: list[uuid.UUID] = []
        self._runstate = "✓ idle"
        self._narrow = False

    # --- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Yield the header, the sidebar | content split, the status bar, and footer."""
        yield Header()
        with Horizontal(id="shell-header", classes="shell-header"):
            yield Select([], prompt="⟨fund-wide⟩", id="hdr-portfolio", allow_blank=True)
            yield Static("● AUTONOMOUS", id="hdr-mode")
            yield Static(self._runstate, id="hdr-runstate")
        with Horizontal(id="shell-body", classes="shell-body"):
            yield ListView(
                *(ListItem(Label(title), id=f"nav-{key}") for key, title in VIEW_SPECS),
                id="sidebar",
                classes="sidebar",
            )
            with ContentSwitcher(
                initial="view-posture", id="content", classes="content"
            ):
                yield from build_views()
        yield Static("", id="status-bar", classes="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        # Seed the picker, then paint once the views have mounted and poll on the
        # cockpit's cadence (the placeholder views' refresh is a no-op counter bump
        # today). The mount-time ``Select.Changed`` the seeding echoes is dropped by
        # ``on_select_changed``'s value-equality guard, not a mount-window flag.
        self._populate_picker()
        self._refresh_runstate()
        if self.size.width:
            self._apply_width(self.size.width)
        self.call_after_refresh(self._refresh)
        self.set_interval(self._poll_interval, self._refresh)

    def on_resize(self, event: Resize) -> None:
        """Collapse/restore the sidebar rail as the terminal crosses the threshold."""
        self._apply_width(event.size.width)

    def _apply_width(self, width: int) -> None:
        """Collapse the sidebar to a single-letter rail below the narrow threshold.

        At narrow widths the sidebar shrinks (``.narrow`` CSS) and each entry's
        label drops to its leading letter, leaving the content pane effectively
        single-pane; above the threshold the full titles and width return.
        """
        self._narrow = width < _NARROW_WIDTH
        try:
            sidebar = self.query_one("#sidebar", ListView)
        except NoMatches:
            return
        sidebar.set_class(self._narrow, "narrow")
        for key, title in VIEW_SPECS:
            self.query_one(f"#nav-{key}", ListItem).query_one(Label).update(
                title[0] if self._narrow else title
            )

    # --- navigation + refresh ----------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Switch the content pane to the picked sidebar entry and repaint it.

        The sidebar ``ListItem`` ids are ``nav-<key>`` and the matching views are
        ``view-<key>`` (see :data:`~fund.tui.views.VIEW_SPECS`), so a pick maps
        straight to the ``ContentSwitcher`` target. An unrecognised id is ignored.
        """
        item_id = event.item.id or ""
        if not item_id.startswith("nav-"):
            return
        self.query_one(ContentSwitcher).current = f"view-{item_id[len('nav-') :]}"
        self._refresh()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Re-focus the shell on the picked portfolio (or fund-wide when blank).

        A no-op pick is dropped: seeding the picker at mount echoes a
        ``Select.Changed`` whose value equals the current focus, and Textual
        dispatches it *after* ``on_mount`` returns — so the guard is value equality,
        not a mount-window flag (async dispatch would already have cleared the flag).
        A real change updates the run-state glyph and repaints. The no-selection
        sentinel is ``Select.NULL`` (the ``NoSelection`` singleton) — *not*
        ``Select.BLANK``, which resolves to ``Widget.BLANK is False`` and would send a
        blank pick down ``uuid.UUID("Select.NULL")``.
        """
        new_id = None if event.value is Select.NULL else uuid.UUID(str(event.value))
        if new_id == self._portfolio_id:
            return
        self._portfolio_id = new_id
        self._refresh_runstate()
        self._refresh()

    def _refresh(self) -> None:
        """Repaint the active view from the read model — synchronous, test-callable.

        Mirrors ``FundTUI._refresh``: a single sync entry point a headless test can
        call directly instead of racing the poll timer. Delegates to the visible
        :class:`~fund.tui.views.ShellView` so each view owns its own repaint.
        """
        current = self.query_one(ContentSwitcher).current
        if current is None:
            return
        self.query_one(f"#{current}", ShellView).refresh_view()

    # --- header picker + run state -----------------------------------------

    def _populate_picker(self) -> None:
        """Seed the header ``Select`` from the fund's known portfolios (T2 reader).

        Reads ``observe.list_portfolios`` through a short read-only session and
        records the ids on :attr:`_portfolio_ids` (a test hook). When the shell
        launched focused on a known portfolio, that option is pre-selected.
        """
        self._portfolio_ids = self._list_portfolios()
        select = self.query_one("#hdr-portfolio", Select)
        select.set_options(
            (self._short_label(pid), str(pid)) for pid in self._portfolio_ids
        )
        if self._portfolio_id in self._portfolio_ids:
            select.value = str(self._portfolio_id)

    def _list_portfolios(self) -> list[uuid.UUID]:
        """Distinct portfolio ids for the picker; ``[]`` on any read error."""
        try:
            with self._session_factory() as session:
                ids = observe.list_portfolios(session)
                session.rollback()
                return ids
        except Exception:  # keep the shell alive on a read error (mirrors the cockpit)
            return []

    def _refresh_runstate(self) -> None:
        """Repaint the header run-state glyph for the focused portfolio."""
        self._runstate = self._run_state_glyph()
        self.query_one("#hdr-runstate", Static).update(self._runstate)

    def _run_state_glyph(self) -> str:
        """The focused portfolio's latest-run glyph: running / awaiting / idle.

        Fund-wide (no focus), no runs, or a read error all read as ``idle``; a
        ``paused`` latest run is ``awaiting`` the adviser, ``pending``/``running``
        is live, everything else (completed / rejected) is idle.
        """
        if self._portfolio_id is None:
            return "✓ idle"
        try:
            with self._session_factory() as session:
                runs = observe.list_portfolio_runs(session, self._portfolio_id)
                session.rollback()
        except Exception:
            return "✓ idle"
        if not runs:
            return "✓ idle"
        status = runs[0].status
        if status == "paused":
            return "⏸ awaiting"
        if status in ("pending", "running"):
            return "▶ running"
        return "✓ idle"

    def _short_label(self, portfolio_id: uuid.UUID) -> str:
        """A compact ``⟨abcd1234⟩`` picker label (no portfolio names persisted yet)."""
        return f"⟨{str(portfolio_id)[:8]}⟩"

    def _set_status(self, message: str) -> None:
        self.last_status = message
        self.query_one("#status-bar", Static).update(message)


def main() -> None:
    """``fund-tui`` entrypoint: build persistence, launch the shell for one run.

    The portfolio id is **optional**: no arg opens the fund-wide shell (Posture
    landing); ``fund-tui <uuid>`` focuses that portfolio. Needs ``DATABASE_URL``
    (for the checkpointer/store pool); the chat model stays lazy — only built when
    the adviser acts — so a bare launch needs no ``OLLAMA_API_KEY``. The pool is
    always closed on exit. ``typer`` is imported here (not at module top) to keep
    ``import fund.tui.app`` light.
    """
    import typer

    def _launch(portfolio_id: str | None = typer.Argument(default=None)) -> None:
        from fund.agents.model import build_primary
        from fund.audit import setup_langgraph
        from fund.database import get_session

        focus = uuid.UUID(portfolio_id) if portfolio_id else None
        persistence = setup_langgraph(settings)
        try:
            app = ShellApp(
                focus,
                session_factory=get_session,
                persistence=persistence,
                model_factory=lambda: build_primary(settings),
            )
            app.run()
        finally:
            persistence.pool.close()

    typer.run(_launch)


if __name__ == "__main__":
    main()
