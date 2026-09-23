"""``fund-tui`` — the four-panel Textual observer over :mod:`fund.observe` (Task 8).

A read-mostly "Bloomberg" cockpit for one portfolio: a transcript, the current-vs-
target state, the HITL approval queue, and the run history. All four panels are thin
over the model-free read model (:mod:`fund.observe`); the App owns every side effect:

* **Polling, not pushing.** :meth:`FundTUI._refresh` runs on a ``set_interval`` timer
  (``poll_interval``, ~2s), opening a *short read-only* session, extracting plain
  dataclasses, and rolling back — the UI never holds a transaction or an ORM row
  across ticks.
* **Resume off the event loop.** Approve/Reject dispatch :meth:`_resume_worker`, a
  ``@work(thread=True)`` worker: the synchronous ``resume_run`` (which dispatches to
  the profiler or rebalance resumer by gate and drives ``agent.invoke``) runs on a
  background thread so the event loop never blocks, then marshals its UI updates back
  with ``call_from_thread``.
* **Lazy model, light import.** ``import fund.tui.app`` pulls only Textual + the
  agent-stack-free read model — no ``deepagents`` / ``langgraph`` / model, so it needs
  no ``OLLAMA_API_KEY`` and no ``DATABASE_URL``. The chat model is built lazily via the
  injected ``model_factory`` only when the adviser approves/rejects; the persistence
  handles (checkpointer + store, needing ``DATABASE_URL``) are built in :func:`main`.

Every collaborator is injected (``session_factory`` / ``persistence`` /
``model_factory``), so :meth:`App.run_test` can drive the whole cockpit headlessly
over SQLite + ``MemorySaver`` + a scripted model with zero network.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static

from fund import observe
from fund.config import FundConfig, settings
from fund.tui.widgets import (
    HistoryPanel,
    HitlQueuePanel,
    PortfolioStatePanel,
    TranscriptPanel,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.widgets import DataTable
    from textual.worker import Worker

_PAST = {"approve": "approved", "reject": "rejected"}


class FundTUI(App):
    """The four-panel observer App for a single portfolio (Phase 8, Task 8)."""

    CSS = """
    Screen { layout: grid; grid-size: 2 2; grid-gutter: 1; }
    .panel { border: round $accent; padding: 0 1; height: 1fr; }
    .panel-title { text-style: bold; color: $accent; }
    #transcript-log { height: 1fr; }
    #gate-detail { height: auto; color: $warning; }
    #hitl-buttons { height: auto; align-horizontal: left; }
    Button { margin: 0 1; }
    #status-bar { dock: bottom; height: 1; background: $panel; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
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


def main() -> None:
    """``fund-tui`` entrypoint: build persistence, launch the App for one portfolio.

    Needs ``DATABASE_URL`` (for the checkpointer/store pool); the chat model stays
    lazy — only built when the adviser approves/rejects — so a bare launch needs no
    ``OLLAMA_API_KEY``. The pool is always closed on exit.
    """
    import argparse

    from fund.agents.model import build_primary
    from fund.audit import setup_langgraph
    from fund.database import get_session

    parser = argparse.ArgumentParser(
        prog="fund-tui", description="Fund observer TUI (read + approve/reject)."
    )
    parser.add_argument("portfolio_id", help="Portfolio UUID to observe.")
    args = parser.parse_args()
    portfolio_id = uuid.UUID(args.portfolio_id)

    persistence = setup_langgraph(settings)
    try:
        app = FundTUI(
            portfolio_id,
            session_factory=get_session,
            persistence=persistence,
            model_factory=lambda: build_primary(settings),
        )
        app.run()
    finally:
        persistence.pool.close()


if __name__ == "__main__":
    main()
