"""View bodies for the multi-view shell + the data-view context seam.

Each view is a panel the :class:`~fund.tui.app.ShellApp` mounts in its
``ContentSwitcher``. Phase 0 renders labeled placeholders exposing a ``render_count``
test counter (mirroring the cockpit panels' ``entry_count`` / ``event_count`` hooks);
later phases fill them with real content — Deliberation with the live transcript,
Decisions with the HITL queue, and so on.

Data-views reach the App's injected collaborators through the
:class:`ShellView` context seam — the narrowed :attr:`ShellView.shell` accessor and
the :meth:`ShellView.read_session` short-session helper — rather than importing
``fund.tui.app`` (a circular import, and the App module is heavier). At module scope
this imports only Textual, the stdlib, and the agent-stack-free ``fund.observe`` read
model, so ``import fund.tui.app`` stays agent-stack-light (the import-hygiene guard in
``test_tui.py``); the agent stack (``fund.actions`` / audit repos) is imported lazily
inside method bodies.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Label, Static

from fund import observe
from fund.observe import RunSummary

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session


class _ShellContext(Protocol):
    """The slice of :class:`~fund.tui.app.ShellApp` the data-views read each refresh.

    A *structural* view of the hosting App (not an import of it), so
    ``fund.tui.views`` need not import ``fund.tui.app`` — which would be a circular
    import and would risk dragging heavier collaborators into this agent-stack-light
    view module.

    Attributes:
        _session_factory: Opens a context-managed sync session for a read tick.
        _persistence: LangGraph bundle (``.saver`` / ``.store``) the reads cross-check.
        _portfolio_id: The focused portfolio, or ``None`` for the fund-wide landing.
    """

    _session_factory: Callable[[], AbstractContextManager[Session]]
    _persistence: Any
    _portfolio_id: uuid.UUID | None


# The six shell views in sidebar order: (key, human title). The key is the id
# fragment for both the sidebar entry (``nav-<key>``) and the view (``view-<key>``).
VIEW_SPECS: tuple[tuple[str, str], ...] = (
    ("posture", "Posture"),
    ("deliberation", "Deliberation"),
    ("decisions", "Decisions"),
    ("mandate", "Mandate"),
    ("audit", "Audit"),
    ("ask", "Ask"),
)


class ShellView(Vertical):
    """A dumb placeholder view: a title line, a body line, and a refresh counter.

    Attributes:
        view_key: Stable id fragment, shared with the sidebar entry key.
        view_title: Human label shown at the top of the view.
        render_count: Incremented by :meth:`refresh_view` — a headless-test hook so a
            test can prove the active view was repainted without scraping the DOM.
    """

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.view_key = view_key
        self.view_title = view_title
        self.render_count = 0

    def compose(self) -> ComposeResult:
        """Yield the view's title label and its placeholder body line."""
        yield Label(self.view_title, classes="view-title")
        yield Static(
            f"[{self.view_key}] — placeholder (Phase 0 scaffold)",
            classes="view-body",
            id=f"body-{self.view_key}",
        )

    def refresh_view(self) -> None:
        """Repaint the view from fresh read-model state.

        A Phase-0 no-op beyond bumping :attr:`render_count`; later phases override
        this to pull the view's data through ``fund.observe`` on each poll tick.
        """
        self.render_count += 1

    # --- context seam -------------------------------------------------------

    @property
    def shell(self) -> _ShellContext:
        """The hosting ``ShellApp``, narrowed to the collaborators a view reads.

        A typed accessor over ``self.app`` so a data-view's :meth:`refresh_view` can
        read ``self.shell._portfolio_id`` / ``._persistence`` without importing the
        App (see :class:`_ShellContext`).
        """
        return cast("_ShellContext", self.app)

    @contextmanager
    def read_session(self) -> Generator[Session]:
        """Yield a short read-only session from the shell, rolled back on exit.

        The read discipline every data-view shares (mirrors ``FundTUI._refresh``):
        open a session from the shell's injected factory, read the model, and roll
        back — the UI never holds a transaction across poll ticks. The caller does
        not commit.

        Yields:
            An injected sync session, valid only inside the ``with`` block.
        """
        with self.shell._session_factory() as session:
            try:
                yield session
            finally:
                session.rollback()


class DecisionsView(ShellView):
    """The global HITL inbox: a cross-portfolio queue of paused gates + gate detail.

    Reads ``observe.pending_hitl(session, saver, None)`` — fund-wide, so no awaiting
    gate slips regardless of the header focus — into a queue table; selecting a row
    reveals the full gate context (gate name, portfolio, the proposed orders) and a
    "why" line. ``a`` / ``r`` resolve the selected gate through
    :func:`fund.actions.apply_action` on a worker thread (dispatched by the App), and
    ``k`` killswitches it (finalise rejected, no undo) behind an explicit-consequence
    confirm; ``e`` / ``s`` (edit / steer) are rendered but inert — they land with the
    co-pilot (``03-copilot``).

    Attributes:
        selected_run_id: The focused queued run, or ``None`` when nothing is selected.
        selected_portfolio_id: That run's portfolio (for the audit attribution).
        gate_text: Mirror of the rendered gate detail — a headless-test hook.
    """

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.selected_run_id: uuid.UUID | None = None
        self.selected_portfolio_id: uuid.UUID | None = None
        self.gate_text: str = ""
        self._queue: list[RunSummary] = []

    def compose(self) -> ComposeResult:
        """Yield the title + awaiting count, the queue | gate split, and a key hint."""
        yield Label(self.view_title, classes="view-title")
        yield Label("0 awaiting", id="decisions-count", classes="view-subtitle")
        with Horizontal(classes="decisions-body"):
            yield DataTable(
                id="decisions-table", cursor_type="row", classes="decisions-queue"
            )
            yield Static(
                "(select a paused gate)", id="decisions-gate", classes="decisions-gate"
            )
        yield Static(
            "a approve · r reject · k killswitch · e edit · s steer"
            "  — e/s arrive with the co-pilot",
            id="decisions-hint",
            classes="decisions-hint",
        )

    def on_mount(self) -> None:
        """Seed the queue table's columns once its DataTable has mounted."""
        self.query_one("#decisions-table", DataTable).add_columns(
            "run", "portfolio", "asof"
        )

    def refresh_view(self) -> None:
        """Repaint the cross-portfolio queue and the selected gate detail.

        Reads fund-wide ``pending_hitl`` each tick; the selected gate is re-read only
        while it is still in the live queue.
        """
        self.render_count += 1
        rows, gate = self._read()
        self._paint_queue(rows)
        self._paint_gate(gate)

    def select_run(self, run_id: uuid.UUID) -> None:
        """Focus a queued run by id, then repaint the gate detail.

        A driver/test hook mirroring the DataTable row selection: the portfolio is
        taken from the last-painted queue so no extra read is needed.
        """
        summary = next((row for row in self._queue if row.run_id == run_id), None)
        self.selected_run_id = run_id
        self.selected_portfolio_id = summary.portfolio_id if summary else None
        self.refresh_view()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A queue-row selection focuses that gate and reveals its detail."""
        key = event.row_key.value
        if key is not None:
            self.select_run(uuid.UUID(key))

    def _read(self) -> tuple[list[RunSummary], str | None]:
        """Read the queue + the selected run's gate text within one short session.

        Returns ``([], None)`` on any read error so the poll loop stays alive. The
        gate text is read only when the selection is still in the live queue.
        """
        saver = self.shell._persistence.saver
        try:
            with self.read_session() as session:
                rows = observe.pending_hitl(session, saver, None)
                gate: str | None = None
                if self.selected_run_id is not None and any(
                    row.run_id == self.selected_run_id for row in rows
                ):
                    gate = _gate_text(session, saver, self.selected_run_id)
                return rows, gate
        except Exception:  # keep the poll loop alive on any read error
            return [], None

    def _paint_queue(self, rows: list[RunSummary]) -> None:
        self._queue = rows
        table = self.query_one("#decisions-table", DataTable)
        table.clear()
        for row in rows:
            table.add_row(
                _short(row.run_id),
                _short(row.portfolio_id) if row.portfolio_id else "fund-wide",
                row.asof.isoformat(),
                key=str(row.run_id),
            )
        self.query_one("#decisions-count", Label).update(f"{len(rows)} awaiting")
        # Drop a selection that has left the queue (e.g. the gate we just resolved).
        if self.selected_run_id is not None and all(
            row.run_id != self.selected_run_id for row in rows
        ):
            self.selected_run_id = None
            self.selected_portfolio_id = None

    def _paint_gate(self, gate: str | None) -> None:
        self.gate_text = gate or ""
        self.query_one("#decisions-gate", Static).update(
            gate or "(select a paused gate)"
        )


def _gate_text(session: Any, saver: Any, run_id: uuid.UUID) -> str | None:
    """Render the selected run's gate detail, or ``None`` when the run is gone.

    Needs the ORM run row (for its ``thread_id``) that ``observe.interrupt_for``
    reads, so the run repo is imported lazily from its submodule — never the
    ``fund.audit`` root (which pulls the langgraph bootstrap).
    """
    from fund.audit.repository import AgentRunRepository

    run = AgentRunRepository(session).get_run(run_id)
    if run is None:
        return None
    return _format_decision_gate(observe.interrupt_for(saver, run), run)


def _format_decision_gate(interrupt: dict[str, Any] | None, run: Any) -> str:
    """Render ``run``'s pending gate: header, the "why", and the proposed orders."""
    pf = _short(run.portfolio_id) if run.portfolio_id else "fund-wide"
    requests = (interrupt or {}).get("action_requests") or []
    request = requests[0] if requests else {}
    name = request.get("name", "gate")
    lines = [f"GATE · {name} · #{_short(run.id)} · {pf}"]
    why = request.get("description")
    lines.append(f"why: {why}" if why else "why: review before it executes")
    weights = (request.get("args") or {}).get("weights")
    if isinstance(weights, dict) and weights:
        lines.append("orders (target weights):")
        lines.extend(
            f"  {ticker}: {weight:.4f}" for ticker, weight in sorted(weights.items())
        )
    return "\n".join(lines)


def _short(value: Any) -> str:
    """First hex group of a UUID (or its string) — a compact table cell."""
    return str(value).split("-", 1)[0]


# Per-key view class; keys absent here fall back to the placeholder ``ShellView``.
_VIEW_CLASSES: dict[str, type[ShellView]] = {
    "decisions": DecisionsView,
}


def build_views() -> list[ShellView]:
    """Build one view per :data:`VIEW_SPECS`, id'd ``view-<key>``.

    Picks the real view class for keys in :data:`_VIEW_CLASSES` (currently
    ``decisions``); every other key still renders the placeholder ``ShellView``.
    """
    return [
        _VIEW_CLASSES.get(key, ShellView)(key, title, id=f"view-{key}")
        for key, title in VIEW_SPECS
    ]


__all__ = ["VIEW_SPECS", "DecisionsView", "ShellView", "build_views"]
