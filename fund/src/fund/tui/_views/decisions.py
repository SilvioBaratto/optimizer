"""The Decisions view: the queue of paused HITL gates and the gate it details."""

from __future__ import annotations

import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Label, Static
from textual.widgets.data_table import RowDoesNotExist

from fund import observe
from fund.observe import RunSummary
from fund.tui._views.shell import ShellView, _keep_cursor, _short

if TYPE_CHECKING:
    from textual.binding import BindingType


class DecisionsView(ShellView):
    """The global HITL inbox: a cross-portfolio queue of paused gates + gate detail.

    Reads ``observe.pending_hitl(session, saver, None)`` — fund-wide, so no awaiting
    gate slips regardless of the header focus — into a queue table; selecting a row
    reveals the full gate context (gate name, portfolio, the proposed orders) and a
    "why" line. ``a`` / ``r`` resolve the selected gate through
    :func:`fund.actions.apply_action` on a worker thread (dispatched by the App), and
    ``k`` killswitches it (finalise rejected, no undo) behind an explicit-consequence
    confirm; ``e`` edits the proposed weights (``apply_action("edit")``) and ``s``
    steers the PM with a directive (``respond``), both dispatched by the App. Every one
    of these acts on the selected gate and is refused while the cursor rests on a
    different row, so the highlighted gate is always the one that gets resolved.

    Attributes:
        selected_run_id: The focused queued run, or ``None`` when nothing is selected.
        selected_portfolio_id: That run's portfolio (for the audit attribution).
        gate_text: Mirror of the rendered gate detail — a headless-test hook.
        gate_unavailable: ``True`` while the selected gate's detail could not be
            rendered. ``a`` / ``r`` fire with no confirm, so they are refused until the
            orders are back on screen.
    """

    # View-scoped so ``e``/``s`` mean edit/steer only while Decisions holds focus; the
    # App's global ``e`` (Audit export) is untouched (R5). The ``app.`` prefix routes to
    # the App's action (the handlers live there, not on the view).
    BINDINGS: ClassVar[list[BindingType]] = [
        ("e", "app.edit_decision", "Edit weights"),
        ("s", "app.steer_decision", "Steer PM"),
    ]

    primary_focus: ClassVar[str | None] = "#decisions-table"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.selected_run_id: uuid.UUID | None = None
        self.selected_portfolio_id: uuid.UUID | None = None
        self.gate_text: str = ""
        self.gate_unavailable = False
        # The gate on screen, so a decision is refused if the run has since moved on.
        self.rendered_gate_id: str | None = None
        self._queue: list[RunSummary] = []

    def compose(self) -> ComposeResult:
        """Yield the title + awaiting count, the queue | gate split, and a key hint."""
        yield Label(self.view_title, classes="view-title")
        yield Label("0 awaiting", id="decisions-count", classes="view-subtitle")
        with Horizontal(classes="decisions-body"):
            yield DataTable(
                id="decisions-table", cursor_type="row", classes="decisions-queue"
            )
            # The gate's "why" is model prose; as markup, "[BLOCK]" would vanish and a
            # stray "[/x]" would raise MarkupError. The panes below follow suit.
            yield Static(
                "(select a paused gate)",
                id="decisions-gate",
                classes="decisions-gate",
                markup=False,
            )
        yield Static(
            "↵ select · a approve · r reject · k killswitch · e edit weights · "
            "s steer PM",
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
        read = self._read()
        if read is None:
            self._paint_read_error()
            return
        rows, gate, unavailable, gate_id = read
        self._paint_queue(rows)
        self._paint_gate(gate, unavailable=unavailable, gate_id=gate_id)

    def select_run(self, run_id: uuid.UUID) -> None:
        """Focus a queued run by id, then repaint the gate detail.

        A driver/test hook mirroring the DataTable row selection: the portfolio is
        taken from the last-painted queue so no extra read is needed. The table cursor
        moves onto the selected row, so the highlight and the selection never diverge.
        """
        summary = next((row for row in self._queue if row.run_id == run_id), None)
        self.selected_run_id = run_id
        self.selected_portfolio_id = summary.portfolio_id if summary else None
        self.refresh_view()
        table = self.query_one("#decisions-table", DataTable)
        with suppress(RowDoesNotExist):
            table.move_cursor(row=table.get_row_index(str(run_id)))

    def highlighted_run_id(self) -> uuid.UUID | None:
        """The queued run under the table cursor, or ``None`` for an empty queue."""
        table = self.query_one("#decisions-table", DataTable)
        if not table.row_count:
            return None
        key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return uuid.UUID(key) if key is not None else None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A queue-row selection focuses that gate and reveals its detail."""
        key = event.row_key.value
        if key is not None:
            self.select_run(uuid.UUID(key))

    def _read(self) -> tuple[list[RunSummary], str | None, bool, str | None] | None:
        """Read the queue + the selected run's gate text within one short session.

        Returns ``None`` on a queue read error — distinct from an empty queue, so the
        poll keeps the last-known queue and selection instead of mistaking a hiccup for
        a cleared inbox. The gate text is read only when the selection is still in the
        live queue, and a gate that cannot be rendered degrades to a one-line notice
        (flagged by the third element) without discarding the queue. The fourth element
        is the id of the gate rendered, read before its text: a run that moves between
        the two reads then leaves an older id than the text, and a decision on it is
        refused instead of landing on a gate nobody saw.
        """
        saver = self.shell._persistence.saver
        try:
            with self.read_session() as session:
                rows = observe.pending_hitl(session, saver, None)
                gate: str | None = None
                gate_id: str | None = None
                unavailable = False
                if self.selected_run_id is not None and any(
                    row.run_id == self.selected_run_id for row in rows
                ):
                    try:
                        gate_id = _live_gate_id(session, saver, self.selected_run_id)
                        gate = _gate_text(session, saver, self.selected_run_id)
                    except Exception as exc:  # one bad gate must not hide the queue
                        gate = f"(gate unavailable: {exc})"
                        gate_id = None
                        unavailable = True
                return rows, gate, unavailable, gate_id
        except Exception:  # keep the poll loop alive on any read error
            return None

    def _paint_read_error(self) -> None:
        """Flag a failed read next to the count, leaving queue + selection as-is."""
        self.query_one("#decisions-count", Label).update(
            f"{len(self._queue)} awaiting · ⚠ read failed, showing last known queue"
        )

    def _paint_queue(self, rows: list[RunSummary]) -> None:
        self._queue = rows
        table = self.query_one("#decisions-table", DataTable)
        with _keep_cursor(table):
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

    def _paint_gate(
        self,
        gate: str | None,
        *,
        unavailable: bool = False,
        gate_id: str | None = None,
    ) -> None:
        self.gate_text = gate or ""
        self.gate_unavailable = unavailable
        self.rendered_gate_id = gate_id
        self.query_one("#decisions-gate", Static).update(
            gate or "(select a paused gate)"
        )


def _live_gate_id(session: Any, saver: Any, run_id: uuid.UUID) -> str | None:
    """The id of the gate ``run_id`` is parked at, or ``None`` when it has none.

    Lazy import of the run repository for the same reason as :func:`_gate_text`.
    """
    from fund.audit.repository import AgentRunRepository

    return observe.live_gate_id(saver, AgentRunRepository(session).get_run(run_id))


def _gate_text(session: Any, saver: Any, run_id: uuid.UUID) -> str | None:
    """Render the selected run's gate detail, or ``None`` when the run is gone.

    Needs the ORM run row (for its ``thread_id``) that ``observe.interrupt_for``
    reads, so the run repo is imported lazily from its submodule, keeping
    ``fund.tui.views`` free of the ``fund.audit`` package at import time.
    """
    from fund.audit.repository import AgentRunRepository

    repo = AgentRunRepository(session)
    run = repo.get_run(run_id)
    if run is None:
        return None
    return _format_decision_gate(
        observe.interrupt_for(saver, run), run, repo.latest_optimizer_weights(run_id)
    )


# The orders block prints weights to four decimals; a PM/audit gap below half a unit
# in the last place is invisible to the adviser and not worth a warning.
_WEIGHT_DISPLAY_TOLERANCE = 5e-5


def _pm_weights_differ(pm_weights: Any, audited: dict[str, float]) -> bool:
    """Whether the PM's ``place_orders`` weights disagree with the audited vector.

    A missing ticker counts as weight 0 and anything non-numeric counts as a
    disagreement, since the PM's tool-call args are unvalidated until after the gate.
    """
    if pm_weights is None:
        return False
    if not isinstance(pm_weights, dict):
        return True
    try:
        return any(
            not abs(
                float(pm_weights.get(ticker, 0.0)) - float(audited.get(ticker, 0.0))
            )
            <= _WEIGHT_DISPLAY_TOLERANCE
            for ticker in pm_weights.keys() | audited.keys()
        )
    except (TypeError, ValueError):
        return True


def _format_decision_gate(
    interrupt: dict[str, Any] | None, run: Any, audited: dict[str, float]
) -> str:
    """Render ``run``'s pending gate: header, the "why", and the orders it executes.

    The orders block shows ``audited`` — the allocator's logged optimizer weights,
    which ``place_orders`` places on approval — not the PM's tool-call args, which
    that tool ignores. A PM vector that disagrees is flagged instead of shown, so the
    adviser never approves a vector other than the one that executes.

    Args:
        interrupt: The run's pending HITL payload (``action_requests``), or ``None``.
        run: The ORM run row (``id`` / ``portfolio_id``) for the header.
        audited: The run's latest audited optimizer weights (``{}`` when the allocator
            never produced a proposal).
    """
    pf = _short(run.portfolio_id) if run.portfolio_id else "fund-wide"
    requests = (interrupt or {}).get("action_requests") or []
    request = requests[0] if requests else {}
    name = request.get("name", "gate")
    lines = [f"GATE · {name} · #{_short(run.id)} · {pf}"]
    why = request.get("description")
    lines.append(f"why: {why}" if why else "why: review before it executes")
    args = request.get("args") or {}
    lines.extend(_format_risk_gate(args.get("risk_gate")))
    if name == "place_orders" or "weights" in args:
        if audited:
            lines.append("orders (audited optimizer weights, executed on approval):")
            lines.extend(
                f"  {ticker}: {weight:.4f}"
                for ticker, weight in sorted(audited.items())
            )
        else:
            lines.append(
                "orders: no audited optimizer weights, approval places no ticket"
            )
        if _pm_weights_differ(args.get("weights"), audited):
            lines.append(
                "⚠ PM-proposed weights differ from the audited vector; "
                "approval executes the audited one"
            )
    return "\n".join(lines)


_RISK_GATE_LABELS: dict[str, str] = {
    "passed": "passed",
    "failed": "FAILED",
    "stale": "STALE (the proposal changed after risk checked it)",
    "not_run": "NOT CHECKED",
}


def _format_risk_gate(gate: Any) -> list[str]:
    """Render the risk verdict pinned on a ``place_orders`` gate (empty when absent).

    A gate that is not ``passed`` is one ``place_orders`` will refuse, so it is
    spelled out in words rather than left to a colour or glyph.
    """
    if not isinstance(gate, dict) or gate.get("status") not in _RISK_GATE_LABELS:
        return []
    lines = [f"risk gate: {_RISK_GATE_LABELS[gate['status']]}"]
    # A stale verdict's breaches belong to a superseded vector, so they are not shown.
    violations = gate.get("violations") if gate["status"] == "failed" else None
    for violation in violations or []:
        if isinstance(violation, dict):
            keys = ("type", "asset", "sector", "reason")
            detail = " ".join(str(violation[k]) for k in keys if k in violation)
            lines.append(f"  - {detail or 'violation'}")
    return lines
