"""The Decisions view: the queue of paused HITL gates and the gate it details."""

from __future__ import annotations

import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Label, Static
from textual.widgets.data_table import RowDoesNotExist

from fund import observe
from fund.observe import RunSummary
from fund.tui._views.candidates import (
    CANDIDATE_COLUMNS,
    ORDER_GATE,
    candidate_row,
    decided_gate,
    find_candidate,
    gate_names,
    is_selectable,
    payload_set_hash,
    pick_line,
    refused_pick_line,
)
from fund.tui._views.shell import ShellView, _keep_cursor, _short

if TYPE_CHECKING:
    from collections.abc import Mapping

    from textual.binding import BindingType

_QUEUE_TABLE = "decisions-table"
_CANDIDATES_TABLE = "decisions-candidates"

_KEY_HINT = (
    "↵ select gate · tab candidates, ↵ select one · a approve (a rebalance gate "
    "confirms the selected candidate) · r reject · k killswitch · e edit weights · "
    "s steer the run"
)
_STALE_SET = (
    "cannot approve: the gate was opened on another candidate set than the run's "
    "latest one - reject the run, or wait for it to re-gate"
)
_NO_SET = "cannot approve: this gate has no candidate set - reject the run"
_NO_PICK = (
    "select a feasible candidate row (enter) before approving - nothing is chosen "
    "for you"
)


@dataclass(frozen=True)
class _GateDetail:
    """What one read of the selected gate found, before it is painted.

    Attributes:
        text: The rendered gate detail, or ``None`` when nothing is selected.
        unavailable: Whether the gate could not be read (``text`` says why).
        gate_id: The id of the gate the run is parked at.
        gate_name: The gated tool an approve decides (see :func:`decided_gate`).
        payload_set_hash: The candidate-set digest the gate payload was opened with.
        set_hash: The digest of the run's newest stored candidate set.
        candidates: That set's candidates as JSON dumps, in id order; empty off the
            order gate.
    """

    text: str | None = None
    unavailable: bool = False
    gate_id: str | None = None
    gate_name: str | None = None
    payload_set_hash: str | None = None
    set_hash: str | None = None
    candidates: tuple[dict[str, Any], ...] = ()


class DecisionsView(ShellView):
    """The global HITL inbox: a cross-portfolio queue of paused gates + gate detail.

    Reads ``observe.pending_hitl(session, saver, None)`` — fund-wide, so no awaiting
    gate slips regardless of the header focus — into a queue table; selecting a row
    reveals the full gate context (gate name, portfolio, its candidates) and a "why"
    line. A rebalance gate also lists its candidates in a table of their own, in id
    order and never sorted; the user selects one row (``↵``) and ``a`` approves
    it behind a confirm naming it. Nothing is selected for them, and an infeasible
    row cannot be selected. At any other gate ``a`` approves at once, without a
    candidate. ``r`` rejects, ``k`` killswitches (finalise rejected, no undo) behind
    an explicit-consequence confirm, ``e`` edits weights starting from the selected
    candidate (``apply_action("edit")``) and ``s`` steers the PM with a directive
    (``respond``); the App dispatches each on a worker. Every one of these acts on
    the selected gate and is refused while the cursor rests on a different row, so
    the highlighted gate is always the one that gets resolved.

    Attributes:
        selected_run_id: The focused queued run, or ``None`` when nothing is selected.
        selected_portfolio_id: That run's portfolio (for the audit attribution).
        gate_text: Mirror of the rendered gate detail — a headless-test hook.
        gate_unavailable: ``True`` while the selected gate's detail could not be
            rendered; every decision is refused until it is back on screen.
        rendered_gate_id: The id of the gate on screen, so a decision is refused if
            the run has since moved on.
        rendered_gate_name: The gated tool the gate on screen decides.
        rendered_set_hash: The digest of the candidate set the table shows.
        selected_candidate_id: The candidate the user selected for the gate on
            screen; ``None`` until they select one. It is cleared when the selected
            run, the gate id or the candidate set changes, so a selection never
            carries over to candidates the user did not see.
    """

    # View-scoped so ``e``/``s`` mean edit/steer only while Decisions holds focus; the
    # App's global ``e`` (Audit export) is untouched (R5). The ``app.`` prefix routes to
    # the App's action (the handlers live there, not on the view).
    BINDINGS: ClassVar[list[BindingType]] = [
        ("e", "app.edit_decision", "Edit weights"),
        ("s", "app.steer_decision", "Steer the run"),
    ]

    primary_focus: ClassVar[str | None] = f"#{_QUEUE_TABLE}"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.selected_run_id: uuid.UUID | None = None
        self.selected_portfolio_id: uuid.UUID | None = None
        self.gate_text: str = ""
        self.gate_unavailable = False
        self.rendered_gate_id: str | None = None
        self.rendered_gate_name: str | None = None
        self.rendered_set_hash: str | None = None
        self.selected_candidate_id: str | None = None
        self._payload_set_hash: str | None = None
        self._candidates: tuple[dict[str, Any], ...] = ()
        # The (gate id, set hash) a selection was made on; any other pair clears it.
        self._pick_anchor: tuple[str | None, str | None] | None = None
        self._queue: list[RunSummary] = []

    def compose(self) -> ComposeResult:
        """Yield the title + awaiting count, the queue | gate split, and a key hint."""
        yield Label(self.view_title, classes="view-title")
        yield Label("0 awaiting", id="decisions-count", classes="view-subtitle")
        with Horizontal(classes="decisions-body"):
            yield DataTable(
                id=_QUEUE_TABLE, cursor_type="row", classes="decisions-queue"
            )
            with Vertical(classes="decisions-detail"):
                yield from self._compose_gate_detail()
        yield Static(_KEY_HINT, id="decisions-hint", classes="decisions-hint")

    def _compose_gate_detail(self) -> ComposeResult:
        """Yield the gate detail, its candidate table and the selection line."""
        # The gate's "why" is model prose; as markup, "[BLOCK]" would vanish and a
        # stray "[/x]" would raise MarkupError. The selection line follows suit.
        yield Static(
            "(select a paused gate)",
            id="decisions-gate",
            classes="decisions-gate",
            markup=False,
        )
        # No cursor until the user moves into the table, so no candidate sits
        # highlighted by default.
        yield DataTable(
            id=_CANDIDATES_TABLE,
            cursor_type="row",
            show_cursor=False,
            classes="decisions-candidates",
        )
        yield Static("", id="decisions-pick", classes="decisions-pick", markup=False)

    def on_mount(self) -> None:
        """Seed both tables' columns once their DataTables have mounted."""
        self.query_one(f"#{_QUEUE_TABLE}", DataTable).add_columns(
            "run", "portfolio", "asof"
        )
        self.query_one(f"#{_CANDIDATES_TABLE}", DataTable).add_columns(
            *CANDIDATE_COLUMNS
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
        rows, detail = read
        self._paint_queue(rows)
        self._paint_candidates(detail)
        self._paint_gate(
            detail.text, unavailable=detail.unavailable, gate_id=detail.gate_id
        )

    def select_run(self, run_id: uuid.UUID) -> None:
        """Focus a queued run by id, then repaint the gate detail.

        A driver/test hook mirroring the DataTable row selection: the portfolio is
        taken from the last-painted queue so no extra read is needed. The table cursor
        moves onto the selected row, so the highlight and the selection never diverge.
        """
        summary = next((row for row in self._queue if row.run_id == run_id), None)
        if run_id != self.selected_run_id:
            self._clear_pick()
        self.selected_run_id = run_id
        self.selected_portfolio_id = summary.portfolio_id if summary else None
        self.refresh_view()
        table = self.query_one(f"#{_QUEUE_TABLE}", DataTable)
        with suppress(RowDoesNotExist):
            table.move_cursor(row=table.get_row_index(str(run_id)))

    def select_candidate(self, candidate_id: str | None) -> bool:
        """Select the candidate ``a`` approves on the gate on screen (``None`` clears).

        The hook behind ``↵`` on a candidate row, and a test driver. Only a feasible
        row of the set on screen can be selected; any other id leaves nothing
        selected and the line under the table says why. The selection is tied to the
        gate and the set it was made on (see :attr:`selected_candidate_id`).

        Args:
            candidate_id: The candidate to select, or ``None`` to clear.

        Returns:
            Whether a candidate is now selected.
        """
        candidate = find_candidate(self._candidates, candidate_id)
        if candidate_id is None or candidate is None or not is_selectable(candidate):
            self._clear_pick()
            refusal = (
                refused_pick_line(candidate_id, candidate)
                if candidate_id is not None
                else None
            )
            self._paint_pick(refusal)
            return False
        self.selected_candidate_id = candidate_id
        self._pick_anchor = (self.rendered_gate_id, self.rendered_set_hash)
        self._paint_pick()
        return True

    @property
    def at_order_gate(self) -> bool:
        """Whether the gate on screen is a ``place_orders`` (rebalance) gate."""
        return self.rendered_gate_name == ORDER_GATE

    def selected_candidate(self) -> Mapping[str, Any] | None:
        """Return the selected candidate's JSON dump while it is still feasible."""
        candidate = find_candidate(self._candidates, self.selected_candidate_id)
        return candidate if candidate is not None and is_selectable(candidate) else None

    def approve_refusal(self) -> str | None:
        """Return why ``a`` may not approve the rebalance gate on screen, or ``None``.

        Fails closed: a gate opened on another candidate set than the run's newest
        one is refused whatever is selected, as is a gate with no set or no selection.
        """
        if self._payload_set_hash != self.rendered_set_hash:
            return _STALE_SET
        if not self._candidates:
            return _NO_SET
        if self.selected_candidate() is None:
            return _NO_PICK
        return None

    def highlighted_run_id(self) -> uuid.UUID | None:
        """The queued run under the table cursor, or ``None`` for an empty queue."""
        table = self.query_one(f"#{_QUEUE_TABLE}", DataTable)
        if not table.row_count:
            return None
        key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return uuid.UUID(key) if key is not None else None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A queue row focuses its gate; a candidate row selects that candidate."""
        key = event.row_key.value
        if key is None:
            return
        if event.data_table.id == _CANDIDATES_TABLE:
            self.select_candidate(key)
        else:
            self.select_run(uuid.UUID(key))

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Show the candidate cursor only while the user is in that table."""
        if event.widget.id == _CANDIDATES_TABLE and isinstance(event.widget, DataTable):
            event.widget.show_cursor = True

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        """Hide the candidate cursor again once focus leaves the table."""
        if event.widget.id == _CANDIDATES_TABLE and isinstance(event.widget, DataTable):
            event.widget.show_cursor = False

    def _read(self) -> tuple[list[RunSummary], _GateDetail] | None:
        """Read the queue + the selected run's gate within one short session.

        Returns ``None`` on a queue read error — distinct from an empty queue, so the
        poll keeps the last-known queue and selection instead of mistaking a hiccup for
        a cleared inbox. The gate is read only when the selection is still in the live
        queue, and a gate that cannot be read degrades to a one-line notice (flagged
        ``unavailable``) without discarding the queue.
        """
        saver = self.shell._persistence.saver
        try:
            with self.read_session() as session:
                rows = observe.pending_hitl(session, saver, None)
                self._read_recovered("queue")
                selected = self.selected_run_id
                if selected is None or all(row.run_id != selected for row in rows):
                    return rows, _GateDetail()
                return rows, self._read_selected_gate(session, saver, selected)
        except Exception as exc:  # keep the poll loop alive on any read error
            self._read_failed("queue", exc)
            return None

    def _read_selected_gate(
        self, session: Any, saver: Any, run_id: uuid.UUID
    ) -> _GateDetail:
        """Read the selected gate, degrading a failure to a one-line notice.

        A driver error's text carries the statement and its bound parameters over
        several lines, so the user sees one ``Type: message`` line and the log
        gets the full error, once per distinct reason.
        """
        try:
            detail = _read_gate(session, saver, run_id)
        except Exception as exc:  # one bad gate must not hide the queue
            reason = self._read_failed("gate", exc)
            return _GateDetail(text=f"(gate unavailable: {reason})", unavailable=True)
        self._read_recovered("gate")
        return detail

    def _paint_read_error(self) -> None:
        """Flag a failed read next to the count, leaving queue + selection as-is."""
        self.query_one("#decisions-count", Label).update(
            f"{len(self._queue)} awaiting · ⚠ read failed, showing last known queue"
        )

    def _paint_queue(self, rows: list[RunSummary]) -> None:
        self._queue = rows
        table = self.query_one(f"#{_QUEUE_TABLE}", DataTable)
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
            self._clear_pick()

    def _paint_candidates(self, detail: _GateDetail) -> None:
        """Repaint the candidate table from ``detail`` and re-check the selection.

        A read that failed keeps the selection, since the gate and its set may not
        have changed, but shows no rows, so nothing can be selected or approved until
        the gate renders again.
        """
        if not detail.unavailable:
            if self._pick_anchor not in (None, (detail.gate_id, detail.set_hash)):
                self._clear_pick()
            self.rendered_gate_name = detail.gate_name
            self.rendered_set_hash = detail.set_hash
            self._payload_set_hash = detail.payload_set_hash
        self._candidates = detail.candidates
        table = self.query_one(f"#{_CANDIDATES_TABLE}", DataTable)
        with _keep_cursor(table, (self.selected_run_id, self.rendered_set_hash)):
            table.clear()
            for candidate in self._candidates:
                table.add_row(
                    *candidate_row(candidate), key=str(candidate["candidate_id"])
                )
        self._paint_pick("" if detail.unavailable else None)

    def _paint_pick(self, refusal: str | None = None) -> None:
        """Say under the table what ``a`` would approve, or why it would not."""
        if refusal is None and self.at_order_gate:
            blocked = (
                _STALE_SET if self._payload_set_hash != self.rendered_set_hash else None
            )
            refusal = pick_line(
                self.selected_candidate(),
                has_candidates=bool(self._candidates),
                blocked=blocked,
            )
        self.query_one("#decisions-pick", Static).update(refusal or "")

    def _clear_pick(self) -> None:
        self.selected_candidate_id = None
        self._pick_anchor = None

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


def _read_gate(session: Any, saver: Any, run_id: uuid.UUID) -> _GateDetail:
    """Read the selected run's gate: its id, its detail and its candidate set.

    The gate id is read from the checkpoint before the detail it labels: a run that
    moves between the two reads then leaves an older id than the text, and a decision
    on it is refused instead of landing on a gate nobody saw. The stored candidate
    set is read only for an order gate, the one gate that shows candidates. The run
    repository is imported lazily from its submodule, keeping ``fund.tui.views`` free
    of the ``fund.audit`` package at import time.
    """
    from fund.audit.repository import AgentRunRepository

    run = AgentRunRepository(session).get_run(run_id)
    if run is None:
        return _GateDetail()
    gate_id = observe.live_gate_id(saver, run)
    interrupt = observe.interrupt_for(saver, run)
    gate_name = decided_gate(gate_names(interrupt))
    candidate_set = (
        observe.candidate_set(session, run_id) if gate_name == ORDER_GATE else None
    )
    return _GateDetail(
        text=_format_decision_gate(interrupt, run),
        gate_id=gate_id,
        gate_name=gate_name,
        payload_set_hash=payload_set_hash(interrupt),
        set_hash=candidate_set.digest() if candidate_set is not None else None,
        candidates=tuple(
            candidate.model_dump(mode="json")
            for candidate in (candidate_set.candidates if candidate_set else ())
        ),
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

    run = AgentRunRepository(session).get_run(run_id)
    if run is None:
        return None
    return _format_decision_gate(observe.interrupt_for(saver, run), run)


def _pending_gate_names(session: Any, saver: Any, run_id: uuid.UUID) -> tuple[str, ...]:
    """The gated tool names ``run_id`` is paused on; empty when it has no gate."""
    from fund.audit.repository import AgentRunRepository

    run = AgentRunRepository(session).get_run(run_id)
    return gate_names(observe.interrupt_for(saver, run)) if run is not None else ()


# The middleware's default description ends with the call's raw args; on a candidate
# gate those are rendered below as candidates, so the "why" stops before them.
_DESCRIPTION_ARGS_MARKER = "\n\nTool: "


def _format_decision_gate(interrupt: dict[str, Any] | None, run: Any) -> str:
    """Render ``run``'s pending gate: header, the "why", and what it would place.

    A rebalance gate lists every candidate its payload carries, in id order (see
    :func:`_format_candidates`); which one is placed is the user's choice alone.
    A ``place_orders`` gate without a candidate set (one left paused from before
    candidates existed) is shown as one an approve refuses. No weight vector the PM
    typed into the tool call is ever shown, since ``place_orders`` ignores it.

    Args:
        interrupt: The run's pending HITL payload (``action_requests``), or ``None``.
        run: The ORM run row (``id`` / ``portfolio_id``) for the header.
    """
    pf = _short(run.portfolio_id) if run.portfolio_id else "fund-wide"
    requests = (interrupt or {}).get("action_requests") or []
    request = requests[0] if requests else {}
    name = request.get("name", "gate")
    args = request.get("args") or {}
    lines = [f"GATE · {name} · #{_short(run.id)} · {pf}"]
    why = request.get("description")
    if why and "candidates" in args:
        why = why.split(_DESCRIPTION_ARGS_MARKER, 1)[0]
    lines.append(f"why: {why}" if why else "why: review before it executes")
    lines.extend(_format_risk_gate(args.get("risk_gate")))
    if "candidates" in args:
        lines.extend(_format_candidates(args["candidates"]))
    elif name == ORDER_GATE or "weights" in args:
        lines.append(
            "orders: none - this gate has no candidate set, so an approve is "
            "refused; reject the run and start a fresh round"
        )
    return "\n".join(lines)


def _format_candidates(candidates: Any) -> list[str]:
    """Render the candidate set a rebalance gate carries: id order, never ranked.

    Each candidate is shown with its label, whether it is feasible and the weights
    approving it would place; which one is placed is the user's choice alone.
    """
    entries = [entry for entry in candidates or [] if isinstance(entry, dict)]
    if not entries:
        return [
            "candidates: none were computed, so an approve is refused; "
            "reject the run and start a fresh round"
        ]
    lines = ["candidates (listed by id; you pick one to approve):"]
    for entry in entries:
        status = "feasible" if entry.get("feasible") else "infeasible"
        weights = ", ".join(
            f"{ticker}: {float(weight):.4f}"
            for ticker, weight in sorted((entry.get("weights") or {}).items())
        )
        lines.append(
            f"  {entry.get('candidate_id')} · {entry.get('label')} · {status}"
            + (f" · {weights}" if weights else "")
        )
    return lines


_RISK_GATE_LABELS: dict[str, str] = {
    "passed": "passed",
    "failed": "FAILED",
    "stale": "STALE (the proposal changed after risk checked it)",
    "not_run": "NOT CHECKED",
}


def _format_risk_gate(gate: Any) -> list[str]:
    """Render the risk verdict pinned on a pre-candidate gate (empty when absent).

    Gates opened before candidates existed carried the risk officer's verdict on
    their one proposal. It is spelled out in words rather than left to a colour or
    glyph.
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
