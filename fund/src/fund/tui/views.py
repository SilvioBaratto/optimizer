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

import datetime as dt
import json
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast
from weakref import WeakKeyDictionary

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.widgets import (
    Checkbox,
    DataTable,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)
from textual.widgets.data_table import RowDoesNotExist

from fund import observe
from fund.observe import DailyBrief, RunSummary
from fund.tui import ask
from fund.tui.readouts import attribution_bar, render_readout
from fund.tui.stream import StepStatus, StreamEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType

    # Annotation-only: ``fund.performance`` pulls ``optimizer``/``skfolio``; with
    # ``from __future__ import annotations`` these are strings, so the Performance
    # view's reads route through ``fund.observe`` (lazy) and ``import fund.tui.*``
    # stays agent-stack-light (and optimizer/skfolio-free) at module scope.
    from fund.performance import HoldingMark, PerformanceSummary


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
        _streaming: Whether a live-streamed round is still pushing steps to the shell.
    """

    _session_factory: Callable[[], AbstractContextManager[Session]]
    _persistence: Any
    _portfolio_id: uuid.UUID | None
    _streaming: bool

    def focus_portfolio(self, portfolio_id: uuid.UUID) -> None:
        """Focus the shell on ``portfolio_id`` (set the header context from a view)."""
        ...

    def ask_submit(self, turn: str) -> None:
        """Handle a submitted Ask turn: resolve a pending action, else route it."""
        ...


# The shell views in sidebar order: (key, human title). The key is the id
# fragment for both the sidebar entry (``nav-<key>``) and the view (``view-<key>``).
VIEW_SPECS: tuple[tuple[str, str], ...] = (
    ("posture", "Posture"),
    ("deliberation", "Deliberation"),
    ("decisions", "Decisions"),
    ("mandate", "Mandate"),
    ("audit", "Audit"),
    ("performance", "Performance"),
    ("ask", "Ask"),
    ("brief", "Daily brief"),
)

# Two-letter nav labels for the collapsed rail, keyed by view key. Leading letters
# collide (Deliberation / Decisions / Daily brief), and the Decisions entry is the
# approve/kill gate, so every view needs its own.
RAIL_LABELS: dict[str, str] = {
    "posture": "Po",
    "deliberation": "Dl",
    "decisions": "Dc",
    "mandate": "Ma",
    "audit": "Au",
    "performance": "Pf",
    "ask": "As",
    "brief": "Br",
}


# The dataset each polled table last painted, so a repaint can tell a poll of the same
# data (hold the cursor) from a swap to different data (start at the top). Weak keys let
# a table's entry die with its view.
_PAINTED_SCOPE: WeakKeyDictionary[DataTable, Hashable] = WeakKeyDictionary()


@contextmanager
def _keep_cursor(table: DataTable, scope: Hashable = None) -> Generator[None]:
    """Hold ``table``'s cursor row and scroll offset across a clear-and-refill repaint.

    ``DataTable.clear()`` resets the cursor to row 0 and scrolls to the top, so a table
    repainted on every poll tick would yank the adviser's highlight and viewport back to
    the top and a following Enter would act on the wrong row. The cursor is re-seated on
    the same row key when that row survives the refill, else on the same index clamped
    to the new row count (rows added without a key get a fresh random key each paint and
    so take that fallback).

    Only a repaint of the *same* dataset is held. ``scope`` identifies the dataset the
    table shows (a filter set, the focused portfolio, the picked run); when it differs
    from the last paint the new rows are a different list, so they start at the top
    rather than inheriting a row index and viewport that belonged to the old one
    (positional keys and unkeyed rows would land the cursor on an arbitrary row).
    """
    held = _PAINTED_SCOPE.get(table, scope) == scope
    _PAINTED_SCOPE[table] = scope
    index = table.cursor_row
    scroll_y = table.scroll_y
    key = (
        table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        if table.row_count
        else None
    )
    yield
    if not held or not table.row_count:
        return
    if key is not None and key in table.rows:
        index = table.get_row_index(key)
    table.move_cursor(row=min(index, table.row_count - 1))
    # Queued after the cursor move's own scroll-into-view so the previous viewport
    # wins; the virtual height is only settled after the next refresh.
    table.call_after_refresh(table.scroll_to, y=scroll_y, animate=False, force=True)


class ShellView(Vertical):
    """A dumb placeholder view: a title line, a body line, and a refresh counter.

    Attributes:
        view_key: Stable id fragment, shared with the sidebar entry key.
        view_title: Human label shown at the top of the view.
        render_count: Incremented by :meth:`refresh_view` — a headless-test hook so a
            test can prove the active view was repainted without scraping the DOM.
    """

    # CSS selector of the control a sidebar pick lands on. Views are not focusable
    # themselves, so their view-scoped keys only fire once focus is inside them.
    primary_focus: ClassVar[str | None] = None

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.view_key = view_key
        self.view_title = view_title
        self.render_count = 0

    def focus_primary(self) -> None:
        """Move focus into the view so its view-scoped keys and arrows act on it.

        A focusable view takes focus itself; otherwise the widget named by
        :attr:`primary_focus` does. A no-op for a view that declares neither.
        """
        if self.can_focus:
            self.focus()
        elif self.primary_focus is not None:
            self.query_one(self.primary_focus).focus()

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

        The read discipline every data-view shares: open a session from the shell's
        injected factory, read the model, and roll back — the UI never holds a
        transaction across poll ticks. The caller does not commit.

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
        rows, gate, unavailable = read
        self._paint_queue(rows)
        self._paint_gate(gate, unavailable=unavailable)

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

    def _read(self) -> tuple[list[RunSummary], str | None, bool] | None:
        """Read the queue + the selected run's gate text within one short session.

        Returns ``None`` on a queue read error — distinct from an empty queue, so the
        poll keeps the last-known queue and selection instead of mistaking a hiccup for
        a cleared inbox. The gate text is read only when the selection is still in the
        live queue, and a gate that cannot be rendered degrades to a one-line notice
        (flagged by the third element) without discarding the queue.
        """
        saver = self.shell._persistence.saver
        try:
            with self.read_session() as session:
                rows = observe.pending_hitl(session, saver, None)
                gate: str | None = None
                unavailable = False
                if self.selected_run_id is not None and any(
                    row.run_id == self.selected_run_id for row in rows
                ):
                    try:
                        gate = _gate_text(session, saver, self.selected_run_id)
                    except Exception as exc:  # one bad gate must not hide the queue
                        gate = f"(gate unavailable: {exc})"
                        unavailable = True
                return rows, gate, unavailable
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

    def _paint_gate(self, gate: str | None, *, unavailable: bool = False) -> None:
        self.gate_text = gate or ""
        self.gate_unavailable = unavailable
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
            keys = ("type", "asset", "sector")
            detail = " ".join(str(violation[k]) for k in keys if k in violation)
            lines.append(f"  - {detail or 'violation'}")
    return lines


def _short(value: Any) -> str:
    """First hex group of a UUID (or its string) — a compact table cell."""
    return str(value).split("-", 1)[0]


# RiskToleranceBand values least→most risk-tolerant (mirrors fund.schemas.enums and
# fund.actions._BAND_RANK); kept local so the module scope stays Textual + observe.
_RISK_BANDS: tuple[str, ...] = (
    "defensive",
    "conservative",
    "balanced",
    "growth",
    "aggressive",
)


def _tick(flag: bool) -> str:
    """A compact on/off glyph for a boolean trigger."""
    return "✓" if flag else "✗"


def _format_caps(constraint_set: dict[str, Any]) -> str:
    """One-line summary of the numeric ConstraintSet caps, for display.

    Reads the raw ``constraint_set`` JSON (no schema rehydration) so a missing nested
    key degrades gracefully rather than raising inside the poll loop.
    """
    parts: list[str] = []
    nu3 = constraint_set.get("nu3")
    if isinstance(nu3, int | float):
        parts.append(f"drawdown ≤{nu3:.0%}")
    name_cap = (constraint_set.get("universe_filters") or {}).get("max_position_cap")
    if isinstance(name_cap, int | float):
        parts.append(f"name ≤{name_cap:.0%}")
    exclusions = (constraint_set.get("esg") or {}).get("exclusions") or []
    parts.append(f"ESG exclude {len(exclusions)}")
    return " · ".join(parts)


# Edit-guard key of each run-policy widget -> the ``PortfolioMandate`` field a save
# writes it to; the band has no entry because a mandate save never persists it.
_SAVED_FIELDS: tuple[tuple[str, str], ...] = (
    ("capital", "capital"),
    ("base_currency", "base_currency"),
    ("drift", "drift_l1_threshold"),
    ("benchmark", "benchmark"),
    ("gate", "hitl_gates"),
    ("directives", "remembered_directives"),
)


class MandateView(ShellView):
    """Per-portfolio policy editor: run-policy (persisted) + guardrails (audit-only).

    RUN POLICY mirrors the editable ``PortfolioMandate`` fields (capital, base currency,
    drift threshold, benchmark); saving upserts through the unified action path
    (``mandate_edit``). GUARDRAILS are read from the active MiFID profile — the
    suitability ``band`` is the ceiling, and proposing a *riskier* band logs a
    ``guardrail_override`` flagged ⚠ (audit-only this phase; enforcement is
    ``03-copilot``). The HITL gates are editable checkboxes (the ``place_orders`` PM
    gate toggles ``hitl_gates``; ``save_profile`` is the profiler's always-on gate, an
    indicator only); remembered directives render in a removable list (``x`` drops the
    highlighted one). Both fold back into the mandate on save.

    Reads through the context seam (``observe.get_mandate`` +
    ``MifidProfileRepository.get_active`` over a short read session); the writes go
    through ``fund.actions`` on a worker thread dispatched by the App.

    The poll repaints the editor every tick, but an editable widget the adviser has
    changed since its last paint keeps that unsaved value: the save and override
    actions read the live widgets, so a repaint would silently revert them to the
    persisted values. A field is released as soon as it matches the persisted value
    again, once :meth:`mark_saved` reports that a save landed it, or when the focused
    portfolio changes.

    Attributes:
        mandate: The focused portfolio's current run-policy, or ``None``.
        suitability_band: The active assessment's ``RiskToleranceBand`` value (the
            ceiling), or ``None`` when the portfolio has no profile yet.
    """

    # View-scoped so ``x`` removes a directive only while Mandate holds focus.
    BINDINGS: ClassVar[list[BindingType]] = [
        ("x", "remove_directive", "Remove directive")
    ]

    # The directives list, not the first input: an Input would swallow the advertised
    # ``x``/``o`` as typed characters, while the list leaves every hinted key live.
    primary_focus: ClassVar[str | None] = "#mandate-directives"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.mandate: Any = None
        self.suitability_band: str | None = None
        self._directives: list[str] = []
        # Value each editable widget was last painted with, to tell an adviser edit
        # from a stale repaint; reset whenever the focused portfolio changes.
        self._painted: dict[str, Any] = {}
        self._painted_pid: uuid.UUID | None = None

    def compose(self) -> ComposeResult:
        """Yield the title, the run-policy | guardrails split, and the key hint."""
        yield Label(self.view_title, classes="view-title")
        yield Label(
            "(no portfolio focused)", id="mandate-subtitle", classes="view-subtitle"
        )
        with Horizontal(classes="mandate-body"):
            with Vertical(classes="mandate-policy"):
                yield Label("RUN POLICY", classes="mandate-section")
                yield Input(placeholder="capital", id="mandate-capital")
                yield Input(placeholder="base currency", id="mandate-base-currency")
                yield Input(placeholder="drift L1 threshold (0-2)", id="mandate-drift")
                yield Input(placeholder="benchmark", id="mandate-benchmark")
                yield Static("", id="mandate-triggers", classes="mandate-readonly")
            with Vertical(classes="mandate-guardrails"):
                yield Label(
                    "GUARDRAILS (from MiFID profile)", classes="mandate-section"
                )
                yield Static(
                    "(no profile)", id="mandate-suitability", classes="mandate-readonly"
                )
                yield Select(
                    [(band, band) for band in _RISK_BANDS],
                    id="mandate-band",
                    allow_blank=True,
                )
                yield Static("", id="mandate-caps", classes="mandate-readonly")
                yield Label("HITL GATES", classes="mandate-section")
                yield Checkbox("Before execution", id="gate-place-orders")
                yield Checkbox(
                    "Save profile (always on)",
                    value=True,
                    disabled=True,
                    id="gate-save-profile",
                )
                yield Label("REMEMBERED DIRECTIVES", classes="mandate-section")
                yield ListView(id="mandate-directives", classes="mandate-directives")
        yield Static(
            "ctrl+s save · o override guardrail · x remove directive",
            id="mandate-hint",
            classes="mandate-hint",
        )

    def refresh_view(self) -> None:
        """Repaint the editor from the focused portfolio's mandate + active profile."""
        self.render_count += 1
        pid = self.shell._portfolio_id
        if pid != self._painted_pid:
            self._painted.clear()
            self._painted_pid = pid
        subtitle = self.query_one("#mandate-subtitle", Label)
        if pid is None:
            subtitle.update("(no portfolio focused — pick one in the header)")
            self.mandate = None
            self.suitability_band = None
            self._paint_policy(None)
            self._paint_guardrails(None, "")
            self._paint_gates(None)
            self._paint_directives(None)
            return
        subtitle.update(f"portfolio {_short(pid)}")
        mandate, band, caps = self._read(pid)
        self.mandate = mandate
        self.suitability_band = band
        self._paint_policy(mandate)
        self._paint_guardrails(band, caps)
        self._paint_gates(mandate)
        self._paint_directives(mandate)

    def proposed_mandate(self) -> Any:
        """Build a ``PortfolioMandate`` from the edited run-policy inputs, re-validated.

        Returns ``None`` when there is no current mandate to edit. Raises
        ``pydantic.ValidationError`` on an out-of-contract edit (e.g. a non-positive
        capital) — the App surfaces that as a status line instead of saving.
        """
        if self.mandate is None:
            return None
        from fund.schemas.mandate import PortfolioMandate

        data = {
            **self.mandate.model_dump(mode="json"),
            "capital": self.query_one("#mandate-capital", Input).value,
            "base_currency": self.query_one("#mandate-base-currency", Input).value,
            "drift_l1_threshold": self.query_one("#mandate-drift", Input).value,
            "benchmark": self.query_one("#mandate-benchmark", Input).value or None,
            "hitl_gates": self._proposed_gates(),
            "remembered_directives": list(self._directives),
        }
        return PortfolioMandate.model_validate(data)

    def _proposed_gates(self) -> list[str]:
        """The PM gates checked in the editor (``save_profile`` is not a gate here)."""
        checked = self.query_one("#gate-place-orders", Checkbox).value
        return ["place_orders"] if checked else []

    def action_remove_directive(self) -> None:
        """Drop the highlighted remembered directive (persisted on the next save)."""
        listview = self.query_one("#mandate-directives", ListView)
        index = listview.index
        if index is None or index >= len(self._directives):
            return
        del self._directives[index]
        self._refill_directives()

    def proposed_band(self) -> str | None:
        """The target band selected in the guardrail editor, or ``None`` when blank."""
        value = self.query_one("#mandate-band", Select).value
        return None if value is Select.NULL else str(value)

    def _read(self, pid: uuid.UUID) -> tuple[Any, str | None, str]:
        """Read the mandate + active guardrails in one short session.

        Returns ``(None, None, "")`` on any read error so the poll loop stays alive.
        """
        try:
            with self.read_session() as session:
                mandate = observe.get_mandate(session, pid)
                band, caps = self._read_guardrails(session, pid)
                return mandate, band, caps
        except Exception:  # keep the poll loop alive on any read error
            return None, None, ""

    def _read_guardrails(
        self, session: Session, pid: uuid.UUID
    ) -> tuple[str | None, str]:
        """The active suitability band + a caps summary, read lazily off the profile.

        Imports the repo from its submodule (never the ``fund.audit`` root, which pulls
        the langgraph persistence bootstrap).
        """
        from fund.audit.mifid_repository import MifidProfileRepository

        profile = MifidProfileRepository(session).get_active(pid)
        if profile is None:
            return None, ""
        band = (profile.suitability or {}).get("band")
        return band, _format_caps(profile.constraint_set or {})

    def _take_persisted(self, key: str, current: Any, persisted: Any) -> bool:
        """Whether ``key``'s widget may be repainted with the ``persisted`` value.

        ``False`` only when the adviser changed the widget since it was last painted
        and it still differs from ``persisted`` — that unsaved edit must survive the
        poll. Otherwise the field is clean (or just saved), so it is released and its
        painted baseline advances to ``persisted``.
        """
        edited = key in self._painted and current != self._painted[key]
        if edited and current != persisted:
            return False
        self._painted[key] = persisted
        return True

    def mark_saved(self, saved: Any) -> None:
        """Release the run-policy widgets that still hold what ``saved`` persisted.

        A save stores the typed value but the next read can render it differently
        (``0.20`` comes back as ``0.2``), so comparing text with the persisted text
        would keep the field "unsaved" forever: an external change could never reach
        it and a later save would overwrite that change. Equality is therefore judged
        on the validated model, once, at save time — not per poll, where treating
        ``0.20`` as equal to ``0.2`` would repaint the field mid-keystroke. A widget
        the adviser edited again after pressing save keeps its protection, and the
        pending band pick is untouched (the band is not part of the mandate).

        Args:
            saved: The ``PortfolioMandate`` whose upsert just committed.
        """
        from pydantic import ValidationError

        try:
            current = self.proposed_mandate()
        except ValidationError:
            return
        if current is None:
            return
        for key, attribute in _SAVED_FIELDS:
            if getattr(current, attribute) == getattr(saved, attribute):
                self._painted.pop(key, None)

    def _paint_policy(self, mandate: Any) -> None:
        capital = self.query_one("#mandate-capital", Input)
        base_currency = self.query_one("#mandate-base-currency", Input)
        drift = self.query_one("#mandate-drift", Input)
        benchmark = self.query_one("#mandate-benchmark", Input)
        triggers = self.query_one("#mandate-triggers", Static)
        persisted = (
            ("capital", capital, ""),
            ("base_currency", base_currency, ""),
            ("drift", drift, ""),
            ("benchmark", benchmark, ""),
        )
        if mandate is not None:
            persisted = (
                ("capital", capital, str(mandate.capital)),
                ("base_currency", base_currency, mandate.base_currency),
                ("drift", drift, str(mandate.drift_l1_threshold)),
                ("benchmark", benchmark, mandate.benchmark or ""),
            )
        for key, field, text in persisted:
            if self._take_persisted(key, field.value, text):
                field.value = text
        if mandate is None:
            triggers.update("(no mandate — profile the portfolio first)")
            return
        triggers.update(
            f"triggers cron {_tick(mandate.triggers.cron)} "
            f"drift {_tick(mandate.triggers.drift)} · "
            f"gates {', '.join(mandate.hitl_gates)}"
        )

    def _paint_guardrails(self, band: str | None, caps: str) -> None:
        suitability = self.query_one("#mandate-suitability", Static)
        select = self.query_one("#mandate-band", Select)
        caps_widget = self.query_one("#mandate-caps", Static)
        if band is None:
            suitability.update("(no profile — p to profile)")
            caps_widget.update("")
            return
        suitability.update(f"suitability band {band} (ceiling)")
        if band in _RISK_BANDS and self._take_persisted(
            "band", self.proposed_band(), band
        ):
            select.value = band
        caps_widget.update(caps)

    def _paint_gates(self, mandate: Any) -> None:
        """Reflect the mandate's PM gates into the toggleable ``place_orders`` box."""
        checkbox = self.query_one("#gate-place-orders", Checkbox)
        persisted = mandate is not None and "place_orders" in mandate.hitl_gates
        if self._take_persisted("gate", checkbox.value, persisted):
            checkbox.value = persisted

    def _paint_directives(self, mandate: Any) -> None:
        """Repopulate the removable directives list from the mandate.

        The rows are rebuilt only when the list actually changes: a rebuild resets the
        ListView highlight, which ``x`` relies on to pick the directive to drop.
        """
        persisted = list(mandate.remembered_directives) if mandate else []
        if not self._take_persisted("directives", self._directives, persisted):
            return
        if self._directives != persisted:
            self._directives = list(persisted)
            self._refill_directives()

    def _refill_directives(self) -> None:
        """Rebuild the directives ``ListView`` rows from :attr:`_directives`."""
        listview = self.query_one("#mandate-directives", ListView)
        listview.clear()
        for directive in self._directives:
            listview.append(ListItem(Label(directive, markup=False)))


# Latest-run status buckets the roster glyph + the fund strip aggregate by.
_RUN_RUNNING: frozenset[str] = frozenset({"pending", "running"})
_RUN_FAILED: frozenset[str] = frozenset({"incomplete", "blocked", "failed"})
# Both live HITL gates read as awaiting: a plain pause and a barge-in that re-gated.
_RUN_AWAITING: frozenset[str] = frozenset({"paused", "responded"})

# Health tiers as (dot, label): the word disambiguates the two tiers sharing ● so a
# roster row still reads apart (the spec's "dot plus label").
_HEALTH_OK: tuple[str, str] = ("●", "ok")
_HEALTH_WARN: tuple[str, str] = ("◐", "warn")
_HEALTH_BREACH: tuple[str, str] = ("●", "breach")


def _run_glyph(status: str | None) -> str:
    """Roster run glyph: ▶ running · ⏸ awaiting · ✓ idle · — no runs."""
    if status is None:
        return "—"
    if status in _RUN_AWAITING:
        return "⏸"
    if status in _RUN_RUNNING:
        return "▶"
    return "✓"


def _posture_health(
    drift_l1: float, threshold: float | None, run_status: str | None
) -> tuple[str, str]:
    """A roster row's ``(dot, label)`` health from drift vs mandate + the run status.

    A failed/blocked latest run is a ``breach`` (the risk gate stopped it before a
    ticket); otherwise L1 drift at or above the mandate's threshold is a ``warn``;
    everything else is ``ok``. A portfolio with no mandate (``threshold is None``) can
    only be ``ok`` or ``breach`` — there is no threshold to breach.

    Args:
        drift_l1: The portfolio's current-vs-target L1 drift.
        threshold: The mandate's ``drift_l1_threshold``, or ``None`` when unmandated.
        run_status: The latest run's status, or ``None`` when the portfolio has none.

    Returns:
        The ``(dot, label)`` pair for the roster health cell.
    """
    if run_status in _RUN_FAILED:
        return _HEALTH_BREACH
    if threshold is not None and drift_l1 >= threshold:
        return _HEALTH_WARN
    return _HEALTH_OK


@dataclass(frozen=True)
class _RosterRow:
    """One Posture roster line: a portfolio, its latest-run glyph, and health."""

    portfolio_id: uuid.UUID
    run_status: str | None
    run_glyph: str
    dot: str
    label: str


class PostureView(ShellView):
    """The fund control tower: a portfolio roster, a fund strip, and focused detail.

    Each tick reads every portfolio through ``observe.list_portfolios`` + per-portfolio
    ``list_portfolio_runs`` / ``portfolio_state`` / ``get_mandate`` on short read
    sessions: the roster shows each portfolio's latest-run glyph and a health dot+label
    (``observe.drift_l1`` vs the mandate threshold, or a failed run = breach); the strip
    aggregates ``#pf / #awaiting / #running / #warning``. Selecting a roster row focuses
    that portfolio through the shell seam (``focus_portfolio`` — header context) and
    reveals its detail; an un-run portfolio shows the
    ``no runs yet · p profile to start`` empty-state. A ``Daily brief → d`` tile jumps
    to the per-day recap for the focused portfolio.

    Attributes:
        detail_text: Mirror of the rendered detail pane — a headless-test hook.
    """

    # View-scoped so ``d`` opens the brief only from Posture — never clobbering the
    # global ``b`` (barge-in); the App action owns the programmatic view switch.
    BINDINGS: ClassVar[list[BindingType]] = [
        ("d", "app.open_brief", "Daily brief"),
    ]

    primary_focus: ClassVar[str | None] = "#posture-roster"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.detail_text: str = ""
        self._roster: list[_RosterRow] = []

    def compose(self) -> ComposeResult:
        """Yield the title, the fund strip, the roster | detail split, and the hint."""
        yield Label(self.view_title, classes="view-title")
        yield Static("0 pf", id="posture-strip", classes="view-subtitle")
        with Horizontal(classes="posture-body"):
            yield DataTable(
                id="posture-roster", cursor_type="row", classes="posture-roster"
            )
            yield Static(
                "(no portfolio focused)", id="posture-detail", classes="posture-detail"
            )
        yield Static(
            "↑↓ select · ↵ focus · d brief",
            id="posture-hint",
            classes="posture-hint",
        )

    def on_mount(self) -> None:
        """Seed the roster table's columns once its DataTable has mounted."""
        self.query_one("#posture-roster", DataTable).add_columns(
            "portfolio", "run", "health"
        )

    def refresh_view(self) -> None:
        """Repaint the roster, the fund strip, and the focused-portfolio detail."""
        self.render_count += 1
        roster, strip = self._read_roster()
        self._paint_roster(roster, strip)
        self._paint_detail()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A roster-row selection focuses that portfolio through the shell seam."""
        key = event.row_key.value
        if key is not None:
            self.shell.focus_portfolio(uuid.UUID(key))

    def _read_roster(self) -> tuple[list[_RosterRow], str]:
        """Read every portfolio's roster row + the fund-strip summary in one session.

        Returns ``([], "0 pf")`` on any read error so the poll loop stays alive. The
        strip mirrors the fund autonomy posture (ARMED/FROZEN) so it surfaces here too,
        not only in the header.
        """
        try:
            with self.read_session() as session:
                rows = [
                    self._roster_row(session, pid)
                    for pid in observe.list_portfolios(session)
                ]
                posture = self._read_autonomy(session)
        except Exception:  # keep the poll loop alive on any read error
            return [], "0 pf"
        return rows, f"{_fund_strip(rows)} · {posture}"

    def _read_autonomy(self, session: Session) -> str:
        """The fund autonomy posture glyph, read off the agent-stack-free repo."""
        from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState

        # Read-only (``fund_posture`` never inserts) so a poll tick never mutates.
        posture = AutonomyRepository(session).fund_posture()
        return "❄ FROZEN" if posture == AutonomyState.FROZEN.value else "● ARMED"

    def _roster_row(self, session: Session, pid: uuid.UUID) -> _RosterRow:
        runs = observe.list_portfolio_runs(session, pid, limit=1)
        status = runs[0].status if runs else None
        mandate = observe.get_mandate(session, pid)
        threshold = mandate.drift_l1_threshold if mandate is not None else None
        drift = observe.portfolio_state(session, pid).drift_l1
        dot, label = _posture_health(drift, threshold, status)
        return _RosterRow(pid, status, _run_glyph(status), dot, label)

    def _paint_roster(self, rows: list[_RosterRow], strip: str) -> None:
        self._roster = rows
        table = self.query_one("#posture-roster", DataTable)
        with _keep_cursor(table):
            table.clear()
            for row in rows:
                table.add_row(
                    _short(row.portfolio_id),
                    row.run_glyph,
                    f"{row.dot} {row.label}",
                    key=str(row.portfolio_id),
                )
        self.query_one("#posture-strip", Static).update(strip)

    def _paint_detail(self) -> None:
        pid = self.shell._portfolio_id
        text = (
            self._detail_text(pid)
            if pid is not None
            else "(no portfolio focused — ↵ a roster row)"
        )
        self.detail_text = text
        self.query_one("#posture-detail", Static).update(text)

    def _detail_text(self, pid: uuid.UUID) -> str:
        """Render the focused portfolio's detail in a short session (``""`` on fail)."""
        try:
            with self.read_session() as session:
                return _render_posture_detail(session, pid)
        except Exception:  # keep the poll loop alive on any read error
            return ""


def _fund_strip(rows: list[_RosterRow]) -> str:
    """The fund-aggregate strip: ``#pf / #awaiting / #running / #warning``."""
    n_awaiting = sum(1 for row in rows if row.run_status in _RUN_AWAITING)
    n_running = sum(1 for row in rows if row.run_status in _RUN_RUNNING)
    n_warning = sum(1 for row in rows if (row.dot, row.label) == _HEALTH_WARN)
    return (
        f"{len(rows)} pf · ⏸{n_awaiting} awaiting · "
        f"▶{n_running} running · ◐{n_warning} warning"
    )


def _render_posture_detail(session: Session, pid: uuid.UUID) -> str:
    """The focused-portfolio detail: state, NAV tile, mandate, active run, events.

    An un-run portfolio renders the ``no runs yet · p profile to start`` empty-state
    (the launchpad to onboarding). The NAV tile shows the real NAV-series period return
    from ``observe.performance_summary``; a run-but-unmarked portfolio degrades to a
    ``no marks yet`` line rather than a fabricated figure.
    """
    runs = observe.list_portfolio_runs(session, pid, limit=1)
    if not runs:
        return f"portfolio {_short(pid)}\nno runs yet · p profile to start"
    latest = runs[0]
    state = observe.portfolio_state(session, pid)
    lines = [
        f"portfolio {_short(pid)} · {_run_glyph(latest.status)} {latest.status}",
        f"drift L1 {state.drift_l1:.4f} · "
        f"{len(state.current)} held → {len(state.target)} target",
        _posture_nav_line(session, pid),
    ]
    mandate = observe.get_mandate(session, pid)
    if mandate is not None:
        lines.append(
            f"mandate {mandate.base_currency} {mandate.capital} · "
            f"drift ≥{mandate.drift_l1_threshold}"
        )
    lines.append(f"ACTIVE RUN #{_short(latest.run_id)} · {latest.status}")
    events = observe.recent_events(session, pid)
    if events.portfolio_events:
        recent = " · ".join(event["as_of"] for event in events.portfolio_events[:3])
        lines.append(f"RECENT {recent}")
    lines.append("Daily brief → d open")
    return "\n".join(lines)


def _posture_nav_line(session: Session, pid: uuid.UUID) -> str:
    """The Posture NAV tile: the NAV-series period return with a dot + label direction.

    Reads ``observe.performance_summary`` over the persisted ``nav_history`` series. A
    run-but-unmarked portfolio (the daily NAV-mark step has not run for it yet) yields
    an all-``None`` summary, which degrades to an explicit ``NAV  no marks yet`` line
    rather than a fabricated figure.
    """
    summary = observe.performance_summary(session, pid)
    period_return = summary.returns.get("max")
    if period_return is None:
        return "NAV  no marks yet"
    dot, label = _perf_direction(period_return)
    horizons = " · ".join(
        f"{h} {_pct(summary.returns.get(h))}" for h in ("1D", "YTD", "max")
    )
    return f"NAV  {dot} {label}  {horizons}"


class AuditView(ShellView):
    """The fund-wide, filterable decision timeline: agents ∪ humans.

    Reads ``observe.audit_timeline`` each tick into a table (when · pf · actor ·
    decision), with composable filters the shell toggles — actor side
    (:attr:`filter_kind`), ⚑ overrides (:attr:`filter_overrides`), and the
    header-focused portfolio (:attr:`filter_portfolio`). Selecting a row reveals the
    full provenance (prev value, source, idempotency key — immutable); ⚑ override / ⚠
    suitability rows are flagged in the decision cell. :meth:`export_csv` writes the
    current (filtered) timeline to a file.

    Attributes:
        filter_kind: ``None`` (both) / ``"agent"`` / ``"human"`` — the actor filter.
        filter_overrides: Keep only ⚑ override rows when ``True``.
        filter_portfolio: Scope to the header-focused portfolio when ``True``.
        detail_text: Mirror of the rendered detail pane — a headless-test hook.
    """

    primary_focus: ClassVar[str | None] = "#audit-table"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.filter_kind: str | None = None
        self.filter_overrides = False
        self.filter_portfolio = False
        self.detail_text = ""
        self.selected_index: int | None = None
        self._entries: list[Any] = []

    def compose(self) -> ComposeResult:
        """Yield the title, the filter line, the table | detail split, and the hint."""
        yield Label(self.view_title, classes="view-title")
        yield Static("", id="audit-filters", classes="view-subtitle")
        with Horizontal(classes="audit-body"):
            yield DataTable(id="audit-table", cursor_type="row", classes="audit-table")
            # Literal text: the record carries human free text (a rationale) whose
            # brackets must neither vanish as style tags nor raise MarkupError.
            yield Static(
                "(select a decision)",
                id="audit-detail",
                classes="audit-detail",
                markup=False,
            )
        yield Static(
            "↑↓ select · t actor · f ⚑ overrides · g this-pf · e export",
            id="audit-hint",
            classes="audit-hint",
        )

    def on_mount(self) -> None:
        """Seed the timeline table's columns once its DataTable has mounted."""
        self.query_one("#audit-table", DataTable).add_columns(
            "when", "pf", "actor", "decision"
        )

    def refresh_view(self) -> None:
        """Repaint the filtered timeline, the filter line, and the row detail."""
        self.render_count += 1
        self._entries = self._read()
        self._paint_table()
        self._paint_filters()
        self._paint_detail()

    def cycle_kind(self) -> None:
        """Advance the actor filter: all → agent → human → all, then repaint."""
        order: tuple[str | None, ...] = (None, "agent", "human")
        self.filter_kind = order[(order.index(self.filter_kind) + 1) % len(order)]
        self.selected_index = None
        self.refresh_view()

    def toggle_overrides(self) -> None:
        """Flip the ⚑ overrides-only filter, then repaint."""
        self.filter_overrides = not self.filter_overrides
        self.selected_index = None
        self.refresh_view()

    def toggle_portfolio_scope(self) -> None:
        """Flip the header-focused-portfolio scope, then repaint."""
        self.filter_portfolio = not self.filter_portfolio
        self.selected_index = None
        self.refresh_view()

    def select_index(self, index: int) -> None:
        """Focus a timeline row by position, then repaint the detail pane."""
        self.selected_index = index
        self.refresh_view()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A row selection reveals that decision's full provenance."""
        self.select_index(event.cursor_row)

    def export_csv(self, path: str) -> str:
        """Write the current (filtered) timeline to ``path`` as CSV; return ``path``.

        One row per entry with the full provenance columns, so the exported file is a
        faithful, portable snapshot of what the view shows. JSON-valued cells are
        serialised deterministically.
        """
        import csv
        from pathlib import Path

        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(_AUDIT_CSV_HEADER)
            for entry in self._entries:
                writer.writerow(_audit_csv_row(entry))
        return path

    def _read(self) -> list[Any]:
        """Read the filtered timeline in one short session (``[]`` on any error)."""
        try:
            with self.read_session() as session:
                pid = self.shell._portfolio_id if self.filter_portfolio else None
                return observe.audit_timeline(
                    session,
                    portfolio_id=pid,
                    kind=self.filter_kind,
                    overrides_only=self.filter_overrides,
                )
        except Exception:  # keep the poll loop alive on any read error
            return []

    def _paint_table(self) -> None:
        table = self.query_one("#audit-table", DataTable)
        pid = self.shell._portfolio_id if self.filter_portfolio else None
        scope = (self.filter_kind, self.filter_overrides, pid)
        with _keep_cursor(table, scope):
            table.clear()
            for entry in self._entries:
                table.add_row(
                    entry.when.strftime("%m-%d %H:%M"),
                    _short(entry.portfolio_id) if entry.portfolio_id else "fund",
                    entry.actor,
                    _audit_label(entry),
                )
        if self.selected_index is not None and self.selected_index >= len(
            self._entries
        ):
            self.selected_index = None

    def _paint_filters(self) -> None:
        parts = [f"actor {self.filter_kind or 'all'}"]
        if self.filter_overrides:
            parts.append("⚑ overrides")
        if self.filter_portfolio:
            parts.append("this pf")
        parts.append(f"{len(self._entries)} rows")
        self.query_one("#audit-filters", Static).update(" · ".join(parts))

    def _paint_detail(self) -> None:
        if self.selected_index is not None and self.selected_index < len(self._entries):
            text = _audit_detail(self._entries[self.selected_index])
        else:
            text = "(select a decision)"
        self.detail_text = text
        self.query_one("#audit-detail", Static).update(text)


_AUDIT_CSV_HEADER: tuple[str, ...] = (
    "when",
    "kind",
    "actor",
    "action",
    "portfolio_id",
    "run_id",
    "prev_value",
    "new_value",
    "source",
    "idempotency_key",
    "override_flag",
    "suitability_flag",
    "target",
    "rationale",
)


def _audit_label(entry: Any) -> str:
    """A timeline decision cell: the action plus any ⚑/⚠ flags."""
    flags = ""
    if entry.override_flag:
        flags += " ⚑"
    if entry.suitability_flag:
        flags += " ⚠"
    return f"{entry.action}{flags}"


def _audit_detail(entry: Any) -> str:
    """The provenance detail for one timeline row (immutable audit record)."""
    lines = [f"{entry.kind} · {entry.actor} · {entry.action}"]
    if entry.portfolio_id:
        lines.append(f"portfolio {_short(entry.portfolio_id)}")
    if entry.run_id:
        lines.append(f"run #{_short(entry.run_id)}")
    if entry.target:
        lines.append(f"target: {entry.target}")
    if entry.prev_value is not None:
        lines.append(f"prev: {entry.prev_value}")
    if entry.new_value is not None:
        lines.append(f"new: {entry.new_value}")
    if entry.source:
        lines.append(f"source: {entry.source}")
    if entry.idempotency_key:
        lines.append(f"idemp: {entry.idempotency_key}")
    if entry.rationale:
        lines.append(f"rationale: {entry.rationale}")
    if entry.override_flag:
        lines.append("⚑ override")
    if entry.suitability_flag:
        lines.append("⚠ outside suitability band")
    lines.append("(immutable audit record)")
    return "\n".join(lines)


# Spreadsheet formula-injection guard: Excel/Calc evaluate a cell whose first character
# is one of these. Audit cells carry agent-generated text (a decision string, a source),
# so a leading trigger is neutralised with a prefixed apostrophe before export.
_CSV_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(text: str) -> str:
    """Neutralise a leading spreadsheet formula trigger in a CSV cell."""
    if text and text[0] in _CSV_FORMULA_TRIGGERS:
        return "'" + text
    return text


def _audit_csv_row(entry: Any) -> list[str]:
    return [
        entry.when.isoformat(),
        _csv_safe(entry.kind),
        _csv_safe(entry.actor),
        _csv_safe(entry.action),
        str(entry.portfolio_id or ""),
        str(entry.run_id or ""),
        _audit_cell(entry.prev_value),
        _audit_cell(entry.new_value),
        _csv_safe(entry.source or ""),
        _csv_safe(entry.idempotency_key or ""),
        str(entry.override_flag),
        str(entry.suitability_flag),
        _csv_safe(entry.target or ""),
        _csv_safe(entry.rationale or ""),
    ]


def _audit_cell(value: Any) -> str:
    """Render a (possibly JSON) audit value as a safe, deterministic CSV cell."""
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return _csv_safe(str(value))
    return _csv_safe(json.dumps(value, sort_keys=True))


class DeliberationView(ShellView):
    """Replay a run's deliberation: a run picker, a step timeline, and step detail.

    Per-portfolio (header focus): lists the focused portfolio's runs, and for a picked
    run renders ``observe.load_run_transcript`` — the checkpoint's PM/subagent turns
    interleaved with the ``agent_decisions`` — as an ordered step timeline (left);
    selecting a step reveals its detail (right, with the decision payload where one
    exists).

    While a round streams it switches to a **live tail**: :meth:`start_live` swaps the
    replay split for a single timeline that :meth:`ingest_event` fills in emission
    order, flipping each prior step's ●/✓ marker as the next event lands; auto-follow
    glues the view to the newest row until the adviser moves the cursor to inspect
    (``f`` / :meth:`action_follow_live` re-snaps). :meth:`end_live` drops back to the
    replay transcript on the terminal event.

    Attributes:
        selected_run_id: The run whose transcript is shown, or ``None``.
        selected_step: The focused step's position, or ``None``.
        detail_text: Mirror of the rendered step detail — a headless-test hook.
        live_run_id: The streaming run while live, else ``None`` (replay mode).
        live_portfolio_id: The live run's portfolio while streaming, else ``None``.
        following: Whether the live tail auto-follows; a cursor move pauses it.
        live_text: Mirror of the rendered live-step detail — a headless-test hook.
    """

    # View-scoped so ``f`` means follow-live only while Deliberation holds focus; the
    # App's global ``f`` (Audit overrides filter) is untouched (R2). ``x`` routes to the
    # App's recompute action (the worker lives there, mirroring the stream worker).
    BINDINGS: ClassVar[list[BindingType]] = [
        ("f", "follow_live", "Follow live"),
        ("x", "app.expand_readout", "Expand read-out"),
    ]

    primary_focus: ClassVar[str | None] = "#delib-runs"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.selected_run_id: uuid.UUID | None = None
        self.selected_step: int | None = None
        self.detail_text = ""
        self._entries: list[Any] = []
        self.live_run_id: uuid.UUID | None = None
        # The portfolio the live run belongs to, pinned at start_live so a barge-in
        # addresses the streaming run's portfolio even if the header focus moves.
        self.live_portfolio_id: uuid.UUID | None = None
        self.following = True
        self.live_text = ""
        self._live_events: list[StreamEvent] = []
        # The worker's recompute for the expanded step: (run_id, agent, key, result),
        # where key is the allocator decision_index (None for risk). Held so it survives
        # poll ticks until the adviser selects a different run/step.
        self._readout: tuple[uuid.UUID, str, int | None, Any] | None = None

    def compose(self) -> ComposeResult:
        """Yield the title, subtitle, the runs/steps/live split, detail, and hint."""
        yield Label(self.view_title, classes="view-title")
        yield Label(
            "(no portfolio focused)", id="delib-subtitle", classes="view-subtitle"
        )
        with Horizontal(classes="delib-body"):
            with Vertical(classes="delib-left"):
                yield DataTable(
                    id="delib-runs", cursor_type="row", classes="delib-runs"
                )
                yield DataTable(
                    id="delib-steps", cursor_type="row", classes="delib-steps"
                )
                yield DataTable(
                    id="delib-live", cursor_type="row", classes="delib-live"
                )
            # Agent prose routinely carries "[k=v, k=v]" runs, which Textual would parse
            # as markup and raise on; every free-text pane below renders literally.
            yield Static(
                "(select a step)",
                id="delib-detail",
                classes="delib-detail",
                markup=False,
            )
        yield Static(
            "↑↓ select · ↵ run then step · f follow live",
            id="delib-hint",
            classes="delib-hint",
        )

    def on_mount(self) -> None:
        """Seed the run/step/live tables' columns; hide the live tail until a stream."""
        self.query_one("#delib-runs", DataTable).add_columns("run", "asof", "status")
        self.query_one("#delib-steps", DataTable).add_columns("idx", "agent", "step")
        live = self.query_one("#delib-live", DataTable)
        live.add_columns("", "agent", "step")
        live.display = False

    def refresh_view(self) -> None:
        """Repaint the run picker and step detail, or the live subtitle while streaming.

        In live mode the timeline is push-driven by :meth:`ingest_event`, so a poll
        tick only refreshes the follow/inspect subtitle and must not overwrite the
        streamed rows with the replay read. A finished stream's tail that no longer
        has a gate to act on is retired first (:meth:`retire_spent_live_tail`).
        """
        self.render_count += 1
        self.retire_spent_live_tail()
        if self.live_run_id is not None:
            self._paint_live_subtitle()
            return
        runs, entries = self._read()
        self._paint_subtitle()
        self._paint_runs(runs)
        self._paint_steps(entries)
        self._paint_detail()

    # --- live tail ----------------------------------------------------------

    def start_live(self, run_id: uuid.UUID, portfolio_id: uuid.UUID) -> None:
        """Enter live mode for ``run_id``: swap the replay split for an empty timeline.

        Clears any prior stream, re-arms auto-follow, and hides the run/step picker so
        the streamed steps fill the pane. Idempotent per run — a fresh call restarts
        the timeline.

        Args:
            run_id: The streaming run whose steps fill the live tail.
            portfolio_id: The run's portfolio, pinned so a barge-in targets it rather
                than whatever the header happens to be focused on at keypress time.
        """
        self.live_run_id = run_id
        self.live_portfolio_id = portfolio_id
        self._live_events = []
        self.following = True
        self.live_text = ""
        self.query_one("#delib-runs", DataTable).display = False
        self.query_one("#delib-steps", DataTable).display = False
        live = self.query_one("#delib-live", DataTable)
        live.clear()
        live.display = True
        self._paint_live_subtitle()
        self.query_one("#delib-detail", Static).update("(waiting for the first step…)")

    def ingest_event(self, event: StreamEvent) -> None:
        """Append one streamed step, flipping the prior LIVE marker to DONE.

        A no-op unless a stream is live, and for an event of any run other than the
        live one (a superseded stream still draining its thread must not interleave
        its rows, whose indexes collide, into the current timeline). The prior row's ●
        becomes ✓ before the new row lands, so markers need no second producer pass;
        the viewport snaps to the tail while :attr:`following`, and the detail pane
        mirrors the newest step.
        """
        if self.live_run_id is None or event.run_id != self.live_run_id:
            return
        table = self.query_one("#delib-live", DataTable)
        if self._live_events and self._live_events[-1].status is StepStatus.LIVE:
            table.update_cell_at(
                Coordinate(len(self._live_events) - 1, 0), StepStatus.DONE.glyph
            )
        self._live_events.append(event)
        table.add_row(
            event.status.glyph,
            event.agent,
            event.step or event.kind,
            key=str(event.index),
        )
        if self.following:
            table.scroll_end(animate=False)
        self._paint_live_detail(event)

    def end_live(self) -> None:
        """Leave live mode and fall back to the replay transcript (terminal event)."""
        self._leave_live()
        self.refresh_view()

    def retire_spent_live_tail(self) -> bool:
        """Leave live mode once a finished stream's tail has nothing left to act on.

        A round that ends at its approval gate keeps its timeline standing, and only
        the stream worker ends live mode. Once the gate is decided (or the header moves
        to another portfolio) the tail would otherwise keep claiming ``LIVE`` over a
        finalised run: the run picker stays hidden, read-out expansion is dead and
        barge-in targets the closed run. The watched run is selected on the way out so
        its final steps stay on screen in the replay transcript.

        Returns:
            ``True`` when live mode was left, ``False`` when the tail still applies.
        """
        watched = self.live_run_id
        if watched is None or not self._live_tail_is_spent():
            return False
        if self.shell._portfolio_id == self.live_portfolio_id:
            self.selected_run_id = watched
            self.selected_step = None
        self._leave_live()
        return True

    def _live_tail_is_spent(self) -> bool:
        """Whether the live tail no longer matches a gate the adviser can act on.

        Never spent while the stream is running. Afterwards the tail is spent when the
        header moved to another portfolio, or when the watched run is readable and has
        left both the live and the awaiting states (approved, rejected, failed...). A
        run that is not readable keeps the tail rather than dropping it on a transient
        read error, and a pending/running row (visible to a reader that shares the
        stream's transaction) is still in flight, not decided.
        """
        shell = self.shell
        if shell._streaming:
            return False
        pid = self.live_portfolio_id
        if pid is None or shell._portfolio_id != pid:
            return True
        try:
            with self.read_session() as session:
                runs = observe.list_portfolio_runs(session, pid)
        except Exception:  # a failed read must not drop a tail that may still apply
            return False
        watched = next((run for run in runs if run.run_id == self.live_run_id), None)
        return watched is not None and watched.status not in (
            _RUN_AWAITING | _RUN_RUNNING
        )

    def _leave_live(self) -> None:
        self.live_run_id = None
        self.live_portfolio_id = None
        self.query_one("#delib-live", DataTable).display = False
        self.query_one("#delib-runs", DataTable).display = True
        self.query_one("#delib-steps", DataTable).display = True

    def action_follow_live(self) -> None:
        """Re-arm auto-follow and re-snap to the live tail (the view-scoped ``f``)."""
        self.following = True
        if self.live_run_id is not None:
            table = self.query_one("#delib-live", DataTable)
            if table.row_count:
                table.move_cursor(row=table.row_count - 1)
                table.scroll_end(animate=False)
        self._paint_live_subtitle()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Pause auto-follow when the adviser moves the live cursor off the tail."""
        if event.data_table.id != "delib-live":
            return
        self.following = event.cursor_row == event.data_table.row_count - 1
        self._paint_live_subtitle()

    def _paint_live_subtitle(self) -> None:
        follow = "following" if self.following else "inspecting (f to follow)"
        self.query_one("#delib-subtitle", Label).update(f"▶ LIVE · {follow}")

    def _paint_live_detail(self, event: StreamEvent) -> None:
        text = _stream_detail(event)
        self.live_text = text
        self.query_one("#delib-detail", Static).update(text)

    def select_run(self, run_id: uuid.UUID) -> None:
        """Focus a run's transcript; a new run clears the step, then repaint."""
        self.selected_run_id = run_id
        self.selected_step = None
        self.refresh_view()

    def select_step(self, index: int) -> None:
        """Focus a transcript step by position, then repaint the detail pane."""
        self.selected_step = index
        self.refresh_view()

    def expand_target(self) -> tuple[uuid.UUID, str, int | None] | None:
        """The ``(run_id, agent, key)`` the expand key should recompute, or ``None``.

        Replay-only — recompute never runs on a ``LIVE`` step, so the walk-forward
        backtest is gated off the live stream. ``key`` is the selected allocator
        decision's ``decision_index`` so the frontier/backtest reflect *that* proposal
        (a run may re-optimize), and ``None`` for a risk step, which the worker checks
        against the run's authoritative allocation.

        Returns:
            The focused run, the selected step's agent, and the per-step key when a
            recompute applies, else ``None`` (live mode, nothing selected, a non-
            ``optimize_portfolio`` allocator row, or an unmapped step).
        """
        if self.live_run_id is not None or self.selected_run_id is None:
            return None
        if self.selected_step is None or self.selected_step >= len(self._entries):
            return None
        entry = self._entries[self.selected_step]
        if entry.agent == "allocator":
            if entry.source != "decision" or entry.step != "optimize_portfolio":
                return None
            return self.selected_run_id, "allocator", entry.index
        if entry.agent == "risk":
            return self.selected_run_id, "risk", None
        return None

    def paint_readout(
        self, run_id: uuid.UUID, agent: str, key: int | None, result: Any
    ) -> None:
        """Store a worker's recompute and repaint the detail pane.

        Called via ``call_from_thread`` from the recompute worker. The result is stored
        keyed by ``(run_id, agent, key)`` and only painted while that matches the
        selection (see :meth:`_compute_for`). Skips the repaint in live mode — a replay
        recompute must never overwrite the streamed detail (mirrors
        :meth:`refresh_view`'s live guard); it stays stored for when replay resumes.
        """
        self._readout = (run_id, agent, key, result)
        if self.live_run_id is not None:
            return
        self._paint_detail()

    def _compute_for(self, entry: Any) -> Any:
        """The stored recompute for ``entry``, or ``None`` for another run/step.

        An allocator result matches only the exact proposal it was computed for (by
        ``decision_index``), so an earlier step never shows a later re-optimize's
        frontier; a risk result matches the run's risk step. A ``message`` row never
        matches an allocator result: message and decision indices are separate
        numberings that merely look alike.
        """
        if self._readout is None:
            return None
        run_id, agent, key, result = self._readout
        if run_id != self.selected_run_id or agent != entry.agent:
            return None
        if agent == "allocator" and (entry.source != "decision" or key != entry.index):
            return None
        return result

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Dispatch: a run row loads its transcript, a step row shows its detail."""
        key = event.row_key.value
        if key is None:
            return
        if event.data_table.id == "delib-runs":
            self.select_run(uuid.UUID(key))
        elif event.data_table.id == "delib-steps":
            self.select_step(int(key))

    def _read(self) -> tuple[list[RunSummary], list[Any]]:
        """Read the focused portfolio's runs + the selected run's transcript.

        Returns ``([], [])`` when nothing is focused or on any read error, so the poll
        loop stays alive.
        """
        pid = self.shell._portfolio_id
        if pid is None:
            return [], []
        try:
            with self.read_session() as session:
                runs = observe.list_portfolio_runs(session, pid)
                return runs, self._read_transcript(session, runs)
        except Exception:  # keep the poll loop alive on any read error
            return [], []

    def _read_transcript(self, session: Session, runs: list[RunSummary]) -> list[Any]:
        """Load the selected run's transcript while it is still in the live list.

        Imports the run repo from its submodule (never the ``fund.audit`` root, which
        pulls the langgraph bootstrap); needs the ORM run row for its ``thread_id`` +
        ``decisions``.
        """
        if self.selected_run_id is None or all(
            run.run_id != self.selected_run_id for run in runs
        ):
            return []
        from fund.audit.repository import AgentRunRepository

        run_row = AgentRunRepository(session).get_run(self.selected_run_id)
        if run_row is None:
            return []
        saver = self.shell._persistence.saver
        return observe.load_run_transcript(session, saver, run_row)

    def _paint_subtitle(self) -> None:
        pid = self.shell._portfolio_id
        subtitle = self.query_one("#delib-subtitle", Label)
        if pid is None:
            subtitle.update("(no portfolio focused — pick one in the header)")
        else:
            subtitle.update(f"portfolio {_short(pid)}")

    def _paint_runs(self, runs: list[RunSummary]) -> None:
        table = self.query_one("#delib-runs", DataTable)
        with _keep_cursor(table, self.shell._portfolio_id):
            table.clear()
            for run in runs:
                table.add_row(
                    _short(run.run_id),
                    run.asof.isoformat(),
                    run.status,
                    key=str(run.run_id),
                )
        if self.selected_run_id is not None and all(
            run.run_id != self.selected_run_id for run in runs
        ):
            self.selected_run_id = None
            self.selected_step = None

    def _paint_steps(self, entries: list[Any]) -> None:
        self._entries = entries
        table = self.query_one("#delib-steps", DataTable)
        with _keep_cursor(table, self.selected_run_id):
            table.clear()
            for position, entry in enumerate(entries):
                table.add_row(
                    str(entry.index),
                    entry.agent,
                    entry.step or entry.source,
                    key=str(position),
                )
        if self.selected_step is not None and self.selected_step >= len(entries):
            self.selected_step = None

    def _paint_detail(self) -> None:
        if self.selected_step is not None and self.selected_step < len(self._entries):
            entry = self._entries[self.selected_step]
            text = _transcript_detail(entry, self._compute_for(entry))
        else:
            text = "(select a step)"
        self.detail_text = text
        self.query_one("#delib-detail", Static).update(text)


def _transcript_detail(entry: Any, compute: Any = None) -> str:
    """Render a transcript step: header, text, then a read-out or the raw payload.

    An allocator/risk step routes its payload through :func:`render_readout` for the
    build/results/risk sub-panels; any other step keeps the raw JSON dump. ``compute``
    is the worker's recompute for this step, or ``None``.
    """
    lines = [f"{entry.agent} · {entry.step or entry.source} · #{entry.index}"]
    if entry.text:
        lines.append(entry.text)
    body = _detail_payload(entry.agent, entry.step, entry.payload, compute)
    if body:
        lines.append(body)
    return "\n".join(lines)


def _stream_detail(event: StreamEvent) -> str:
    """Render a live step: header, summary, then a read-out or the raw payload.

    Mirrors :func:`_transcript_detail` so live and replay render alike.
    """
    lines = [f"{event.agent} · {event.step or event.kind} · #{event.index}"]
    if event.text:
        lines.append(event.text)
    body = _detail_payload(event.agent, event.step, event.payload)
    if body:
        lines.append(body)
    return "\n".join(lines)


def _detail_payload(
    agent: str, step: str | None, payload: dict[str, Any] | None, compute: Any = None
) -> str:
    """Structured read-out for an allocator/risk step, else the raw JSON dump (or "").

    A risk step carries no payload, so :func:`render_readout` is always consulted; only
    an unmapped step with a payload falls back to the JSON dump. ``compute`` fills the
    recomputed frontier/backtest/verdict/sector lines when present.
    """
    readout = render_readout(agent, step, payload, compute)
    if readout is not None:
        return readout
    if payload:
        return json.dumps(payload, indent=2, sort_keys=True)
    return ""


# Daily-brief P&L tiers as (dot, label) — the word travels with the glyph so the day's
# direction never rides on colour alone (mirrors _HEALTH_*). The two ● tiers (up/down)
# read apart by their label; ◐ marks a flat day.
_BRIEF_UP: tuple[str, str] = ("●", "up")
_BRIEF_FLAT: tuple[str, str] = ("◐", "flat")
_BRIEF_DOWN: tuple[str, str] = ("●", "down")

_NO_DATA = "  no data"
# Worded rather than coloured so the caveat survives monochrome terminals; a fit this
# close to saturation shows an R2 and betas that look more precise than they are.
_SHORT_HISTORY_NOTE = "short history: thin window, OLS fallback; read with caution"


class DailyBriefView(ShellView):
    """The per-day recap: today's P&L, macro attribution, events, the economist read.

    Reads ``observe.daily_brief(session, saver, pid, asof=self.asof)`` each tick and
    renders its five sections into one text pane — the P&L strip (dot + label
    direction), the macro-attribution meters, the ``market_journal`` narrative, the
    carried-forward economist read (dated ``(from MM-DD)`` when it predates the day),
    and the ``portfolio_journal`` overlays. A missing section degrades to an explicit
    "no data" line, never a crash. ``self.asof`` is the viewed day (``None`` opens on
    the latest NAV mark); day navigation and export arrive in later slices.

    Attributes:
        asof: The viewed day, or ``None`` to open on the latest NAV-marked day.
        brief: The last-assembled :class:`~fund.observe.DailyBrief`, or ``None`` — a
            headless-test hook.
        detail_text: Mirror of the rendered detail pane — a headless-test hook.
    """

    # Focusable so ← / → reach the view's bindings when the brief pane holds focus —
    # a view-scoped scope, never a global arrow hijack (the day state is view-local).
    can_focus = True
    # ``e`` is view-scoped so it exports the brief here and leaves the global Audit
    # export ``e`` untouched (mirrors DecisionsView overriding ``e`` for edit).
    BINDINGS: ClassVar[list[BindingType]] = [
        ("left", "prev_day", "Prev day"),
        ("right", "next_day", "Next day"),
        ("enter", "app.open_deliberation", "Open deliberation"),
        ("e", "app.export_brief", "Export"),
    ]

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.asof: dt.date | None = None
        self.brief: DailyBrief | None = None
        self.detail_text = ""

    def compose(self) -> ComposeResult:
        """Yield the title, the five-section detail pane, and the key hint."""
        yield Label(self.view_title, classes="view-title")
        yield Static(
            "(loading…)", id="brief-detail", classes="brief-detail", markup=False
        )
        yield Static(
            "← prev day · → next · ↵ open deliberation · e export · esc",
            id="brief-hint",
            classes="brief-hint",
        )

    def refresh_view(self) -> None:
        """Repaint the recap for the viewed day from a fresh assembler read."""
        self.render_count += 1
        self.brief = self._read()
        text = (
            _render_brief_detail(self.brief)
            if self.brief is not None
            else "(no portfolio focused)"
        )
        self.detail_text = text
        self.query_one("#brief-detail", Static).update(text)

    def _read(self) -> DailyBrief | None:
        """Assemble the brief in one short session (None on no focus or any error)."""
        pid = self.shell._portfolio_id
        if pid is None:
            return None
        try:
            with self.read_session() as session:
                saver = self.shell._persistence.saver
                return observe.daily_brief(session, saver, pid, asof=self.asof)
        except Exception:  # keep the poll loop alive on any read error
            return None

    def action_prev_day(self) -> None:
        """Step to the previous NAV-marked day; a no-op at the earliest mark."""
        self._step_day(self.brief.prev_day if self.brief else None)

    def action_next_day(self) -> None:
        """Step to the next NAV-marked day; a no-op at the latest mark."""
        self._step_day(self.brief.next_day if self.brief else None)

    def _step_day(self, target: dt.date | None) -> None:
        """Re-read the recap for ``target``; a None target (a grid end) is a no-op."""
        if target is None:
            return
        self.asof = target
        self.refresh_view()

    def export_markdown(self, path: str) -> str:
        """Write the current brief to ``path`` as markdown; return ``path``.

        Re-export overwrites (idempotent). The caller guards on a loaded brief; a
        missing brief raises ``ValueError`` rather than writing an empty file. Any
        write failure surfaces as ``OSError`` for the caller to report.
        """
        from pathlib import Path

        if self.brief is None:
            raise ValueError("no brief loaded to export")
        Path(path).write_text(_render_brief_markdown(self.brief), encoding="utf-8")
        return path


def _render_brief_detail(brief: DailyBrief) -> str:
    """Render a DailyBrief's five sections as detail-pane text."""
    lines = [
        f"Daily brief · {brief.asof.isoformat()}",
        _brief_pnl_line(brief.performance),
        "",
        "MACRO ATTRIBUTION (today)",
        *_brief_attribution_lines(brief.attribution),
        "",
        "WHAT HAPPENED (market_journal)",
        *_brief_event_lines(brief.market_events),
        "",
        "ECONOMIST READ",
        *_brief_economist_lines(brief),
        "",
        "FUND DID (portfolio_journal)",
        *_brief_event_lines(brief.fund_events),
    ]
    return "\n".join(lines)


def _sorted_contributions(attribution: Any) -> list[tuple[str, float]]:
    """Return contributions plus the ``idio`` residual, descending by magnitude."""
    items = [*attribution.contributions.items(), ("idio", attribution.residual)]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items


def _carried_suffix(brief: DailyBrief) -> str:
    """Return the ``(from MM-DD)`` tag when the read predates the day, else ``""``."""
    read_date = brief.economist_read_date
    if read_date is not None and read_date != brief.asof:
        return f" (from {read_date:%m-%d})"
    return ""


def _brief_pnl_line(perf: Any) -> str:
    """Render the P&L strip with a dot + label direction, or a no-data line."""
    if perf is None:
        return f"TODAY {_NO_DATA.strip()}"
    day_return = perf.returns.get("1D")
    if day_return is None:
        return f"TODAY {_NO_DATA.strip()}"
    if day_return > 0:
        dot, label = _BRIEF_UP
    elif day_return < 0:
        dot, label = _BRIEF_DOWN
    else:
        dot, label = _BRIEF_FLAT
    excess = perf.excess.get("1D")
    vs_bench = f" · vs bench {excess:+.2%}" if excess is not None else ""
    return f"TODAY  {dot} {label}  P&L {day_return:+.2%}{vs_bench}"


def _brief_attribution_lines(attribution: Any) -> list[str]:
    """Render per-factor contributions (+ ``idio`` residual) as sorted bar meters."""
    if attribution is None or not attribution.contributions:
        return [_NO_DATA]
    items = _sorted_contributions(attribution)
    scale = max(abs(value) for _, value in items)
    lines = [
        f"  {name:<6} {value:+.2%}  {attribution_bar(value, scale)}"
        for name, value in items
    ]
    lines.append(f"  {'total':<6} {attribution.total_return:+.2%}")
    if attribution.insufficient_history:
        lines.append(f"  ({_SHORT_HISTORY_NOTE})")
    return lines


def _brief_event_lines(events: tuple[dict[str, Any], ...]) -> list[str]:
    """Render journal rows as dated narrative bullets, or a no-data line."""
    if not events:
        return [_NO_DATA]
    return [
        f"  • {event.get('narrative', '')}  ({event.get('as_of', '')})"
        for event in events
    ]


def _brief_economist_lines(brief: DailyBrief) -> list[str]:
    """Render the economist read, dated ``(from MM-DD)`` when carried forward."""
    if brief.economist_read is None:
        return [_NO_DATA]
    return [f'  "{brief.economist_read.strip()}"{_carried_suffix(brief)}']


def _render_brief_markdown(brief: DailyBrief) -> str:
    """Serialize a DailyBrief to a markdown recap, one heading per section."""
    lines = [
        f"# Daily brief · {brief.portfolio_id} · {brief.asof.isoformat()}",
        "",
        "## Today's P&L",
        _brief_pnl_line(brief.performance),
        "",
        "## Macro attribution (today)",
        *_brief_attribution_table(brief.attribution),
        "",
        "## What happened",
        *_brief_markdown_events(brief.market_events),
        "",
        "## Economist read",
        *_brief_markdown_economist(brief),
        "",
        "## Fund did",
        *_brief_markdown_events(brief.fund_events),
        "",
    ]
    return "\n".join(lines)


def _brief_attribution_table(attribution: Any) -> list[str]:
    """Render attribution contributions (+ ``idio`` residual) as a markdown table."""
    if attribution is None or not attribution.contributions:
        return ["_no data_"]
    rows = _sorted_contributions(attribution)
    table = ["| factor | contribution |", "| --- | --- |"]
    table += [f"| {name} | {value:+.2%} |" for name, value in rows]
    table.append(f"| **total** | **{attribution.total_return:+.2%}** |")
    if attribution.insufficient_history:
        table += ["", f"_{_SHORT_HISTORY_NOTE}_"]
    return table


def _brief_markdown_events(events: tuple[dict[str, Any], ...]) -> list[str]:
    """Render journal rows as markdown bullets, or a no-data line."""
    if not events:
        return ["_no data_"]
    return [
        f"- {event.get('narrative', '')} ({event.get('as_of', '')})" for event in events
    ]


def _brief_markdown_economist(brief: DailyBrief) -> list[str]:
    """Render the economist read as a dated markdown blockquote."""
    if brief.economist_read is None:
        return ["_no data_"]
    return [f"> {brief.economist_read.strip()}{_carried_suffix(brief)}"]


# Performance P&L direction as (dot, label) — the word travels with the glyph so a
# gain/loss never rides on colour alone (mirrors _HEALTH_* / _BRIEF_*). The two ●
# tiers (up/down) read apart by their label; ◐ marks a flat/unknown period.
_PERF_UP: tuple[str, str] = ("●", "up")
_PERF_FLAT: tuple[str, str] = ("◐", "flat")
_PERF_DOWN: tuple[str, str] = ("●", "down")

# Return horizons the metrics strip surfaces, in display order (mirrors
# fund.performance._HORIZONS; kept local so the module scope stays Textual + observe).
_PERF_HORIZONS: tuple[str, ...] = ("1D", "1M", "YTD", "1Y", "max")


def _pct(value: float | None) -> str:
    """A signed-percent metric cell, or ``—`` when the metric is unavailable.

    The dash is load-bearing: an unmarked horizon degrades to ``—`` rather than a
    fabricated ``0.00%`` so the strip never implies a figure the NAV series lacks.
    """
    return f"{value:+.2%}" if value is not None else "—"


def _ratio(value: float | None) -> str:
    """A 2-dp ratio cell (sharpe / beta), or ``—`` when unavailable."""
    return f"{value:.2f}" if value is not None else "—"


def _amt(value: float | None) -> str:
    """A grouped 2-dp amount cell (value / shares / P&L), or ``—`` when unavailable."""
    return f"{value:,.2f}" if value is not None else "—"


def _perf_direction(period_return: float | None) -> tuple[str, str]:
    """The ``(dot, label)`` P&L direction for a period return.

    A missing or exactly-flat return reads ``◐ flat`` — the strip signals direction
    with a dot **and** a word, never colour alone.
    """
    if period_return is None or period_return == 0:
        return _PERF_FLAT
    return _PERF_UP if period_return > 0 else _PERF_DOWN


class PerformanceView(ShellView):
    """The lightweight performance surface: a metrics strip + a marked holdings table.

    Reads the focused portfolio through ``observe.performance_summary`` (the persisted
    ``nav_history`` series → multi-horizon returns + sharpe/vol/max-dd/excess/TE) and
    ``observe.holdings_summary`` (the current paper book marked to market) on one short
    read session each tick. The strip leads with a dot + label P&L direction; the
    holdings table lists per-holding value / 1-day return / contribution / unrealized
    P&L, and selecting a row drills into that holding's full mark. No equity-curve chart
    (it renders poorly in a TUI). An unmarked portfolio degrades to an all-``—`` strip
    and an empty table rather than a fabricated figure.

    Both reads go through the agent-stack-free ``fund.observe`` seam, which lazy-imports
    the ``optimizer``/``skfolio`` tracker inside its body — so ``import fund.tui.*``
    stays agent-stack-light and pulls no optimizer/skfolio at module scope.

    Attributes:
        summary: The last-read :class:`~fund.performance.PerformanceSummary`, or
            ``None`` — a headless-test hook.
        summary_text: Mirror of the rendered metrics strip — a headless-test hook.
        detail_text: Mirror of the rendered holding drill-down — a headless-test hook.
        selected_ticker: The drilled-into holding's ticker, or ``None``.
    """

    primary_focus: ClassVar[str | None] = "#perf-holdings"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.summary: PerformanceSummary | None = None
        self.summary_text: str = ""
        self.detail_text: str = ""
        self.selected_ticker: str | None = None
        self._marks: list[HoldingMark] = []

    def compose(self) -> ComposeResult:
        """Yield the title, returns + risk strip, holdings | drill split, and hint."""
        yield Label(self.view_title, classes="view-title")
        yield Static(
            "(no portfolio focused)", id="perf-returns", classes="view-subtitle"
        )
        yield Static("", id="perf-risk", classes="perf-risk")
        with Horizontal(classes="perf-body"):
            yield DataTable(
                id="perf-holdings", cursor_type="row", classes="perf-holdings"
            )
            yield Static("(select a holding)", id="perf-detail", classes="perf-detail")
        yield Static(
            "↑↓ select · ↵ drill into a holding",
            id="perf-hint",
            classes="perf-hint",
        )

    def on_mount(self) -> None:
        """Seed the holdings table's columns once its DataTable has mounted."""
        self.query_one("#perf-holdings", DataTable).add_columns(
            "ticker", "value", "1D", "contrib", "P&L"
        )

    def refresh_view(self) -> None:
        """Repaint the metrics strip, the holdings table, and the drill-down detail."""
        self.render_count += 1
        pid = self.shell._portfolio_id
        if pid is None:
            self.summary = None
            self._marks = []
            self.summary_text = ""
            self.query_one("#perf-returns", Static).update(
                "(no portfolio focused — pick one in the header)"
            )
            self.query_one("#perf-risk", Static).update("")
            self._paint_holdings([])
            self._paint_detail()
            return
        summary, marks = self._read(pid)
        self.summary = summary
        self._paint_strip(summary)
        self._paint_holdings(marks)
        self._paint_detail()

    def select_ticker(self, ticker: str) -> None:
        """Drill into a holding by ticker, then repaint the detail pane.

        A driver/test hook mirroring the DataTable row selection; the mark is taken
        from the last-painted book so no extra read is needed.
        """
        self.selected_ticker = ticker
        self._paint_detail()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A holdings-row selection drills into that holding's detail."""
        key = event.row_key.value
        if key is not None:
            self.select_ticker(key)

    def _read(
        self, pid: uuid.UUID
    ) -> tuple[PerformanceSummary | None, list[HoldingMark]]:
        """Read the summary + the marked book, each guarded independently.

        The two reads are isolated so a holdings-mark failure — e.g. an
        ``FxCoverageError`` when the book holds a currency the FX feed never populated —
        degrades only the holdings table, never the FX-independent returns/risk strip.
        Either read failing returns its own empty sentinel; the poll loop stays alive on
        any read error. The tracker is lazy-imported inside ``observe``, keeping this
        module light.
        """
        summary: PerformanceSummary | None = None
        marks: list[HoldingMark] = []
        try:
            with self.read_session() as session:
                try:
                    summary = observe.performance_summary(session, pid)
                except Exception:  # keep the strip independent of a holdings failure
                    summary = None
                try:
                    marks = observe.holdings_summary(session, pid)
                except Exception:  # an FX gap degrades only the table, not the strip
                    marks = []
        except Exception:  # read_session itself failing → keep the poll loop alive
            return None, []
        return summary, marks

    def _paint_strip(self, summary: PerformanceSummary | None) -> None:
        returns_widget = self.query_one("#perf-returns", Static)
        risk_widget = self.query_one("#perf-risk", Static)
        if summary is None:
            self.summary_text = ""
            returns_widget.update("(performance unavailable)")
            risk_widget.update("")
            return
        dot, label = _perf_direction(summary.returns.get("max"))
        horizons = " · ".join(
            f"{h} {_pct(summary.returns.get(h))}" for h in _PERF_HORIZONS
        )
        returns_line = f"RETURNS  {dot} {label}  {horizons}"
        excess = " · ".join(
            f"{h} {_pct(summary.excess.get(h))}" for h in _PERF_HORIZONS
        )
        risk_line = (
            f"RISK  sharpe {_ratio(summary.sharpe)} · vol {_pct(summary.volatility)} · "
            f"max-dd {_pct(summary.max_drawdown)} · "
            f"TE {_pct(summary.tracking_error)} · β {_ratio(summary.beta)}\n"
            f"EXCESS  {excess}"
        )
        returns_widget.update(returns_line)
        risk_widget.update(risk_line)
        self.summary_text = f"{returns_line}\n{risk_line}"

    def _paint_holdings(self, marks: list[HoldingMark]) -> None:
        self._marks = marks
        table = self.query_one("#perf-holdings", DataTable)
        with _keep_cursor(table, self.shell._portfolio_id):
            table.clear()
            for mark in marks:
                table.add_row(
                    mark.ticker,
                    _amt(mark.value),
                    _pct(mark.return_1d),
                    _pct(mark.contribution),
                    _amt(mark.unrealized_pnl),
                    key=mark.ticker,
                )
        # Drop a drill-down selection that has left the book (e.g. after a rebalance).
        if self.selected_ticker is not None and all(
            mark.ticker != self.selected_ticker for mark in marks
        ):
            self.selected_ticker = None

    def _paint_detail(self) -> None:
        mark = next((m for m in self._marks if m.ticker == self.selected_ticker), None)
        text = _holding_detail(mark) if mark is not None else "(select a holding)"
        self.detail_text = text
        self.query_one("#perf-detail", Static).update(text)


def _holding_detail(mark: HoldingMark) -> str:
    """Render a holding's full mark: value, weight, shares, returns, contribution."""
    return "\n".join(
        [
            f"{mark.ticker}",
            f"value {_amt(mark.value)} · weight {_pct(mark.weight)}",
            f"shares {_amt(mark.shares)}",
            f"return 1D {_pct(mark.return_1d)} · "
            f"1M {_pct(mark.return_1m)} · 1Y {_pct(mark.return_1y)}",
            f"contribution {_pct(mark.contribution)}",
            f"unrealized P&L {_amt(mark.unrealized_pnl)}",
        ]
    )


class AskView(ShellView):
    """Conversational Q&A over the fund state: a log, an input, and a grounded answer.

    Read-first. Each turn is classified ``query | action`` (see
    :func:`fund.tui.ask.classify`); recognized action verbs are the decision verbs only
    (approve / reject / killswitch / freeze / re-arm / run / steer). A query is answered
    from ``fund.observe`` reads summarized by the lazily-built model on a worker thread
    the App owns; an action turn resolves its target and routes through the shared
    confirm + ``apply_action`` path — never silent execution. The view itself only
    decides and hands off; the App owns the model and every mutation.

    At module scope this imports only Textual, the stdlib, and the agent-stack-free
    classify/retrieve helpers (:mod:`fund.tui.ask`), so ``import fund.tui.app`` stays
    agent-stack-light.

    Attributes:
        last_answer: Mirror of the most recent rendered answer — a headless-test hook.
        pending_action: A resolved-but-unconfirmed action awaiting disambiguation, or
            ``None`` — the cross-turn state the disambiguation flow carries.
    """

    # Focusable so the view-scoped ``e`` export reaches this view's binding; ``e`` is
    # view-scoped here (mirrors DailyBriefView) so it exports the Ask conversation and
    # leaves the global Audit export ``e`` untouched.
    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        ("e", "app.export_ask", "Export"),
    ]

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.last_answer: str = ""
        self.pending_action: ask.PendingAction | None = None
        self._log: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        """Yield the title, the scrollable conversation log, the input, and the hint."""
        yield Label(self.view_title, classes="view-title")
        with VerticalScroll(id="ask-log-wrap", classes="ask-log-wrap"):
            # markup=False: the log holds LLM answers and exception text, whose
            # "[AAPL]" / "[/INST]" fragments the markup parser would drop or reject.
            yield Static(
                "ask about the fund — e.g. “how are we doing?” or “approve the gate”",
                id="ask-log",
                classes="ask-log",
                markup=False,
            )
        yield Input(placeholder="ask…", id="ask-input")
        yield Static("↵ ask · e export", id="ask-hint", classes="ask-hint")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """A submitted turn is classified and routed, then the input is cleared."""
        if event.input.id != "ask-input":
            return
        event.stop()
        turn = event.value.strip()
        event.input.value = ""
        if turn:
            self.submit_turn(turn)

    def submit_turn(self, turn: str) -> None:
        """Log a turn and hand it to the App to resolve a pending action or route it.

        The App resolves a carried disambiguation first, else classifies the turn — a
        ``query`` dispatches the lazy summarize worker; an ``action`` routes through the
        shared confirm + ``apply_action`` path. Nothing mutates here — the view only
        logs and hands off, so an action never executes from this call.
        """
        self.append_log("you", turn)
        self.shell.ask_submit(turn)

    def append_log(self, speaker: str, text: str) -> None:
        """Append a ``(speaker, text)`` line to the conversation and repaint the log."""
        self._log.append((speaker, text))
        self._repaint_log()

    def set_answer(self, answer: str) -> None:
        """Record the model's answer (the ``last_answer`` hook) and log it."""
        self.last_answer = answer
        self.append_log("ask", answer)

    def refresh_view(self) -> None:
        """Bump the counter; the log is push-driven by submit/answer, not polled."""
        self.render_count += 1

    def _repaint_log(self) -> None:
        body = "\n".join(f"{speaker}: {text}" for speaker, text in self._log)
        self.query_one("#ask-log", Static).update(body or "ask about the fund…")
        self.query_one("#ask-log-wrap", VerticalScroll).scroll_end(animate=False)

    def export_markdown(self, path: str) -> str:
        """Write the conversation to ``path`` as markdown; return ``path``.

        Re-export overwrites (idempotent). Each turn renders as a bolded speaker label
        over its text. Any write failure surfaces as ``OSError`` for the caller to
        report on the status line.
        """
        from pathlib import Path

        lines = ["# Ask conversation", ""]
        for speaker, text in self._log:
            lines.append(f"**{speaker}:** {text}")
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        return path


# Per-key view class; keys absent here fall back to the placeholder ``ShellView``.
_VIEW_CLASSES: dict[str, type[ShellView]] = {
    "posture": PostureView,
    "deliberation": DeliberationView,
    "decisions": DecisionsView,
    "mandate": MandateView,
    "audit": AuditView,
    "performance": PerformanceView,
    "ask": AskView,
    "brief": DailyBriefView,
}


def build_views() -> list[ShellView]:
    """Build one view per :data:`VIEW_SPECS`, id'd ``view-<key>``.

    Picks the real view class for keys in :data:`_VIEW_CLASSES` (currently
    ``decisions``); every other key still renders the placeholder ``ShellView``.
    """
    return [
        (_VIEW_CLASSES.get(key) or ShellView)(key, title, id=f"view-{key}")
        for key, title in VIEW_SPECS
    ]


__all__ = [
    "RAIL_LABELS",
    "VIEW_SPECS",
    "AskView",
    "AuditView",
    "DailyBriefView",
    "DecisionsView",
    "DeliberationView",
    "MandateView",
    "PerformanceView",
    "PostureView",
    "ShellView",
    "build_views",
]
