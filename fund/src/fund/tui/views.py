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

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Input, Label, Select, Static

from fund import observe
from fund.observe import RunSummary
from fund.tui.stream import StepStatus, StreamEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType


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

    def focus_portfolio(self, portfolio_id: uuid.UUID) -> None:
        """Focus the shell on ``portfolio_id`` (set the header context from a view)."""
        ...


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
    steers the PM with a directive (``respond``), both dispatched by the App.

    Attributes:
        selected_run_id: The focused queued run, or ``None`` when nothing is selected.
        selected_portfolio_id: That run's portfolio (for the audit attribution).
        gate_text: Mirror of the rendered gate detail — a headless-test hook.
    """

    # View-scoped so ``e``/``s`` mean edit/steer only while Decisions holds focus; the
    # App's global ``e`` (Audit export) is untouched (R5). The ``app.`` prefix routes to
    # the App's action (the handlers live there, not on the view).
    BINDINGS: ClassVar[list[BindingType]] = [
        ("e", "app.edit_decision", "Edit weights"),
        ("s", "app.steer_decision", "Steer PM"),
    ]

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
            "a approve · r reject · k killswitch · e edit weights · s steer PM",
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
    """One-line summary of the numeric ConstraintSet caps (display-only this phase).

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


class MandateView(ShellView):
    """Per-portfolio policy editor: run-policy (persisted) + guardrails (audit-only).

    RUN POLICY mirrors the editable ``PortfolioMandate`` fields (capital, base currency,
    drift threshold, benchmark); saving upserts through the unified action path
    (``mandate_edit``). GUARDRAILS are read from the active MiFID profile — the
    suitability ``band`` is the ceiling, and proposing a *riskier* band logs a
    ``guardrail_override`` flagged ⚠ (audit-only this phase; enforcement is
    ``03-copilot``). Mandatory-stops + remembered-directives are rendered read-only.

    Reads through the context seam (``observe.get_mandate`` +
    ``MifidProfileRepository.get_active`` over a short read session); the writes go
    through ``fund.actions`` on a worker thread dispatched by the App.

    Attributes:
        mandate: The focused portfolio's current run-policy, or ``None``.
        suitability_band: The active assessment's ``RiskToleranceBand`` value (the
            ceiling), or ``None`` when the portfolio has no profile yet.
    """

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.mandate: Any = None
        self.suitability_band: str | None = None

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
                yield Input(placeholder="drift L1 threshold", id="mandate-drift")
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
                yield Static(
                    "mandatory stops + directives — read-only (live in 03-copilot)",
                    id="mandate-directives",
                    classes="mandate-readonly",
                )
        yield Static(
            "ctrl+s save run-policy · o override guardrail (⚠ if out of band)",
            id="mandate-hint",
            classes="mandate-hint",
        )

    def refresh_view(self) -> None:
        """Repaint the editor from the focused portfolio's mandate + active profile."""
        self.render_count += 1
        pid = self.shell._portfolio_id
        subtitle = self.query_one("#mandate-subtitle", Label)
        if pid is None:
            subtitle.update("(no portfolio focused — pick one in the header)")
            self.mandate = None
            self.suitability_band = None
            self._paint_policy(None)
            self._paint_guardrails(None, "")
            return
        subtitle.update(f"portfolio {_short(pid)}")
        mandate, band, caps = self._read(pid)
        self.mandate = mandate
        self.suitability_band = band
        self._paint_policy(mandate)
        self._paint_guardrails(band, caps)

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
        }
        return PortfolioMandate.model_validate(data)

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

    def _paint_policy(self, mandate: Any) -> None:
        capital = self.query_one("#mandate-capital", Input)
        base_currency = self.query_one("#mandate-base-currency", Input)
        drift = self.query_one("#mandate-drift", Input)
        benchmark = self.query_one("#mandate-benchmark", Input)
        triggers = self.query_one("#mandate-triggers", Static)
        if mandate is None:
            for field in (capital, base_currency, drift, benchmark):
                field.value = ""
            triggers.update("(no mandate — profile the portfolio first)")
            return
        capital.value = str(mandate.capital)
        base_currency.value = mandate.base_currency
        drift.value = str(mandate.drift_l1_threshold)
        benchmark.value = mandate.benchmark or ""
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
        if band in _RISK_BANDS:
            select.value = band
        caps_widget.update(caps)


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
    ``no runs yet · p profile to start`` empty-state. NAV/P&L are deferred to the
    performance tracker (Phase 05).

    Attributes:
        detail_text: Mirror of the rendered detail pane — a headless-test hook.
    """

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
        yield Static("↑↓ select · ↵ focus", id="posture-hint", classes="posture-hint")

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

        Returns ``([], "0 pf")`` on any read error so the poll loop stays alive.
        """
        try:
            with self.read_session() as session:
                rows = [
                    self._roster_row(session, pid)
                    for pid in observe.list_portfolios(session)
                ]
        except Exception:  # keep the poll loop alive on any read error
            return [], "0 pf"
        return rows, _fund_strip(rows)

    def _roster_row(self, session: Session, pid: uuid.UUID) -> _RosterRow:
        runs = observe.list_portfolio_runs(session, pid)
        status = runs[0].status if runs else None
        mandate = observe.get_mandate(session, pid)
        threshold = mandate.drift_l1_threshold if mandate is not None else None
        drift = observe.portfolio_state(session, pid).drift_l1
        dot, label = _posture_health(drift, threshold, status)
        return _RosterRow(pid, status, _run_glyph(status), dot, label)

    def _paint_roster(self, rows: list[_RosterRow], strip: str) -> None:
        self._roster = rows
        table = self.query_one("#posture-roster", DataTable)
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
    """The focused-portfolio detail: state, mandate, active run, recent events.

    An un-run portfolio renders the ``no runs yet · p profile to start`` empty-state
    (the launchpad to onboarding). NAV/P&L are placeholders until the Phase-05 tracker.
    """
    runs = observe.list_portfolio_runs(session, pid)
    if not runs:
        return f"portfolio {_short(pid)}\nno runs yet · p profile to start"
    latest = runs[0]
    state = observe.portfolio_state(session, pid)
    lines = [
        f"portfolio {_short(pid)} · {_run_glyph(latest.status)} {latest.status}",
        f"drift L1 {state.drift_l1:.4f} · "
        f"{len(state.current)} held → {len(state.target)} target",
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
    lines.append("NAV/P&L — pending performance tracker (Phase 05)")
    return "\n".join(lines)


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
            yield Static(
                "(select a decision)", id="audit-detail", classes="audit-detail"
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
    if entry.prev_value is not None:
        lines.append(f"prev: {entry.prev_value}")
    if entry.new_value is not None:
        lines.append(f"new: {entry.new_value}")
    if entry.source:
        lines.append(f"source: {entry.source}")
    if entry.idempotency_key:
        lines.append(f"idemp: {entry.idempotency_key}")
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
        following: Whether the live tail auto-follows; a cursor move pauses it.
        live_text: Mirror of the rendered live-step detail — a headless-test hook.
    """

    # View-scoped so ``f`` means follow-live only while Deliberation holds focus; the
    # App's global ``f`` (Audit overrides filter) is untouched (R2).
    BINDINGS: ClassVar[list[BindingType]] = [("f", "follow_live", "Follow live")]

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.selected_run_id: uuid.UUID | None = None
        self.selected_step: int | None = None
        self.detail_text = ""
        self._entries: list[Any] = []
        self.live_run_id: uuid.UUID | None = None
        self.following = True
        self.live_text = ""
        self._live_events: list[StreamEvent] = []

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
            yield Static("(select a step)", id="delib-detail", classes="delib-detail")
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
        streamed rows with the replay read.
        """
        self.render_count += 1
        if self.live_run_id is not None:
            self._paint_live_subtitle()
            return
        runs, entries = self._read()
        self._paint_subtitle()
        self._paint_runs(runs)
        self._paint_steps(entries)
        self._paint_detail()

    # --- live tail ----------------------------------------------------------

    def start_live(self, run_id: uuid.UUID) -> None:
        """Enter live mode for ``run_id``: swap the replay split for an empty timeline.

        Clears any prior stream, re-arms auto-follow, and hides the run/step picker so
        the streamed steps fill the pane. Idempotent per run — a fresh call restarts
        the timeline.
        """
        self.live_run_id = run_id
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

        A no-op unless a stream is live. The prior row's ● becomes ✓ before the new
        row lands, so markers need no second producer pass; the viewport snaps to the
        tail while :attr:`following`, and the detail pane mirrors the newest step.
        """
        if self.live_run_id is None:
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
        self.live_run_id = None
        self.query_one("#delib-live", DataTable).display = False
        self.query_one("#delib-runs", DataTable).display = True
        self.query_one("#delib-steps", DataTable).display = True
        self.refresh_view()

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
            text = _transcript_detail(self._entries[self.selected_step])
        else:
            text = "(select a step)"
        self.detail_text = text
        self.query_one("#delib-detail", Static).update(text)


def _transcript_detail(entry: Any) -> str:
    """Render one transcript step: header, its text, and the decision payload if any."""
    lines = [f"{entry.agent} · {entry.step or entry.source} · #{entry.index}"]
    if entry.text:
        lines.append(entry.text)
    if entry.payload:
        lines.append(json.dumps(entry.payload, indent=2, sort_keys=True))
    return "\n".join(lines)


def _stream_detail(event: StreamEvent) -> str:
    """Render one live step: header, its summary, and the interrupt payload if any."""
    lines = [f"{event.agent} · {event.step or event.kind} · #{event.index}"]
    if event.text:
        lines.append(event.text)
    if event.payload:
        lines.append(json.dumps(event.payload, indent=2, sort_keys=True))
    return "\n".join(lines)


# Per-key view class; keys absent here fall back to the placeholder ``ShellView``.
_VIEW_CLASSES: dict[str, type[ShellView]] = {
    "posture": PostureView,
    "deliberation": DeliberationView,
    "decisions": DecisionsView,
    "mandate": MandateView,
    "audit": AuditView,
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


__all__ = [
    "VIEW_SPECS",
    "AuditView",
    "DecisionsView",
    "DeliberationView",
    "MandateView",
    "PostureView",
    "ShellView",
    "build_views",
]
