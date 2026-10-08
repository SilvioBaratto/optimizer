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
from textual.widgets import DataTable, Input, Label, Select, Static

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


# Per-key view class; keys absent here fall back to the placeholder ``ShellView``.
_VIEW_CLASSES: dict[str, type[ShellView]] = {
    "decisions": DecisionsView,
    "mandate": MandateView,
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


__all__ = ["VIEW_SPECS", "DecisionsView", "MandateView", "ShellView", "build_views"]
