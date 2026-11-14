"""The Posture view: the fund-wide roster, its health glyphs, a detail pane."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Label, Static

from fund import observe
from fund.tui._views.performance import _pct, _perf_direction
from fund.tui._views.shell import ShellView, _keep_cursor, _short, _short_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType


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
_HEALTH_UNREADABLE: tuple[str, str] = ("?", "error")


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

    def poll_task(self) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
        """Read the roster and detail off the loop: each portfolio's book is marked.

        The roster marks every portfolio's holdings against its price history and the
        detail marks the focused one again, so on the event loop a tick cost more the
        more portfolios the fund ran. The painted result is dropped when the focus
        moved while the read was in flight; the next tick repaints it.
        """
        pid = self.shell._portfolio_id
        session_factory = self.shell._session_factory

        def read() -> tuple[uuid.UUID | None, list[_RosterRow], str, str | None]:
            roster, strip = _load_roster(session_factory)
            detail = _load_posture_detail(session_factory, pid) if pid else None
            return pid, roster, strip, detail

        def apply(
            result: tuple[uuid.UUID | None, list[_RosterRow], str, str | None],
        ) -> None:
            read_pid, roster, strip, detail = result
            if read_pid != self.shell._portfolio_id:
                return
            self.render_count += 1
            self._paint_roster(roster, strip)
            self._show_detail(detail if read_pid is not None else None)

        return read, apply

    def _read_roster(self) -> tuple[list[_RosterRow], str]:
        """Read every portfolio's roster row + the fund-strip summary in one session.

        Returns ``([], "0 pf")`` when the portfolio list itself cannot be read, so the
        poll loop stays alive; a single unreadable portfolio degrades to an ``error``
        row instead. The strip mirrors the fund autonomy posture (ARMED/FROZEN) so it
        surfaces here too, not only in the header.
        """
        return _load_roster(self.shell._session_factory)

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
        self._show_detail(
            _load_posture_detail(self.shell._session_factory, pid)
            if pid is not None
            else None
        )

    def _show_detail(self, detail: str | None) -> None:
        """Paint the focused-portfolio detail; ``None`` is the unfocused prompt."""
        text = (
            detail if detail is not None else "(no portfolio focused — ↵ a roster row)"
        )
        self.detail_text = text
        self.query_one("#posture-detail", Static).update(text)


def _load_roster(
    session_factory: Callable[[], AbstractContextManager[Session]],
) -> tuple[list[_RosterRow], str]:
    """Every portfolio's roster row plus the fund strip, in one short session.

    Takes the session factory rather than a view so a poll worker thread can run it
    without touching the app. A failed portfolio list reads as ``([], "0 pf")``, and a
    single unreadable portfolio degrades to an ``error`` row, so the poll loop stays
    alive.
    """
    try:
        with _short_session(session_factory) as session:
            rows = [
                _roster_row_or_error(session, pid)
                for pid in observe.list_portfolios(session)
            ]
            posture = _read_autonomy(session)
    except Exception:  # keep the poll loop alive on any read error
        return [], "0 pf"
    return rows, f"{_fund_strip(rows)} · {posture}"


def _read_autonomy(session: Session) -> str:
    """The fund autonomy posture glyph, read off the agent-stack-free repo."""
    from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState

    # Read-only (``fund_posture`` never inserts) so a poll tick never mutates.
    posture = AutonomyRepository(session).fund_posture()
    return "❄ FROZEN" if posture == AutonomyState.FROZEN.value else "● ARMED"


def _roster_row_or_error(session: Session, pid: uuid.UUID) -> _RosterRow:
    """One portfolio's roster row, or an ``error`` row when its read fails.

    A stored mandate that no longer validates must not hide every healthy
    portfolio, so the failure stays scoped to its own row.
    """
    try:
        return _roster_row(session, pid)
    except Exception:  # keep the roster alive on one bad portfolio
        # A database-level error leaves the transaction aborted, which would fail
        # every later portfolio in this session too.
        session.rollback()
        dot, label = _HEALTH_UNREADABLE
        return _RosterRow(pid, None, "?", dot, label)


def _roster_row(session: Session, pid: uuid.UUID) -> _RosterRow:
    runs = observe.list_portfolio_runs(session, pid, limit=1)
    status = runs[0].status if runs else None
    mandate = observe.get_mandate(session, pid)
    threshold = mandate.drift_l1_threshold if mandate is not None else None
    drift = observe.portfolio_state(session, pid).drift_l1
    dot, label = _posture_health(drift, threshold, status)
    return _RosterRow(pid, status, _run_glyph(status), dot, label)


def _load_posture_detail(
    session_factory: Callable[[], AbstractContextManager[Session]], pid: uuid.UUID
) -> str:
    """Render the focused portfolio's detail in a short session (blank on failure)."""
    try:
        with _short_session(session_factory) as session:
            return _render_posture_detail(session, pid)
    except Exception:  # keep the poll loop alive on any read error
        return ""


def _fund_strip(rows: list[_RosterRow]) -> str:
    """The fund-aggregate strip: ``#pf / #awaiting / #running / #warning``."""
    n_awaiting = sum(1 for row in rows if row.run_status in _RUN_AWAITING)
    n_running = sum(1 for row in rows if row.run_status in _RUN_RUNNING)
    n_warning = sum(1 for row in rows if (row.dot, row.label) == _HEALTH_WARN)
    strip = (
        f"{len(rows)} pf · ⏸{n_awaiting} awaiting · "
        f"▶{n_running} running · ◐{n_warning} warning"
    )
    n_unreadable = sum(1 for row in rows if (row.dot, row.label) == _HEALTH_UNREADABLE)
    return f"{strip} · ?{n_unreadable} error" if n_unreadable else strip


def _render_posture_detail(session: Session, pid: uuid.UUID) -> str:
    """The focused-portfolio detail: state, NAV tile, mandate, active run, events.

    An un-run portfolio renders the ``no runs yet · p profile to start`` empty-state
    (the launchpad to onboarding). The NAV tile shows the real NAV-series period return
    from ``observe.performance_summary``; a run-but-unmarked portfolio degrades to a
    ``no marks yet`` line rather than a fabricated figure.
    """
    runs = observe.list_portfolio_runs(session, pid, limit=1)
    warnings = _mandate_warning_lines(session, pid)
    if not runs:
        return "\n".join(
            [f"portfolio {_short(pid)}", "no runs yet · p profile to start", *warnings]
        )
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
    lines.extend(warnings)
    lines.append(f"ACTIVE RUN #{_short(latest.run_id)} · {latest.status}")
    events = observe.recent_events(session, pid)
    if events.portfolio_events:
        recent = " · ".join(event["as_of"] for event in events.portfolio_events[:3])
        lines.append(f"RECENT {recent}")
    lines.append("Daily brief → d open")
    return "\n".join(lines)


def _mandate_warning_lines(session: Session, pid: uuid.UUID) -> list[str]:
    """The mandate's silent-disable warnings as detail lines (none if unreadable).

    An unreadable mandate already shows as the roster's error row, so it must not
    also blank the detail pane here.
    """
    try:
        return [f"⚠ {warning}" for warning in observe.mandate_warnings(session, pid)]
    except Exception:  # the roster row reports a mandate that does not validate
        return []


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
