"""The Performance view: the return and risk strip and the holdings it marks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Label, Static

from fund import observe
from fund.tui._views.shell import ShellView, _keep_cursor, _short_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session

    # Annotation-only: ``fund.performance`` pulls ``optimizer``/``skfolio``; with
    # ``from __future__ import annotations`` these are strings, so the Performance
    # view's reads route through ``fund.observe`` (lazy) and ``import fund.tui.*``
    # stays agent-stack-light (and optimizer/skfolio-free) at module scope.
    from fund.performance import HoldingMark, PerformanceSummary


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


def _unsigned_pct(value: float | None) -> str:
    """An unsigned-percent cell for a magnitude metric (volatility, tracking error).

    These are non-negative by construction, so a ``+`` would only imply a gain.
    """
    return f"{value:.2%}" if value is not None else "—"


def _drawdown_pct(value: float | None) -> str:
    """A drawdown cell shown as the loss it is (``-6.00%``), or ``—`` when unavailable.

    ``PerformanceSummary.max_drawdown`` is a non-negative magnitude, so the sign is
    applied here; a drawdown that rounds to zero stays unsigned rather than ``-0.00%``.
    """
    if value is None:
        return "—"
    shown = f"{abs(value):.2%}"
    return shown if shown == "0.00%" else f"-{shown}"


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


@dataclass(frozen=True)
class _PerformanceRead:
    """One poll tick's Performance reads, each failing independently.

    Attributes:
        summary: The metrics summary, or ``None`` when its read failed.
        marks: The marked book, or ``None`` when its read failed — distinct from an
            empty book, so the view keeps the last rows and drill-down selection.
        summary_exc: What the summary read raised, else ``None``.
        holdings_exc: What the holdings read raised, else ``None``.
    """

    summary: PerformanceSummary | None
    marks: list[HoldingMark] | None
    summary_exc: BaseException | None = None
    holdings_exc: BaseException | None = None


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
        holdings_error: Why the last holdings read failed (e.g. an FX gap), or ``None``
            when it succeeded; the last good rows and selection stay put meanwhile.
    """

    primary_focus: ClassVar[str | None] = "#perf-holdings"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.summary: PerformanceSummary | None = None
        self.summary_text: str = ""
        self.detail_text: str = ""
        self.selected_ticker: str | None = None
        self.holdings_error: str | None = None
        self._marks: list[HoldingMark] = []
        # The portfolio whose book the table holds, so a failed read never leaves
        # another portfolio's holdings on screen after a focus change.
        self._marks_pid: uuid.UUID | None = None

    def compose(self) -> ComposeResult:
        """Yield the title, returns + risk strip, holdings | drill split, and hint."""
        yield Label(self.view_title, classes="view-title")
        # The strip and detail carry read-error text, which can contain ``[...]``.
        yield Static(
            "(no portfolio focused)",
            id="perf-returns",
            classes="view-subtitle",
            markup=False,
        )
        yield Static("", id="perf-risk", classes="perf-risk")
        with Horizontal(classes="perf-body"):
            yield DataTable(
                id="perf-holdings", cursor_type="row", classes="perf-holdings"
            )
            yield Static(
                "(select a holding)",
                id="perf-detail",
                classes="perf-detail",
                markup=False,
            )
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
            self.holdings_error = None
            self._marks_pid = None
            self._paint_holdings([])
            self._paint_detail()
            return
        self._paint(pid, self._read(pid))

    def poll_task(self) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
        """Load the book off the loop: marking holdings reads every price history.

        The painted result is dropped when the focused portfolio changed while the
        read was in flight.
        """
        pid = self.shell._portfolio_id
        if pid is None:
            return None
        session_factory = self.shell._session_factory

        def read() -> tuple[uuid.UUID, _PerformanceRead]:
            return pid, _read_performance(session_factory, pid)

        def apply(result: tuple[uuid.UUID, _PerformanceRead]) -> None:
            read_pid, performance = result
            if read_pid != self.shell._portfolio_id:
                return
            self.render_count += 1
            self._paint(read_pid, performance)

        return read, apply

    def _paint(self, pid: uuid.UUID, read: _PerformanceRead) -> None:
        """Paint one tick's reads; a failed read keeps its last rows and says why.

        The failure bookkeeping (log once per distinct reason) runs here, on the UI
        thread, so the read itself can run on a worker without touching view state.
        """
        summary_error = self._failure("summary", read.summary_exc)
        self.holdings_error = self._failure("holdings", read.holdings_exc)
        self.summary = read.summary
        self._paint_strip(read.summary, summary_error)
        if read.marks is not None:
            self._paint_holdings(read.marks)
            self._marks_pid = pid
        elif self._marks_pid != pid:
            self._paint_holdings([])
            self._marks_pid = pid
        self._paint_detail()

    def _failure(self, source: str, exc: BaseException | None) -> str | None:
        """One-line reason a ``source`` read failed, or ``None`` when it succeeded."""
        if exc is None:
            self._read_recovered(source)
            return None
        return self._read_failed(source, exc)

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

    def _read(self, pid: uuid.UUID) -> _PerformanceRead:
        """Read the summary + the marked book for ``pid`` on the calling thread."""
        return _read_performance(self.shell._session_factory, pid)

    def _paint_strip(
        self, summary: PerformanceSummary | None, error: str | None = None
    ) -> None:
        returns_widget = self.query_one("#perf-returns", Static)
        risk_widget = self.query_one("#perf-risk", Static)
        if summary is None:
            self.summary_text = ""
            reason = f": {error}" if error else ""
            returns_widget.update(f"(performance unavailable{reason})")
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
            f"RISK  sharpe {_ratio(summary.sharpe)} · "
            f"vol {_unsigned_pct(summary.volatility)} · "
            f"max-dd {_drawdown_pct(summary.max_drawdown)} · "
            f"TE {_unsigned_pct(summary.tracking_error)} · β {_ratio(summary.beta)}\n"
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
        if self.holdings_error is not None:
            note = f"holdings unavailable: {self.holdings_error}"
            text = (
                note
                if mark is None
                else f"{note} (showing the last read)\n{_holding_detail(mark)}"
            )
        elif mark is not None:
            text = _holding_detail(mark)
        else:
            text = "(select a holding)"
        self.detail_text = text
        self.query_one("#perf-detail", Static).update(text)


def _read_performance(
    session_factory: Callable[[], AbstractContextManager[Session]], pid: uuid.UUID
) -> _PerformanceRead:
    """Read the summary + the marked book, each guarded independently.

    The two reads are isolated so a holdings-mark failure — e.g. an
    ``FxCoverageError`` when the book holds a currency the FX feed never populated —
    degrades only the holdings table, never the FX-independent returns/risk strip. A
    failed read leaves its field ``None`` with what it raised alongside (a failed
    holdings read is distinct from an empty book, so the view keeps its last rows); the
    poll loop stays alive on any read error. The tracker is lazy-imported inside
    ``observe``, keeping this module light. Safe to run on a worker thread: it takes
    the session factory as an argument rather than reaching for the App.
    """
    summary: PerformanceSummary | None = None
    marks: list[HoldingMark] | None = None
    summary_exc: BaseException | None = None
    holdings_exc: BaseException | None = None
    try:
        with _short_session(session_factory) as session:
            try:
                summary = observe.performance_summary(session, pid)
            except Exception as exc:  # keep the strip independent of a holdings failure
                summary_exc = exc
            try:
                marks = observe.holdings_summary(session, pid)
            except Exception as exc:  # an FX gap degrades only the table, not the strip
                holdings_exc = exc
    except Exception as exc:  # the session factory itself failing → keep the poll alive
        return _PerformanceRead(None, None, exc, exc)
    return _PerformanceRead(summary, marks, summary_exc, holdings_exc)


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
