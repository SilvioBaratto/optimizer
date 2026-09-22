"""The four observer panels for :class:`fund.tui.app.FundTUI` (Phase 8, Task 8).

Dumb renderers, thin over :mod:`fund.observe`: each panel exposes a ``show_*``
method that takes the model-free read-model dataclasses (``RunSummary`` /
``TranscriptEntry`` / ``PortfolioState``) or the interrupt dict and paints itself.
All data flow — the polling read session, the run selection, the worker-thread
resume — lives in the ``App``; a panel never opens a session, builds a model, or
imports the agent stack. Test-friendly counters (``entry_count`` / ``gate_text``)
mirror what was last rendered so headless ``App.run_test()`` assertions read state
without scraping widget internals.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, RichLog, Static

from fund.observe import PortfolioState, RunSummary, TranscriptEntry


def _short(value: Any) -> str:
    """First hex group of a UUID (or its string) — a compact table cell."""
    return str(value).split("-", 1)[0]


class TranscriptPanel(Vertical):
    """Panel 1: the selected run's merged narrative + structured transcript."""

    entry_count: int = 0

    def compose(self) -> ComposeResult:
        yield Label("Transcript", classes="panel-title")
        yield RichLog(id="transcript-log", wrap=True, highlight=False, markup=False)

    def show(self, entries: list[TranscriptEntry]) -> None:
        """Repaint the transcript log; RichLog auto-scrolls to the newest line."""
        log = self.query_one("#transcript-log", RichLog)
        log.clear()
        for entry in entries:
            step = f"/{entry.step}" if entry.step else ""
            log.write(
                f"[{entry.index}] {entry.agent}{step} ({entry.source}): {entry.text}"
            )
        self.entry_count = len(entries)


class PortfolioStatePanel(Vertical):
    """Panel 2: current-vs-target holdings, L1 drift, best-effort metrics."""

    def compose(self) -> ComposeResult:
        yield Label("Portfolio state", classes="panel-title")
        yield Label("drift_l1: -", id="drift-readout")
        yield DataTable(id="holdings-table", cursor_type="none")
        yield Label("metrics: -", id="metrics-readout")

    def on_mount(self) -> None:
        self.query_one("#holdings-table", DataTable).add_columns(
            "ticker", "current", "target", "Δ"
        )

    def show(self, state: PortfolioState | None) -> None:
        """Repaint the holdings table + drift/metrics readouts from ``state``."""
        table = self.query_one("#holdings-table", DataTable)
        drift = self.query_one("#drift-readout", Label)
        metrics = self.query_one("#metrics-readout", Label)
        table.clear()
        if state is None:
            drift.update("drift_l1: -")
            metrics.update("metrics: -")
            return
        drift.update(f"drift_l1: {state.drift_l1:.4f}")
        for ticker in sorted(set(state.current) | set(state.target)):
            current = state.current.get(ticker, 0.0)
            target = state.target.get(ticker, 0.0)
            table.add_row(
                ticker, f"{current:.4f}", f"{target:.4f}", f"{target - current:+.4f}"
            )
        metrics.update(
            "metrics: "
            + (
                ", ".join(f"{k}={v:.4f}" for k, v in sorted(state.metrics.items()))
                if state.metrics
                else "-"
            )
        )


class HitlQueuePanel(Vertical):
    """Panel 3: paused runs awaiting a human decision, plus the gate + buttons."""

    gate_text: str = ""

    def compose(self) -> ComposeResult:
        yield Label("HITL queue", classes="panel-title")
        yield DataTable(id="hitl-table", cursor_type="row")
        yield Static("(select a paused run)", id="gate-detail")
        with Horizontal(id="hitl-buttons"):
            yield Button("Approve", id="approve-btn", variant="success")
            yield Button("Reject", id="reject-btn", variant="error")

    def on_mount(self) -> None:
        self.query_one("#hitl-table", DataTable).add_columns("run", "asof", "weights")

    def show_runs(self, rows: list[RunSummary]) -> None:
        """Repaint the queue table, keyed by ``str(run_id)`` for row lookup."""
        table = self.query_one("#hitl-table", DataTable)
        table.clear()
        for row in rows:
            table.add_row(
                _short(row.run_id),
                row.asof.isoformat(),
                str(row.n_weights),
                key=str(row.run_id),
            )

    def show_gate(self, interrupt: dict[str, Any] | None) -> None:
        """Render the selected run's ``action_requests`` (the reviewable gate)."""
        text = _format_gate(interrupt)
        self.gate_text = text
        self.query_one("#gate-detail", Static).update(text)


class HistoryPanel(Vertical):
    """Panel 4: every run for the portfolio; selecting one drives panels 1-2."""

    def compose(self) -> ComposeResult:
        yield Label("History + audit", classes="panel-title")
        yield DataTable(id="history-table", cursor_type="row")

    def on_mount(self) -> None:
        self.query_one("#history-table", DataTable).add_columns(
            "run", "status", "asof", "weights", "hitl"
        )

    def show_runs(self, rows: list[RunSummary]) -> None:
        """Repaint the history table, keyed by ``str(run_id)`` for row lookup."""
        table = self.query_one("#history-table", DataTable)
        table.clear()
        for row in rows:
            table.add_row(
                _short(row.run_id),
                row.status,
                row.asof.isoformat(),
                str(row.n_weights),
                "yes" if row.awaiting_hitl else "-",
                key=str(row.run_id),
            )


def _format_gate(interrupt: dict[str, Any] | None) -> str:
    """One-line-per-request rendering of an interrupt's ``action_requests``."""
    if not interrupt:
        return "(no pending gate)"
    requests = interrupt.get("action_requests") or []
    lines: list[str] = []
    for request in requests:
        name = request.get("name", "?")
        lines.append(f"gate: {name}")
        weights = (request.get("args") or {}).get("weights")
        if isinstance(weights, dict):
            lines.extend(
                f"  {ticker}: {weight:.4f}"
                for ticker, weight in sorted(weights.items())
            )
    return "\n".join(lines) if lines else "(gate payload empty)"
