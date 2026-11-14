"""The Audit view: the filtered audit trail and its CSV export."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Label, Static

from fund import observe
from fund.tui._views.shell import ShellView, _keep_cursor, _short

# Rows the Audit table shows; the export is not bound by it.
_AUDIT_VIEW_LIMIT = 200


class AuditView(ShellView):
    """The fund-wide, filterable decision timeline: agents ∪ humans.

    Reads ``observe.audit_timeline`` each tick into a table (when · pf · actor ·
    decision), with composable filters the shell toggles — actor side
    (:attr:`filter_kind`), ⚑ overrides (:attr:`filter_overrides`), and the
    header-focused portfolio (:attr:`filter_portfolio`). Selecting a row reveals the
    full provenance (prev value, source, idempotency key — immutable); ⚑ override / ⚠
    suitability rows are flagged in the decision cell. The table shows the newest
    ``_AUDIT_VIEW_LIMIT`` rows and says so in the filter line when older ones exist;
    :meth:`export_csv` re-reads and writes the whole filtered timeline to a file.

    Attributes:
        filter_kind: ``None`` (both) / ``"agent"`` / ``"human"`` — the actor filter.
        filter_overrides: Keep only ⚑ override rows when ``True``.
        filter_portfolio: Scope to the header-focused portfolio when ``True``.
        selected_entry_id: The selected record's stable ``AuditEntry.entry_id``, or
            ``None``. Selection is by identity, not list position: the timeline is
            newest-first and re-read each tick, so a position drifts onto another
            record as rows arrive.
        detail_text: Mirror of the rendered detail pane — a headless-test hook.
    """

    primary_focus: ClassVar[str | None] = "#audit-table"

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.filter_kind: str | None = None
        self.filter_overrides = False
        self.filter_portfolio = False
        self.detail_text = ""
        self.selected_entry_id: str | None = None
        self._entries: list[Any] = []
        self._truncated = False

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
        self.selected_entry_id = None
        self.refresh_view()

    def toggle_overrides(self) -> None:
        """Flip the ⚑ overrides-only filter, then repaint."""
        self.filter_overrides = not self.filter_overrides
        self.selected_entry_id = None
        self.refresh_view()

    def toggle_portfolio_scope(self) -> None:
        """Flip the header-focused-portfolio scope, then repaint."""
        self.filter_portfolio = not self.filter_portfolio
        self.selected_entry_id = None
        self.refresh_view()

    def select_index(self, index: int) -> None:
        """Select the row currently shown at ``index``, then repaint the detail pane.

        A driver/test hook: the position is resolved against the last-painted
        timeline and remembered as that record's identity.
        """
        in_range = 0 <= index < len(self._entries)
        self.selected_entry_id = self._entries[index].entry_id if in_range else None
        self.refresh_view()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A row selection reveals that decision's full provenance."""
        self.selected_entry_id = event.row_key.value
        self.refresh_view()

    def export_csv(self, path: str) -> str:
        """Write the full filtered timeline to ``path`` as CSV; return ``path``.

        One row per entry with the full provenance columns, under the same filters as
        the view but without its row cap. The timeline is re-read here rather than
        taken from the last paint, which may be capped or blank after a failed tick.
        JSON-valued cells are serialised deterministically.

        Raises:
            Exception: Any read error propagates before the file is opened, so a
                failed read never leaves a header-only file reported as exported.
        """
        import csv

        from fund.tui.exports import open_private

        entries = self._query(None)
        with open_private(path, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_AUDIT_CSV_HEADER)
            for entry in entries:
                writer.writerow(_audit_csv_row(entry))
        return path

    def _query(self, limit: int | None) -> list[Any]:
        """Read the filtered timeline in one short session; errors propagate."""
        with self.read_session() as session:
            pid = self.shell._portfolio_id if self.filter_portfolio else None
            return observe.audit_timeline(
                session,
                portfolio_id=pid,
                kind=self.filter_kind,
                overrides_only=self.filter_overrides,
                limit=limit,
            )

    def _read(self) -> list[Any]:
        """Read the newest rows the table shows (``[]`` on any error).

        Asks for one row past the cap so a full page that happens to be the whole
        timeline is told apart from a truncated one.
        """
        try:
            entries = self._query(_AUDIT_VIEW_LIMIT + 1)
        except Exception:  # keep the poll loop alive on any read error
            self._truncated = False
            return []
        self._truncated = len(entries) > _AUDIT_VIEW_LIMIT
        return entries[:_AUDIT_VIEW_LIMIT]

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
                    Text(entry.actor),
                    Text(_audit_label(entry)),
                    key=entry.entry_id,
                )
        if self._selected_entry() is None:
            self.selected_entry_id = None

    def _selected_entry(self) -> Any | None:
        """The selected record from the freshly read timeline, or ``None`` if gone."""
        if self.selected_entry_id is None:
            return None
        return next(
            (e for e in self._entries if e.entry_id == self.selected_entry_id), None
        )

    def _paint_filters(self) -> None:
        parts = [f"actor {self.filter_kind or 'all'}"]
        if self.filter_overrides:
            parts.append("⚑ overrides")
        if self.filter_portfolio:
            focused = self.shell._portfolio_id is not None
            parts.append("this pf" if focused else "this pf (none focused)")
        count = f"{len(self._entries)} rows"
        if self._truncated:
            count += " · older omitted (e exports all)"
        parts.append(count)
        self.query_one("#audit-filters", Static).update(" · ".join(parts))

    def _paint_detail(self) -> None:
        entry = self._selected_entry()
        text = _audit_detail(entry) if entry is not None else "(select a decision)"
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
