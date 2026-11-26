"""Audit actions of the shell: the trail filters and the CSV export."""

from __future__ import annotations

import datetime as dt

from textual.css.query import NoMatches
from textual.widgets import ContentSwitcher

from fund.tui._mixins.host import ShellHost
from fund.tui.exports import export_path
from fund.tui.views import AuditView, ShellView


class AuditMixin(ShellHost):
    """Audit actions: the actor, overrides and scope filters and the CSV export.

    Mixed into :class:`~fund.tui.app.ShellApp`.
    """

    def action_audit_cycle_actor(self) -> None:
        """Cycle the Audit actor filter all → agent → human (the ``t`` binding)."""
        view = self._active_audit_view()
        if view is not None:
            view.cycle_kind()

    def action_audit_toggle_overrides(self) -> None:
        """Toggle the Audit ⚑ overrides-only filter (the ``f`` binding)."""
        view = self._active_audit_view()
        if view is not None:
            view.toggle_overrides()

    def action_audit_toggle_scope(self) -> None:
        """Toggle Audit's scope to the header-focused portfolio (the ``g`` binding)."""
        view = self._active_audit_view()
        if view is not None:
            view.toggle_portfolio_scope()

    def action_export_audit(self) -> None:
        """Export the Audit timeline to a timestamped CSV (the ``e`` binding).

        A silent no-op off the Audit pane (so ``e`` stays free for other views). The
        full filtered timeline is written; the path is reported on the status bar, or
        ``export failed`` when the read or the write fails.
        """
        view = self._active_audit_view()
        if view is None:
            return
        name = f"audit-export-{dt.datetime.now():%Y%m%d-%H%M%S}.csv"
        try:
            written = view.export_csv(str(export_path(name)))
        except Exception as exc:  # a read or write failure surfaces, never crashes
            self._set_status(f"export failed: {exc}")
            return
        self._set_status(f"exported audit to {written}")

    def _active_audit_view(self) -> AuditView | None:
        """The Audit view when it is the active content pane, else ``None``."""
        try:
            if self.query_one(ContentSwitcher).current != "view-audit":
                return None
            view = self.query_one("#view-audit", ShellView)
        except NoMatches:
            return None
        return view if isinstance(view, AuditView) else None
