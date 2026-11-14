"""The shell's header: the portfolio picker, mode and run-state, status line."""

from __future__ import annotations

import uuid

from textual.widgets import Select, Static

from fund import observe
from fund.tui._mixins.host import ShellHost
from fund.tui.safe_text import strip_control


class PickerMixin(ShellHost):
    """The header: the portfolio picker, the mode and run-state glyphs, the status.

    Mixed into :class:`~fund.tui.app.ShellApp`.
    """

    def _populate_picker(self) -> None:
        """Seed the header ``Select`` from the fund's known portfolios (T2 reader).

        The options are the ids :meth:`_picker_ids` resolves, recorded on
        :attr:`_portfolio_ids` (a test hook). The ``Select`` always ends up showing the
        focused portfolio (or blank, fund-wide). ``set_options`` resets the selection,
        so the whole re-seed runs with ``Select.Changed`` prevented: otherwise the
        reset to blank would be dispatched after this returns and, read as a pick,
        move the shell's focus to fund-wide and back.
        """
        self._apply_picker(self._picker_ids())

    def _sync_picker(self) -> None:
        """Re-seed the picker when the fund's portfolio set changed since the last seed.

        Portfolios appear after launch (the ``fund`` CLI, the scheduler, another
        session), and a mount-time read that failed leaves the picker empty. Re-seeding
        only on a change keeps an open dropdown stable between ticks.
        """
        ids = self._picker_ids()
        if ids != self._portfolio_ids:
            self._apply_picker(ids)

    def _picker_ids(self) -> list[uuid.UUID]:
        """The portfolio ids the picker should offer, focused portfolio included.

        A focused portfolio with no rows yet (a ``fund-tui <new-uuid>`` launch) is still
        offered, so the header can name it. On a read error the current options are
        kept rather than blanked.
        """
        listed = self._list_portfolios()
        ids = list(self._portfolio_ids) if listed is None else listed
        if self._portfolio_id is not None and self._portfolio_id not in ids:
            ids.append(self._portfolio_id)
        return ids

    def _apply_picker(self, ids: list[uuid.UUID]) -> None:
        """Replace the picker's options with ``ids`` and re-select the focus, silently.

        ``set_options`` resets the selection to blank; with the message prevented that
        reset (and the re-selection) is never read as an adviser pick.
        """
        self._portfolio_ids = ids
        select = self.query_one("#hdr-portfolio", Select)
        with select.prevent(Select.Changed):
            select.set_options((self._short_label(pid), str(pid)) for pid in ids)
            if self._portfolio_id in ids:
                select.value = str(self._portfolio_id)

    def _list_portfolios(self) -> list[uuid.UUID] | None:
        """Distinct portfolio ids for the picker; ``None`` on any read error."""
        try:
            with self._session_factory() as session:
                ids = observe.list_portfolios(session)
                session.rollback()
                return ids
        except Exception:  # keep the shell alive on a read error (mirrors the cockpit)
            return None

    def _refresh_runstate(self) -> None:
        """Repaint the header run-state + autonomy-mode glyphs for the focused scope."""
        self._runstate = self._run_state_glyph()
        self.query_one("#hdr-runstate", Static).update(self._runstate)
        self.query_one("#hdr-mode", Static).update(self._mode_glyph())

    def _mode_glyph(self) -> str:
        """The autonomy posture glyph: ``❄ FROZEN`` when frozen, else ``● ARMED``.

        Fund scope wins: a FROZEN fund reads frozen regardless of the focused
        portfolio's own row. A read error reads ``? UNKNOWN`` — never the permissive
        ``● ARMED``, which would tell the adviser a frozen fund had resumed. Portfolios
        frozen *other than* the focused one are counted after the glyph (``· ❄ 1 other
        frozen``), since a freeze from the fund-wide Decisions queue can land on a
        portfolio the header is not focused on and would otherwise show nowhere.
        """
        from fund.audit.autonomy_repository import AutonomyRepository, AutonomyState

        frozen_value = AutonomyState.FROZEN.value
        focus = self._portfolio_id
        try:
            with self._session_factory() as session:
                repo = AutonomyRepository(session)  # read-only: never creates rows
                frozen = repo.fund_posture() == frozen_value
                others: list[uuid.UUID] = []
                if not frozen:
                    frozen = (
                        focus is not None
                        and repo.portfolio_posture(focus) == frozen_value
                    )
                    others = [p for p in repo.frozen_portfolio_ids() if p != focus]
                session.rollback()
        except Exception:  # keep the shell alive on a read error
            return "? UNKNOWN"
        glyph = "❄ FROZEN" if frozen else "● ARMED"
        if others:
            qualifier = "other " if focus is not None else ""
            glyph += f" · ❄ {len(others)} {qualifier}frozen"
        return glyph

    def _run_state_glyph(self) -> str:
        """The focused portfolio's latest-run glyph: live / running / awaiting / idle.

        A streaming round latches ``▶ LIVE`` (poll-stable). Otherwise: fund-wide (no
        focus) or no runs read as ``idle`` and a read error as ``? unknown``; a
        ``paused`` or ``responded`` (barge-in re-gate) latest run is ``awaiting`` the
        adviser, ``pending``/``running`` is live, everything else (completed /
        rejected) is idle.
        """
        if self._streaming:
            return "▶ LIVE"
        if self._portfolio_id is None:
            return "✓ idle"
        try:
            with self._session_factory() as session:
                runs = observe.list_portfolio_runs(session, self._portfolio_id, limit=1)
                session.rollback()
        except Exception:
            return "? unknown"
        if not runs:
            return "✓ idle"
        status = runs[0].status
        if status in ("paused", "responded"):
            return "⏸ awaiting"
        if status in ("pending", "running"):
            return "▶ running"
        return "✓ idle"

    def _short_label(self, portfolio_id: uuid.UUID) -> str:
        """A compact ``⟨abcd1234⟩`` picker label (no portfolio names persisted yet)."""
        return f"⟨{str(portfolio_id)[:8]}⟩"

    def _set_status(self, message: str) -> None:
        message = strip_control(message)
        self.last_status = message
        self.query_one("#status-bar", Static).update(message)
