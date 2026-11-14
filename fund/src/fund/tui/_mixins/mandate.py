"""Mandate actions of the shell: save the run policy, override a guardrail."""

from __future__ import annotations

import uuid
from typing import Any

from textual import work
from textual.css.query import NoMatches
from textual.widgets import ContentSwitcher

from fund.tui._mixins.host import ShellHost
from fund.tui.questionnaire import describe_error
from fund.tui.views import MandateView, ShellView


class MandateMixin(ShellHost):
    """Mandate actions: save the run policy and override a guardrail.

    Both write on worker threads. Mixed into :class:`~fund.tui.app.ShellApp`.
    """

    def action_save_mandate(self) -> None:
        """Persist the Mandate view's run-policy edits (``ctrl+s``).

        A no-op with a status hint unless the Mandate view is active and has a current
        mandate. The proposed mandate is re-validated here (a bad edit surfaces as a
        status line); the upsert + audit happen on a worker thread.
        """
        view = self._active_mandate_view()
        if view is not None and view.mandate is None and view.read_error:
            self._set_status(f"mandate unreadable: {view.read_error}")
            return
        if view is None or view.mandate is None:
            self._set_status("focus a portfolio with a mandate first")
            return
        try:
            proposed = view.proposed_mandate()
        except Exception as exc:  # pydantic ValidationError on an out-of-contract edit
            self._set_status(f"invalid mandate: {describe_error(exc)}")
            return
        prev_value = view.mandate.model_dump(mode="json")
        self.last_worker = self._mandate_worker(
            self._portfolio_id, proposed, prev_value
        )

    def action_override_guardrail(self) -> None:
        """Log a guardrail override from the Mandate view (``o``), ⚠ if out of band.

        A no-op with a status hint unless the Mandate view is active, a target band is
        picked, and the portfolio has an active suitability band to check against. The
        band check + audit write happen on a worker thread (audit-only this phase).
        """
        view = self._active_mandate_view()
        if view is None or view.suitability_band is None:
            self._set_status("no active suitability band to check against")
            return
        proposed = view.proposed_band()
        if proposed is None:
            self._set_status("pick a target risk band first")
            return
        self.last_worker = self._guardrail_worker(
            self._portfolio_id, proposed, view.suitability_band
        )

    def _active_mandate_view(self) -> MandateView | None:
        """The Mandate view when it is the active content pane, else ``None``."""
        try:
            if self.query_one(ContentSwitcher).current != "view-mandate":
                return None
            view = self.query_one("#view-mandate", ShellView)
        except NoMatches:
            return None
        return view if isinstance(view, MandateView) else None

    @work(thread=True, group="mandate")
    def _mandate_worker(
        self, portfolio_id: uuid.UUID | None, mandate: Any, prev_value: Any
    ) -> None:
        """Upsert the run-policy mandate through the unified path, off the loop.

        ``prev_value`` is the mandate the edit was built from; the save is refused when
        the stored mandate has moved since (a remembered directive, the CLI), so a
        stale view never overwrites a change the adviser has not seen. A save already in
        flight refuses this one rather than racing it.
        """
        from fund.actions import apply_action

        if not self._mandate_lock.acquire(blocking=False):
            self.call_from_thread(
                self._set_status,
                "a mandate save is already in progress - wait for it to finish",
            )
            return
        try:
            with self._session_factory() as session:
                result = apply_action(
                    "mandate_edit",
                    session=session,
                    portfolio_id=portfolio_id,
                    mandate=mandate,
                    prev_value=prev_value,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        finally:
            self._mandate_lock.release()
        self.call_from_thread(
            self._after_mandate_saved, result.message, portfolio_id, mandate
        )

    def _after_mandate_saved(
        self, message: str, portfolio_id: uuid.UUID | None, saved: Any
    ) -> None:
        """On the main thread once a mandate save committed: release its edit guard.

        The Mandate view is looked up whether or not it is the active pane — the
        adviser may have switched away while the worker ran, and the guard would
        otherwise keep the saved fields frozen on return. A save for a portfolio that
        is no longer focused has nothing to release: the focus change already reset it.
        """
        if portfolio_id == self._portfolio_id:
            try:
                view = self.query_one("#view-mandate", ShellView)
            except NoMatches:
                view = None
            if isinstance(view, MandateView):
                view.mark_saved(saved)
        self._after_action(message)

    @work(thread=True, exclusive=True, group="mandate")
    def _guardrail_worker(
        self, portfolio_id: uuid.UUID | None, proposed_band: str, suitability_band: str
    ) -> None:
        """Log a guardrail override (⚑/⚠ when out of band) — off the event loop."""
        from fund.actions import apply_action, is_within_suitability_band

        out_of_band = not is_within_suitability_band(proposed_band, suitability_band)
        try:
            with self._session_factory() as session:
                result = apply_action(
                    "guardrail_override",
                    session=session,
                    portfolio_id=portfolio_id,
                    target="risk_band",
                    prev_value=suitability_band,
                    new_value=proposed_band,
                    override_flag=out_of_band,
                    suitability_flag=out_of_band,
                    source="mifid_profile",
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)
