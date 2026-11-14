"""Decisions actions of the shell: resolve, freeze, re-arm, edit and steer a gate."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from textual import work
from textual.css.query import NoMatches
from textual.widgets import ContentSwitcher

from fund import observe
from fund.tui._mixins.flight import _single_flight
from fund.tui._mixins.host import ShellHost
from fund.tui.barge_in import BargeInResult, BargeInScreen
from fund.tui.screens import (
    ConfirmScreen,
    OrderEditScreen,
    RationaleConfirmScreen,
    ScopePickScreen,
)
from fund.tui.views import DecisionsView, ShellView

# Records keep the shell's logger name, which operators and the tests filter on.
logger = logging.getLogger("fund.tui.app")


class DecisionsMixin(ShellHost):
    """Decisions actions: approve, reject, killswitch, re-arm, edit and steer a gate.

    The gate-mutating work runs on worker threads behind ``_single_flight``.
    Mixed into :class:`~fund.tui.app.ShellApp`.
    """

    def action_decide_approve(self) -> None:
        """Approve the selected Decisions gate (the ``a`` binding)."""
        self._decide("approve")

    def action_decide_reject(self) -> None:
        """Reject the selected Decisions gate (the ``r`` binding)."""
        self._decide("reject")

    def _decide(self, decision: str) -> None:
        """Dispatch ``decision`` for the Decisions view's selected gate on a worker.

        A no-op with a status message unless the Decisions view is active, a paused
        run is selected, and that gate's detail rendered — the same guard the cockpit
        uses for a blind approve.
        """
        view = self._selected_gate_view("select a paused run first")
        if view is None or view.selected_run_id is None:
            return
        # Approve/reject act at once with no confirm, so the adviser must have the
        # orders on screen: a failed gate render fails closed rather than releasing a
        # gate they could not read.
        if view.gate_unavailable:
            self._set_status(
                f"cannot {decision}: gate details unavailable - "
                "retry once the gate renders again"
            )
            return
        self.last_worker = self._action_worker(
            view.selected_run_id,
            view.selected_portfolio_id,
            decision,
            view.rendered_gate_id,
        )

    def action_decide_killswitch(self) -> None:
        """Killswitch the selected Decisions gate at a chosen scope (the ``k`` binding).

        Pushes a scope picker (no undo): ``run`` finalises the paused run REJECTED (the
        original killswitch, logged distinctly); ``portfolio`` / ``fund`` FREEZE the
        autonomy at that scope (``apply_action("freeze")``), halting live runs in scope
        until an explicit re-arm. Cancelling is a no-op. Inert — with a status hint —
        unless Decisions is active and a run is selected (approve/reject's guard).
        """
        view = self._selected_gate_view("select a paused run first")
        if view is None or view.selected_run_id is None:
            return
        run_id = view.selected_run_id
        portfolio_id = view.selected_portfolio_id
        prompt = (
            "KILLSWITCH — pick a scope (no undo):\n"
            "Run: finalise this paused run REJECTED.\n"
            "Portfolio: FREEZE this portfolio's autonomy (halts its runs).\n"
            "Fund: FREEZE the whole fund (halts every portfolio)."
        )
        scopes = [("Run", "run"), ("Portfolio", "portfolio"), ("Fund", "fund")]

        def _resolved(scope: str | None) -> None:
            if scope is None:
                self._set_status("killswitch cancelled")
            elif scope == "run":
                self.last_worker = self._action_worker(
                    run_id, portfolio_id, "killswitch"
                )
            else:
                self.last_worker = self._freeze_worker(scope, portfolio_id)

        self.push_screen(ScopePickScreen(prompt, scopes), _resolved)

    def action_rearm(self) -> None:
        """Re-arm autonomy at a chosen scope (the ``K`` binding).

        Pushes a scope picker: ``portfolio`` re-arms the focused portfolio behind a
        plain confirm; ``fund`` re-arms the whole fund behind a rationale-gated confirm
        (``apply_action("rearm")`` requires a non-empty rationale for fund scope).
        """
        prompt = (
            "RE-ARM autonomy — pick a scope:\n"
            "Portfolio: re-arm the focused portfolio.\n"
            "Fund: re-arm the whole fund (rationale required)."
        )
        scopes = [("Portfolio", "portfolio"), ("Fund", "fund")]

        def _resolved(scope: str | None) -> None:
            if scope is None:
                self._set_status("re-arm cancelled")
            elif scope == "portfolio":
                self._rearm_portfolio()
            else:
                self._rearm_fund()

        self.push_screen(ScopePickScreen(prompt, scopes), _resolved)

    def _rearm_portfolio(self) -> None:
        """Confirm, then re-arm the focused portfolio (no rationale required)."""
        pid = self._portfolio_id
        if pid is None:
            self._set_status("focus a portfolio to re-arm")
            return

        def _confirmed(ok: bool | None) -> None:
            if ok:
                self.last_worker = self._rearm_worker("portfolio", pid, None)
            else:
                self._set_status("re-arm cancelled")

        self.push_screen(
            ConfirmScreen(f"Re-arm portfolio {pid}? Autonomy resumes."), _confirmed
        )

    def _rearm_fund(self) -> None:
        """Re-arm the whole fund behind the rationale-gated confirm ceremony."""

        def _confirmed(rationale: str | None) -> None:
            if rationale is None:
                self._set_status("re-arm cancelled")
                return
            self.last_worker = self._rearm_worker("fund", None, rationale)

        self.push_screen(
            RationaleConfirmScreen("Re-arm the whole fund? Rationale required."),
            _confirmed,
        )

    def _active_decisions_view(self) -> DecisionsView | None:
        """The Decisions view when it is the active content pane, else ``None``."""
        try:
            if self.query_one(ContentSwitcher).current != "view-decisions":
                return None
            view = self.query_one("#view-decisions", ShellView)
        except NoMatches:
            return None
        return view if isinstance(view, DecisionsView) else None

    def _selected_gate_view(self, no_selection_hint: str) -> DecisionsView | None:
        """The Decisions view when its selected gate is also the highlighted row.

        Returns ``None`` with a status message when Decisions is not active, nothing is
        selected (``no_selection_hint``), or the cursor has moved off the selected gate.
        Selection only follows Enter or a click, so acting then would resolve a gate
        other than the one the adviser is looking at — they must press Enter on it
        first. These actions release orders or kill a run, hence the explicit guard.
        """
        view = self._active_decisions_view()
        if view is None or view.selected_run_id is None:
            self._set_status(no_selection_hint)
            return None
        highlighted = view.highlighted_run_id()
        if highlighted is not None and highlighted != view.selected_run_id:
            self._set_status(
                f"highlighted gate #{str(highlighted)[:8]} is not the selected gate "
                f"#{str(view.selected_run_id)[:8]} - press Enter on it first"
            )
            return None
        return view

    def action_edit_decision(self) -> None:
        """Edit the selected gate's proposed weights before committing (``e`` binding).

        View-scoped to Decisions (``DecisionsView.BINDINGS``) so it does not collide
        with the global Audit export ``e``. Reads the gate's proposed weights, opens the
        editor, and on Commit threads the edit through ``apply_action("edit")``. A no-op
        with a hint off Decisions or without a selected paused gate.
        """
        view = self._selected_gate_view("select a paused gate to edit")
        if view is None or view.selected_run_id is None:
            return
        run_id = view.selected_run_id
        portfolio_id = view.selected_portfolio_id
        shown_gate = view.rendered_gate_id
        weights = self._proposed_weights(run_id)
        if not weights:
            self._set_status("no proposed weights to edit")
            return

        def _edited(edited: dict[str, float] | None) -> None:
            if edited is None:
                self._set_status("edit cancelled")
                return
            self.last_worker = self._edit_worker(
                run_id, portfolio_id, edited, shown_gate
            )

        self.push_screen(OrderEditScreen(weights), _edited)

    def action_steer_decision(self) -> None:
        """Steer the PM on the selected gate with a directive (the ``s`` binding).

        View-scoped to Decisions; reuses the barge-in overlay and threads the directive
        through ``apply_action("respond")`` so the run re-gates ``responded``. A no-op
        with a hint off Decisions or without a selected paused gate.
        """
        view = self._selected_gate_view("select a paused gate to steer")
        if view is None or view.selected_run_id is None:
            return
        run_id = view.selected_run_id
        portfolio_id = view.selected_portfolio_id  # the gate's own portfolio

        def _resolved(result: BargeInResult | None) -> None:
            if result is None:
                self._set_status("steer cancelled")
                return
            self.last_worker = self._barge_in_worker(
                run_id, portfolio_id, result.text, result.remember
            )

        self.push_screen(BargeInScreen(_gate_target(run_id, portfolio_id)), _resolved)

    def _proposed_weights(self, run_id: uuid.UUID) -> dict[str, float]:
        """The weights to seed the editor with, or ``{}`` on any read error.

        Needs a pending gate (its ``place_orders`` interrupt). The seed is the run's
        audited optimizer output at full precision — the vector approving actually
        places — never the PM's own ``place_orders`` args, which that tool ignores, so
        the editor starts from what approval would place. The run repo is imported from
        its submodule (never the ``fund.audit`` root, which pulls the langgraph
        bootstrap) so this stays agent-stack-light.
        """
        from fund.audit.repository import AgentRunRepository

        try:
            with self._session_factory() as session:
                repo = AgentRunRepository(session)
                run = repo.get_run(run_id)
                interrupt = (
                    observe.interrupt_for(self._persistence.saver, run) if run else None
                )
                weights = repo.latest_optimizer_weights(run_id) if interrupt else {}
                session.rollback()
        except Exception:  # keep the shell alive on a read error
            return {}
        return {str(ticker): float(weight) for ticker, weight in weights.items()}

    @work(thread=True, group="action")
    @_single_flight("resolving gate")
    def _action_worker(
        self,
        run_id: uuid.UUID,
        portfolio_id: uuid.UUID | None,
        decision: str,
        shown_gate: str | None = None,
    ) -> None:
        """Resolve a gate through the unified action path — off the event loop.

        ``shown_gate`` is the id of the gate the adviser was looking at; the decision
        is refused when the run is parked at another one (see
        :func:`fund.actions.resolution_key`).

        The model is built lazily (a missing key surfaces as a status line, not a
        crash) and only for decisions that drive the graph: reject / killswitch settle
        in the DB, so a down provider never blocks the emergency stop.
        ``apply_action`` resumes + logs on a freshly opened session, the caller commits,
        and the UI touches marshal back via ``call_from_thread``.
        """
        from fund.actions import MODEL_FREE_ACTIONS, apply_action, resolution_key

        model = None
        if decision not in MODEL_FREE_ACTIONS:
            try:
                model = self._model_factory()
            except Exception as exc:  # surface a clean message, never crash the App
                self.call_from_thread(self._set_status, f"model unavailable: {exc}")
                return
        try:
            with self._session_factory() as session:
                result = apply_action(
                    decision,
                    session=session,
                    run_id=run_id,
                    portfolio_id=portfolio_id,
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    model=model,
                    fallback=self._build_fallback() if model is not None else None,
                    idempotency_key=resolution_key(
                        session,
                        self._persistence.saver,
                        run_id,
                        decision,
                        expected_gate=shown_gate,
                    ),
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    @work(thread=True, group="action")
    @_single_flight("committing edit")
    def _edit_worker(
        self,
        run_id: uuid.UUID,
        portfolio_id: uuid.UUID | None,
        weights: dict[str, float],
        shown_gate: str | None = None,
    ) -> None:
        """Commit adviser-edited weights through the unified path — off the event loop.

        ``shown_gate`` is the gate the editor was opened on; the edit is refused when
        the run has since moved to another one.

        Dispatches ``apply_action("edit", new_value=weights)``: the edit validates the
        risk invariants, appends the override as a fresh allocator decision, and resumes
        the run via ``approve`` so every downstream record reflects the edit. A risk
        violation or any failure surfaces as a status line rather than crashing the App.
        """
        from fund.actions import apply_action, resolution_key

        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"model unavailable: {exc}")
            return
        try:
            with self._session_factory() as session:
                result = apply_action(
                    "edit",
                    session=session,
                    run_id=run_id,
                    portfolio_id=portfolio_id,
                    new_value=weights,
                    model=model,
                    fallback=self._build_fallback(),
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    idempotency_key=resolution_key(
                        session,
                        self._persistence.saver,
                        run_id,
                        "edit",
                        expected_gate=shown_gate,
                    ),
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    @work(thread=True, group="action")
    def _freeze_worker(self, scope: str, portfolio_id: uuid.UUID | None) -> None:
        """Freeze autonomy at ``portfolio`` / ``fund`` scope — off the event loop.

        Dispatches ``apply_action("freeze", scope=…)``: the autonomy row flips FROZEN
        and every live run in scope is rejected (DB-level). A ``fund`` freeze ignores
        ``portfolio_id`` (the single fund row). Any failure surfaces as a status line.

        Deliberately outside :func:`_single_flight`: the freeze is the adviser's
        emergency stop, so it is never refused behind an in-flight approve / edit /
        steer — a graph resume that hangs would otherwise leave every freeze refused
        until the TUI restarts. It is safe to run alongside one because a freeze never
        resumes the graph, the hazard the lock exists for (the run-scope killswitch does
        resume it, via :meth:`_action_worker`, and stays guarded). It cannot cancel a
        resume that is already running, and an approve in flight may still finalise a
        run the freeze just rejected; the FROZEN autonomy row is what keeps further
        runs halted.
        """
        from fund.actions import apply_action

        self.call_from_thread(self._set_status, "freezing autonomy…")
        try:
            with self._session_factory() as session:
                result = apply_action(
                    "freeze",
                    session=session,
                    scope=scope,
                    portfolio_id=portfolio_id,
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    @work(thread=True, group="action")
    @_single_flight("re-arming autonomy")
    def _rearm_worker(
        self, scope: str, portfolio_id: uuid.UUID | None, rationale: str | None
    ) -> None:
        """Re-arm autonomy at ``portfolio`` / ``fund`` scope — off the event loop.

        Dispatches ``apply_action("rearm", scope=…)``: the autonomy row flips back ARMED
        (clearing the frozen provenance). A ``fund`` re-arm requires ``rationale`` (the
        action raises otherwise). Any failure surfaces as a status line.
        """
        from fund.actions import apply_action

        try:
            with self._session_factory() as session:
                result = apply_action(
                    "rearm",
                    session=session,
                    scope=scope,
                    portfolio_id=portfolio_id,
                    rationale=rationale,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    def _build_fallback(self) -> Any | None:
        """Build the retry model, or ``None`` when none is configured or it fails.

        The fallback only adds resilience, so a provider problem with it must not
        stop an action whose primary model built fine.
        """
        if self._fallback_factory is None:
            return None
        try:
            return self._fallback_factory()
        except Exception as exc:
            logger.warning("no fallback model for this action: %s", exc)
            return None

    def _after_action(self, message: str) -> None:
        """On the main thread once a worker committed: report + repaint view/header."""
        self._set_status(message)
        self._refresh()
        self._refresh_runstate()


def _gate_target(run_id: uuid.UUID | None, portfolio_id: uuid.UUID | None) -> str:
    """Name a gate the way every prompt does: its run id and its portfolio."""
    if run_id is None:
        return "the selected gate"
    target = f"gate #{str(run_id)[:8]}"
    if portfolio_id is not None:
        target += f" (portfolio {str(portfolio_id)[:8]})"
    return target
