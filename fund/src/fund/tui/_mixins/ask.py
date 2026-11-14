"""The shell's Ask flow: route a turn, confirm an action, keep the history."""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from textual import work
from textual.css.query import NoMatches
from textual.widgets import ContentSwitcher

from fund.tui import ask
from fund.tui._mixins.decisions import _gate_target
from fund.tui._mixins.host import ShellHost
from fund.tui.ask_history import AskHistory, scope_of
from fund.tui.barge_in import BargeInResult, BargeInScreen
from fund.tui.exports import export_path
from fund.tui.safe_text import describe_error as describe_ask_error
from fund.tui.screens import ConfirmScreen, RationaleConfirmScreen
from fund.tui.views import AskView, ShellView, _live_gate_id

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session
    from textual.worker import Worker

# Logs under ``fund.tui.app``, the shell's name that operators and tests filter on.
logger = logging.getLogger("fund.tui.app")


class AskMixin(ShellHost):
    """The Ask flow: route a turn to a query or an action, confirm it, keep the history.

    Answers run on worker threads; the saved transcript is read and written on
    the shell's ``_ask_io`` executor. Mixed into :class:`~fund.tui.app.ShellApp`.
    """

    def ask_submit(self, turn: str) -> None:
        """Handle a submitted Ask turn: resolve a pending action, else classify + route.

        A carried disambiguation (``view.pending_action``) is resolved first; otherwise
        the turn is classified — a ``query`` dispatches the summarize worker, an
        ``action`` routes through resolve → confirm → ``apply_action``. The view's seam
        call; nothing mutates without a deliberate confirm.
        """
        view = self._active_ask_view()
        if view is None:
            return
        # Action turns read the database and the checkpointer on the UI thread; an
        # unhandled error here would end the whole TUI and lose the conversation.
        try:
            if view.pending_action is not None:
                self._resolve_pending(view, turn)
            elif ask.classify(turn) == "query":
                self.ask_query(turn)
            else:
                self.ask_action(turn)
        except Exception as exc:  # surface a clean message, never crash the App
            self._set_status(f"ask failed: {exc}")
            view.append_log("ask", f"(ask failed: {exc})")

    def ask_query(self, turn: str) -> None:
        """Dispatch the Ask query-summarize worker for ``turn``.

        Resolves the active Ask view here, on the event loop, and hands it to the
        worker: the worker thread must never query the DOM itself — it touches the view
        only through ``call_from_thread``. The grounded context build + the lazy model
        invoke then run on the worker thread, so the event loop stays responsive while a
        query is answered.
        """
        view = self._active_ask_view()
        if view is None:
            return
        # The worker answers for the scope on screen *now*: the adviser may switch
        # portfolio before it finishes, and the answer must stay with its own question.
        focus = self._portfolio_id
        scope = scope_of(focus)
        history = view.history if view.history_scope == scope else None
        self._ask_pending += 1
        self.last_worker = self._ask_worker(turn, view, focus, scope, history)

    def ask_action(self, turn: str) -> None:
        """Route an Ask action turn through resolve → confirm → ``apply_action``.

        An unambiguous target is confirmed and dispatched through the shared action
        workers (never a new mutation path); an ambiguous target holds a
        ``pending_action`` on the view and asks a disambiguation follow-up instead of a
        confirm; an unresolved (dead-end) target just reports why. Nothing mutates
        without a deliberate confirm, and never on a guess.
        """
        view = self._active_ask_view()
        if view is None:
            return
        resolution = self._resolve_ask(turn)
        if resolution.status == "resolved" and resolution.action is not None:
            self._confirm_ask_action(view, resolution.action)
        elif resolution.status == "ambiguous":
            view.pending_action = ask.pending_action(turn, resolution)
            view.append_log("ask", resolution.prompt)
        else:
            view.append_log("ask", resolution.prompt)

    def _resolve_pending(self, view: AskView, turn: str) -> None:
        """Resolve a carried pending action against ``turn``, else process it fresh.

        A pending action is single-shot — cleared before this turn is handled whatever
        the outcome. When the follow-up names the target, it reaches the confirm; an
        unrelated follow-up (no match) is routed as a fresh turn, so the pending never
        lingers and never executes on a guess.
        """
        pending = view.pending_action
        view.pending_action = None
        if pending is None:
            return
        resolution = self._read_resolution(
            lambda session: ask.resolve_pending(
                pending, turn, session, self._persistence.saver, self._portfolio_id
            )
        )
        if resolution.status == "resolved" and resolution.action is not None:
            self._confirm_ask_action(view, resolution.action)
        elif ask.classify(turn) == "query":
            self.ask_query(turn)
        else:
            self.ask_action(turn)

    def _confirm_ask_action(self, view: AskView, action: ask.ResolvedAction) -> None:
        """Push the confirm matching ``action``; dispatch it only on a deliberate yes.

        Every Ask action — even approve/reject — goes behind a confirm (unlike the
        Decisions view's direct approve), so nothing mutates from a typed sentence
        without a second, explicit step. A **fund** re-arm takes the rationale-gated
        ceremony; a steer collects its directive in the barge-in overlay; the rest take
        a plain spelled-out-consequence confirm.
        """
        if action.verb == "rearm" and action.scope == "fund":
            self._ask_fund_rearm(view)
            return
        if action.verb == "respond":
            self._ask_steer(view, action)
            return

        shown_gate: str | None = None

        def _resolved(ok: bool | None) -> None:
            if ok:
                self.last_worker = self._dispatch_ask_action(
                    view, action, shown_gate=shown_gate
                )
            else:
                self._ask_cancelled(view, action.verb)

        prompt = _ask_confirm_prompt(action)
        if action.verb == "approve" and action.run_id is not None:
            # The id first, then the text it labels (see DecisionsView._read).
            shown_gate = self._shown_gate_id(action.run_id)
            gate = self._ask_approve_gate(action.run_id)
            if gate is None:
                reason = "cannot approve: gate details unavailable - retry shortly"
                view.append_log("ask", reason)
                self._set_status(reason)
                return
            prompt = f"{prompt}\n\n{gate}"
        self.push_screen(ConfirmScreen(prompt), _resolved)

    def _shown_gate_id(self, run_id: uuid.UUID) -> str | None:
        """The id of the gate ``run_id`` is parked at, or ``None`` on any read error."""
        try:
            with self._session_factory() as session:
                gate = _live_gate_id(session, self._persistence.saver, run_id)
                session.rollback()
        except Exception:
            logger.warning("ask approve: gate id unreadable", exc_info=True)
            return None
        return gate

    def _dispatch_ask_action(
        self,
        view: AskView,
        action: ask.ResolvedAction,
        shown_gate: str | None = None,
    ) -> Worker[None] | None:
        """Dispatch a confirmed ``action`` through the matching action worker.

        ``shown_gate`` is the gate whose orders the confirm displayed; an approve of a
        run that has moved to another gate since is refused.
        """
        view.append_log("ask", f"{action.verb} confirmed")
        verb = action.verb
        if verb in ("approve", "reject", "killswitch") and action.run_id is not None:
            return self._action_worker(
                action.run_id, action.portfolio_id, verb, shown_gate
            )
        if verb == "freeze" and action.scope is not None:
            return self._freeze_worker(action.scope, action.portfolio_id)
        if verb == "rearm" and action.scope is not None:
            return self._rearm_worker(action.scope, action.portfolio_id, None)
        if verb == "run" and action.portfolio_id is not None:
            worker = self._launch_run(action.portfolio_id, None)
            if worker is None:
                view.append_log("ask", "run refused — a run is already in progress")
            return worker
        return None

    def _ask_fund_rearm(self, view: AskView) -> None:
        """Re-arm the whole fund from Ask behind the rationale-gated ceremony."""

        def _confirmed(rationale: str | None) -> None:
            if rationale is None:
                self._ask_cancelled(view, "rearm")
                return
            view.append_log("ask", "rearm confirmed")
            self.last_worker = self._rearm_worker("fund", None, rationale)

        self.push_screen(
            RationaleConfirmScreen("Re-arm the whole fund? Rationale required."),
            _confirmed,
        )

    def _ask_steer(self, view: AskView, action: ask.ResolvedAction) -> None:
        """Steer the PM from Ask: collect the directive in the barge-in overlay."""

        def _resolved(result: BargeInResult | None) -> None:
            if result is None:
                self._ask_cancelled(view, "steer")
                return
            view.append_log("ask", "steer sent")
            self.last_worker = self._barge_in_worker(
                action.run_id, action.portfolio_id, result.text, result.remember
            )

        self.push_screen(
            BargeInScreen(_gate_target(action.run_id, action.portfolio_id)), _resolved
        )

    def _ask_cancelled(self, view: AskView, verb: str) -> None:
        """Report a cancelled Ask action on the log + status line (a no-op mutation)."""
        view.append_log("ask", f"{verb} cancelled")
        self._set_status(f"{verb} cancelled")

    def _resolve_ask(self, turn: str) -> ask.TargetResolution:
        """Resolve an action turn's target over a short read session (no execution)."""
        return self._read_resolution(
            lambda session: ask.resolve_target(
                turn, session, self._persistence.saver, self._portfolio_id
            )
        )

    def _read_resolution(
        self, resolve: Callable[[Session], ask.TargetResolution]
    ) -> ask.TargetResolution:
        """Run ``resolve`` over a short read session; a read error is a dead end.

        Resolution runs on the event loop inside the ``Input.Submitted`` handler, where
        an escaped exception exits the whole App. A failed gate read therefore comes
        back as an ``unresolved`` resolution (and a status line) instead, so the
        adviser sees why nothing happened and nothing is confirmed on an unread gate
        list.
        """
        try:
            with self._session_factory() as session:
                resolution = resolve(session)
                session.rollback()
        except Exception as exc:  # a read error must never exit the shell
            message = f"could not read gates: {exc}"
            self._set_status(message)
            return ask.TargetResolution("unresolved", None, (), message)
        return resolution

    def action_export_ask(self) -> None:
        """Export the Ask conversation to ``ask-<pf>-<timestamp>.md`` (``e``).

        Written to the export directory (see :mod:`fund.tui.exports`).

        View-scoped to Ask (``AskView.BINDINGS``) so it does not collide with the global
        Audit export ``e``. Re-export overwrites (idempotent); a write failure surfaces
        on the status line. A silent no-op off the Ask pane.
        """
        view = self._active_ask_view()
        if view is None:
            return
        pid = self._portfolio_id or "fund"
        name = f"ask-{pid}-{dt.datetime.now():%Y%m%d-%H%M%S}.md"
        try:
            written = view.export_markdown(str(export_path(name)))
        except OSError as exc:  # write failure → status bar, never crash
            self._set_status(f"export failed: {exc}")
            return
        self._set_status(f"exported ask to {written}")

    def _sync_ask_history(self, view: AskView) -> None:
        """Show the focused scope's saved conversation, loading it only on a change.

        Runs on every repaint of the Ask view, so a portfolio picked in the header
        switches the conversation at once: the screen shows a placeholder and the
        messages are read off the event loop (:meth:`_load_history`). A history that
        cannot be read leaves Ask working in memory, as before, says so on the status
        line, and is tried again after a pause.
        """
        scope = scope_of(self._portfolio_id)
        if view.history_scope == scope and (
            view.history is not None or time.monotonic() < self._ask_history_retry_at
        ):
            return
        store = getattr(self._persistence, "store", None)
        if store is None:
            view.bind_history(None, scope)
            return
        history = AskHistory(store, scope)
        view.begin_binding(history, scope)
        with suppress(RuntimeError):  # the executor is shut down while quitting
            self._ask_io.submit(self._load_history, view, history, scope)

    def _load_history(self, view: AskView, history: AskHistory, scope: str) -> None:
        """Read a scope's saved messages (history thread), then hand them over."""
        rows: list[tuple[str, str]] | None = None
        error: Exception | None = None
        try:
            rows = history.load()
        except Exception as exc:  # an unreadable history must not end the cockpit
            logger.warning("ask history unreadable", exc_info=True)
            error = exc
        with suppress(RuntimeError):  # the app may be gone by now
            self.call_from_thread(
                self._history_loaded, view, history, scope, rows, error
            )

    def _history_loaded(
        self,
        view: AskView,
        history: AskHistory,
        scope: str,
        rows: list[tuple[str, str]] | None,
        error: Exception | None,
    ) -> None:
        """Show the messages that were read, unless the adviser has moved on since."""
        if view.history is not history:
            return
        if rows is not None:
            self._ask_history_retry_at = 0.0
            view.finish_binding(rows)
            return
        view.bind_history(None, scope)
        self._ask_history_retry_at = time.monotonic() + _ASK_HISTORY_RETRY_SECONDS
        reason = describe_ask_error(error) if error else "unknown error"
        self._set_status(f"saved Ask history unavailable: {reason}")

    def ask_persist(self, history: AskHistory, speaker: str, text: str) -> None:
        """Save one Ask message off the event loop; a failure is logged and dropped."""
        with suppress(RuntimeError):  # the executor is shut down while quitting
            self._ask_io.submit(self._save_message, history, speaker, text)

    def _save_message(self, history: AskHistory, speaker: str, text: str) -> None:
        """Write one message (history thread)."""
        try:
            history.append(speaker, text)
        except Exception:
            logger.warning("saving an Ask message failed", exc_info=True)

    def flush_ask_history(self, timeout: float = 10.0) -> None:
        """Block until every queued history read, write and delete has finished.

        A test hook, and what a caller needs before reading the store directly.
        """
        self._ask_io.submit(lambda: None).result(timeout)

    def on_unmount(self) -> None:
        """Stop accepting history work; what is queued still finishes in its thread."""
        self._ask_io.shutdown(wait=False)

    def _ask_busy(self) -> bool:
        """Whether a question has been dispatched and has not finished.

        Counted when the question is dispatched, not when its worker takes the lock: the
        worker first builds the model, and a cancelled worker is reported finished while
        its agent thread is still writing.
        """
        return self._ask_pending > 0

    def _ask_done(self) -> None:
        """A dispatched question has ended, however it ended."""
        self._ask_pending = max(0, self._ask_pending - 1)

    def ask_turn_refused(self) -> bool:
        """Whether a typed turn must wait because the history is being deleted.

        Says why on the status line, so the adviser knows to retype it.
        """
        if self._ask_clearing:
            self._set_status("Ask history is being deleted - try again in a moment")
            return True
        return False

    def ask_clear(self) -> None:
        """Delete the focused scope's Ask conversation after a confirmation.

        Typed as ``/clear`` in the Ask box.

        The confirm names what goes: every saved message and the agent's memory of
        them. It defaults to Cancel, since the deletion cannot be undone. Refused while
        an answer is still being written, because the agent would save its memory again
        after the delete.
        """
        view = self._active_ask_view()
        if view is None or self._ask_clearing:
            return
        if self._ask_busy():
            self._set_status("wait for the answer before clearing the history")
            return
        scope = scope_of(self._portfolio_id)

        def _resolved(ok: bool | None) -> None:
            if ok and not self._ask_busy() and not self._ask_clearing:
                self._ask_clearing = True
                self.last_worker = self._ask_clear_worker(view, scope)

        label = "the whole fund" if scope == "fund" else f"portfolio {scope[:8]}"
        self.push_screen(
            ConfirmScreen(
                f"Delete the Ask conversation for {label}?\n\n"
                "Every saved question and answer is erased for good and Ask forgets "
                "them. This cannot be undone. Conversations you exported to a file "
                "with e are not touched."
            ),
            _resolved,
        )

    @work(thread=True, group="ask-clear")
    def _ask_clear_worker(self, view: AskView, scope: str) -> None:
        """Delete the saved messages, then the agent's memory and files, off the loop.

        The screen follows the saved messages: once they are deleted it is emptied, even
        if the agent's memory then fails to delete, and that failure is named on the
        status line. The screen is touched only if it still shows ``scope``.
        """
        from fund.agents.ask_agent import clear_ask_state

        def empty_screen() -> None:
            if view.history_scope == scope:
                view.clear_log()

        store = getattr(self._persistence, "store", None)
        messages_gone = False
        removed = 0
        try:
            with self._ask_lock:
                removed = (
                    self._ask_io.submit(AskHistory(store, scope).clear).result()
                    if store is not None
                    else 0
                )
                messages_gone = True
                clear_ask_state(self._persistence.saver, scope)
        except Exception as exc:  # report it; never claim a clear that did not happen
            logger.warning("clearing the Ask history failed", exc_info=True)
            if messages_gone:
                self.call_from_thread(empty_screen)
            self.call_from_thread(
                self._set_status,
                f"could not delete the history: {describe_ask_error(exc)}",
            )
            self.call_from_thread(self._ask_clear_done)
            return
        self.call_from_thread(empty_screen)
        self.call_from_thread(
            self._set_status, f"Ask history deleted ({removed} messages)"
        )
        self.call_from_thread(self._ask_clear_done)

    def _ask_clear_done(self) -> None:
        """The clear has ended; questions are accepted again."""
        self._ask_clearing = False

    def _active_ask_view(self) -> AskView | None:
        """The Ask view when it is the active content pane, else ``None``."""
        try:
            if self.query_one(ContentSwitcher).current != "view-ask":
                return None
            view = self.query_one("#view-ask", ShellView)
        except NoMatches:
            return None
        return view if isinstance(view, AskView) else None

    @work(thread=True, exclusive=True, group="ask")
    def _ask_worker(
        self,
        turn: str,
        view: AskView,
        focus: uuid.UUID | None,
        scope: str,
        history: AskHistory | None,
    ) -> None:
        """Answer one question with the read-only deep agent, off the event loop.

        Lazily builds the chat model via the injected factory and the agent over it,
        runs the question on this worker thread (each tool the agent calls is shown on
        the status line as it happens), and marshals the answer back via
        ``call_from_thread``. ``view`` is resolved by the caller on the event loop and
        passed in — this thread never queries the DOM, touching the view only through
        ``call_from_thread``. A missing/failed model degrades to a status line + a log
        note, never a crash. A newer question cancels this worker, but a thread cannot
        be force-stopped, so every paint is dropped once it is cancelled: a slow stale
        reply never lands under the newer question. The turns of one scope share one
        agent thread, so they run one at a time under ``_ask_lock``. ``scope`` and
        ``history`` are the conversation the question was asked in, captured when it was
        dispatched: if the adviser has moved to another portfolio by the time the answer
        arrives, it is saved to its own conversation and not shown in the new one. The
        agent stack is imported lazily so ``import fund.tui.app`` stays
        agent-stack-light.
        """
        from textual.worker import get_current_worker

        worker = get_current_worker()
        try:
            self._answer_in_scope(turn, view, focus, scope, history, worker)
        finally:
            with suppress(Exception):
                self.call_from_thread(self._ask_done)

    def _answer_in_scope(
        self,
        turn: str,
        view: AskView,
        focus: uuid.UUID | None,
        scope: str,
        history: AskHistory | None,
        worker: Any,
    ) -> None:
        """Run one question to its answer on the worker thread (see ``_ask_worker``)."""

        def deliver(paint: Callable[..., None], *args: Any) -> None:
            # Cancellation is checked on the event loop, where the newer question
            # cancels this worker, so no stale paint can slip in between the check and
            # the call.
            def guarded() -> None:
                if not worker.is_cancelled:
                    paint(*args)

            self.call_from_thread(guarded)

        def note(text: str) -> None:
            """Show an error note, never saving it as if it were an answer."""
            if view.history_scope == scope:
                view.append_log("ask", text, save=False)

        def answer_arrived(text: str) -> None:
            """Show the answer here, or keep it with the scope it belongs to."""
            if view.history_scope == scope:
                view.set_answer(text)
                return
            if history is not None:
                try:
                    history.append("ask", text)
                except Exception:
                    logger.warning("saving an Ask answer failed", exc_info=True)
            self._set_status("an answer for another portfolio was saved to its history")

        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            deliver(self._set_status, f"model unavailable: {describe_ask_error(exc)}")
            deliver(note, f"(model unavailable: {describe_ask_error(exc)})")
            return
        if model is None:
            deliver(self._set_status, "no model configured")
            deliver(note, "(no model configured)")
            return
        from fund.agents.ask_agent import build_ask_agent, run_ask_turn

        if self._ask_lock.locked():
            deliver(self._set_status, "ask: waiting for the previous answer…")
        with self._ask_lock:
            if worker.is_cancelled:
                return
            try:
                agent = build_ask_agent(
                    model=model,
                    session_factory=self._session_factory,
                    persistence=self._persistence,
                    scope=scope,
                    focus=focus,
                    config=self._config,
                )
                answer = run_ask_turn(
                    agent,
                    scope=scope,
                    question=turn,
                    on_step=lambda step: deliver(self._set_status, f"ask: {step}…"),
                    max_steps=self._config.ask_max_steps,
                    timeout_seconds=self._config.ask_timeout_seconds,
                )
            except Exception as exc:  # surface a clean message, never crash the App
                deliver(self._set_status, f"ask failed: {describe_ask_error(exc)}")
                deliver(note, f"(ask failed: {describe_ask_error(exc)})")
                return
        deliver(self._set_status, "ask: answered")
        deliver(answer_arrived, answer)


# Seconds before a failed load of the saved Ask history is tried again.
_ASK_HISTORY_RETRY_SECONDS = 30.0


# The spelled-out consequence per Ask action verb, for the confirm dialog.
_ASK_CONSEQUENCE: dict[str, str] = {
    "approve": "Approve and execute",
    "reject": "Reject (finalise)",
    "killswitch": "KILLSWITCH — finalise rejected (no undo)",
    "freeze": "FREEZE autonomy (halts runs)",
    "rearm": "Re-arm autonomy",
    "run": "Run a paper rebalance for",
}


# The verbs that act on one paused gate (a run); every other verb targets a scope or
# the focused portfolio, whatever ``run_id`` happens to hold.
_ASK_GATE_VERBS: frozenset[str] = frozenset({"approve", "reject", "killswitch"})


def _ask_confirm_prompt(action: Any) -> str:
    """The spelled-out confirm prompt for a resolved Ask ``action``."""
    verb = action.verb
    if action.scope == "fund":
        target = "the whole fund"
    elif verb in _ASK_GATE_VERBS and action.run_id is not None:
        target = _gate_target(action.run_id, action.portfolio_id)
    elif action.portfolio_id is not None:
        target = f"portfolio {str(action.portfolio_id)[:8]}"
    else:
        target = "this"
    return f"{_ASK_CONSEQUENCE.get(verb, verb)} {target}?"
