"""Onboarding and drive flows of the shell: profile, run, stream, barge in."""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from textual import work
from textual.css.query import NoMatches
from textual.widgets import ContentSwitcher, Static

from fund import observe
from fund.tui._mixins.flight import _ROUND_BUSY, _round_flight, _single_flight
from fund.tui._mixins.host import ShellHost
from fund.tui.barge_in import BargeInResult, BargeInScreen
from fund.tui.questionnaire import QuestionnaireScreen, describe_error
from fund.tui.views import DeliberationView, ShellView

if TYPE_CHECKING:
    from textual.worker import Worker

    from fund.tui.questionnaire import OnboardingSubmission

# Records keep the shell's logger name, which operators and the tests filter on.
logger = logging.getLogger("fund.tui.app")


@dataclass(frozen=True)
class _QueuedDirective:
    """A barge-in directive held back until its run reaches the committed gate."""

    run_id: uuid.UUID
    portfolio_id: uuid.UUID | None
    text: str
    remember: bool


# How often a barge-in-held stream worker re-checks whether the app is quitting.
_HOLD_POLL_SECONDS = 0.25


class DriveMixin(ShellHost):
    """Onboarding and drive flows: new portfolio, profile, run, stream and barge in.

    Rounds run on worker threads behind ``_round_flight``. Mixed into
    :class:`~fund.tui.app.ShellApp`.
    """

    def action_new_portfolio(self) -> None:
        """Onboard a new portfolio from Posture (the ``n`` binding).

        Opens the MiFID questionnaire; on submit it mints a fresh id, builds the
        portfolio's initial run-policy mandate, and drives ``new_portfolio`` + the
        profiler on a worker. A no-op with a status hint unless Posture is the active
        pane (the launchpad).
        """
        if not self._posture_active():
            self._set_status("open Posture to onboard a portfolio")
            return
        if self._round_in_flight():
            self._set_status(_ROUND_BUSY)
            return

        def _submitted(submission: OnboardingSubmission | None) -> None:
            if submission is None:
                self._set_status("onboarding cancelled")
                return
            new_id = uuid.uuid4()
            try:
                mandate = _initial_mandate(new_id, submission)
            except Exception as exc:  # pydantic ValidationError (e.g. bad capital)
                self._set_status(f"invalid onboarding: {describe_error(exc)}")
                return
            self.last_worker = self._onboard_worker(
                new_id,
                submission.answers.model_dump_json(),
                mandate,
                submission.answers.base_currency,
            )

        self.push_screen(QuestionnaireScreen(ask_capital=True), _submitted)

    def action_profile_portfolio(self) -> None:
        """(Re)profile the focused portfolio from Posture (the ``p`` binding).

        Opens the questionnaire for an existing portfolio (no new mandate); on submit
        it drives the profiler on a worker. A no-op with a hint unless Posture is
        active and a portfolio is focused.
        """
        if not self._posture_active():
            self._set_status("open Posture to profile a portfolio")
            return
        pid = self._portfolio_id
        if pid is None:
            self._set_status("focus a portfolio to profile")
            return
        if self._round_in_flight():
            self._set_status(_ROUND_BUSY)
            return

        def _submitted(submission: OnboardingSubmission | None) -> None:
            if submission is None:
                self._set_status("profiling cancelled")
                return
            self.last_worker = self._onboard_worker(
                pid,
                submission.answers.model_dump_json(),
                None,
                submission.answers.base_currency,
            )

        self.push_screen(QuestionnaireScreen(ask_capital=False), _submitted)

    def action_run_portfolio(self, asof: dt.date | None = None) -> None:
        """Drive one paper rebalance for the focused portfolio (the ``u`` binding).

        Routes on the active view: from **Deliberation** the adviser watches the
        agents think, so ``u`` launches the **live-streamed** run (``_stream_worker``);
        from **Posture** it keeps the existing **blocking** run (``_run_worker``). The
        ``u`` key leaves the bar unset so the run resolves the newest fillable one;
        callers may pass an explicit decision bar. A no-op with a hint off both
        launchpads or without a focused portfolio.
        """
        deliberation = self._deliberation_active()
        if not (deliberation or self._posture_active()):
            self._set_status("open Posture or Deliberation to run a portfolio")
            return
        pid = self._portfolio_id
        if pid is None:
            self._set_status("focus a portfolio to run")
            return
        view = None
        if deliberation:
            view = self._deliberation_view()
            if view is None:
                self._set_status("open Deliberation to run a portfolio")
                return
        worker = self._launch_run(pid, asof, view)
        if worker is not None:
            self.last_worker = worker

    def _launch_run(
        self,
        portfolio_id: uuid.UUID,
        asof: dt.date | None,
        view: DeliberationView | None = None,
    ) -> Worker[None] | None:
        """Start one rebalance — streamed into ``view``, else blocking — single-flight.

        A round is minutes of LLM work that draws nothing until its first step
        reports, so a repeat ``u`` (or an Ask "run") is easy to fire, and a second
        concurrent round would double the spend, open a competing gate and interleave
        both streams into one timeline. A launch while the previous run worker is still
        alive is refused with a status line instead. ``asof=None`` leaves the bar to
        the run, which resolves the newest fillable one. Returns the new worker, or
        ``None`` when refused.
        """
        if self._round_in_flight():
            self._set_status("a run is already in progress — wait for it to finish")
            return None
        self._set_status(f"run started for portfolio {self._short_label(portfolio_id)}")
        if view is not None:
            worker = self._stream_worker(portfolio_id, asof, view)
        else:
            worker = self._run_worker(portfolio_id, asof)
        return worker

    def _round_in_flight(self) -> bool:
        """Whether a drive or stream worker is still running a round.

        A cancelled Textual handle reads as finished while its thread keeps running, so
        a second launch would run next to the first: both threads feed one live table
        (a duplicate-row error aborts one) and each opens its own run and gate for the
        same bar. The round lock, held by the thread, is what proves it is still alive.
        """
        return self._round_lock.locked() or any(
            worker.group in ("drive", "stream") and not worker.is_finished
            for worker in self.workers
        )

    def _posture_active(self) -> bool:
        """Whether the Posture view is the active content pane (the drive launchpad)."""
        try:
            return self.query_one(ContentSwitcher).current == "view-posture"
        except NoMatches:
            return False

    def _deliberation_active(self) -> bool:
        """Whether Deliberation is the active content pane (the live-run launchpad)."""
        try:
            return self.query_one(ContentSwitcher).current == "view-deliberation"
        except NoMatches:
            return False

    def _deliberation_view(self) -> DeliberationView | None:
        """The Deliberation view when mounted, else ``None`` — the stream's sink."""
        try:
            view = self.query_one("#view-deliberation", ShellView)
        except NoMatches:
            return None
        return view if isinstance(view, DeliberationView) else None

    def action_barge_in(self) -> None:
        """Barge into the live round with a PM directive (the ``b`` binding).

        A no-op with a hint unless the Deliberation view is active and a live round is
        showing. Holds the stream at the next step boundary, pushes the directive
        overlay (showing where it paused), and on Send threads the directive to the PM
        at the gate via ``respond``; Cancel resumes the stream unchanged. Both paths
        re-arm the stream so the hold is never left standing. A directive sent while
        the round is still streaming is queued and delivered when the stream ends at
        its gate, since only then is the run committed and awaiting an answer.
        """
        view = self._deliberation_view()
        if view is not None:
            view.retire_spent_live_tail()  # a gate decided since the last poll tick
        if (
            not self._deliberation_active()
            or view is None
            or view.live_run_id is None
            or view.live_portfolio_id is None
        ):
            self._set_status("barge in from a live Deliberation round")
            return
        self._stream_resume.clear()  # hold a streaming worker at the next boundary
        run_id = view.live_run_id
        paused_at = view.held_step_label()

        # The streaming run's own portfolio, NOT self._portfolio_id — the header focus
        # may have moved since the stream started, which would otherwise mis-route the
        # directive (and a remembered directive) to the wrong portfolio's mandate.
        portfolio_id = view.live_portfolio_id

        def _resolved(result: BargeInResult | None) -> None:
            self._stream_resume.set()  # resume the stream regardless of the choice
            if result is None:
                self._set_status("barge-in cancelled")
                return
            if self._streaming:
                replaced = self._queued_directive is not None
                self._queued_directive = _QueuedDirective(
                    run_id, portfolio_id, result.text, result.remember
                )
                self._set_status(
                    "directive queued"
                    + (" (replaces the earlier one)" if replaced else "")
                    + " — sent when the run reaches its gate"
                )
                return
            self.last_worker = self._barge_in_worker(
                run_id, portfolio_id, result.text, result.remember
            )

        self.push_screen(BargeInScreen(paused_at), _resolved)

    def _deliver_queued_directive(
        self, gate_run_id: uuid.UUID | None, *, failure: str | None = None
    ) -> None:
        """Send the directive queued during a stream, now that the stream has ended.

        Runs on the main thread after the stream worker finished. ``gate_run_id`` is
        the run the stream left paused at its gate, or ``None`` when the round ended
        without one; a directive with no matching gate to answer is dropped with a
        status line rather than sent to a run that cannot take it. ``failure`` is the
        status the stream ended with when it errored: the drop notice is appended to it
        so the single status line keeps the reason the round died.
        """
        queued, self._queued_directive = self._queued_directive, None
        if queued is None:
            return
        if gate_run_id != queued.run_id:
            reason = failure or "run ended without a gate"
            self._set_status(f"{reason} — queued directive dropped")
            return
        self.last_worker = self._barge_in_worker(
            queued.run_id, queued.portfolio_id, queued.text, queued.remember
        )

    def action_expand_readout(self) -> None:
        """Recompute the selected step's live read-out (the Deliberation ``x`` binding).

        A no-op with a hint unless Deliberation is active with an allocator or risk step
        selected in replay. Spawns the recompute worker; the frontier/backtest (an
        allocator step) or verdict/sector exposures (a risk step) paint in on return.
        """
        view = self._deliberation_view()
        if not self._deliberation_active() or view is None:
            self._set_status("expand a read-out from Deliberation")
            return
        target = view.expand_target()
        if target is None:
            self._set_status("select an allocator or risk step to expand")
            return
        run_id, agent, key = target
        self._set_status("recomputing read-out…")
        self.last_worker = self._recompute_readout_worker(run_id, agent, key, view)

    @work(thread=True, exclusive=True, group="readout")
    def _recompute_readout_worker(
        self, run_id: uuid.UUID, agent: str, key: int | None, view: DeliberationView
    ) -> None:
        """Recompute a step's read-out off the loop, then paint it onto the view.

        The optimizer-backed compute module is imported lazily so the module import
        stays agent-stack-light. On a short read-only session it recomputes the matching
        read-out: for an allocator step the *selected* proposal (``key`` is its
        ``decision_index``) so the frontier/backtest match the weights shown for that
        step, even in a re-optimized run; for a risk step the run's authoritative
        (latest) allocation. ``view`` is resolved by the caller on the event loop and
        passed in — this thread never queries the DOM, touching the view only through
        ``call_from_thread``. The result marshals back that way, unless a newer expand
        has cancelled this worker, and replaces the ``recomputing`` status with a ready
        line. Any failure surfaces as a status line.
        """
        from textual.worker import get_current_worker

        from fund import readouts_compute
        from fund.audit.repository import AgentRunRepository

        try:
            with self._session_factory() as session:
                repo = AgentRunRepository(session)
                run = repo.get_run(run_id)
                if run is None:
                    self.call_from_thread(self._set_status, "run not found")
                    return
                if agent == "allocator":
                    decision = next(
                        (d for d in run.decisions if d.decision_index == key), None
                    )
                    weights = AgentRunRepository.decode_weights(decision)
                    constraint_set: Any = (
                        decision.constraint_set if decision is not None else None
                    )
                    mandate = (
                        observe.get_mandate(session, run.portfolio_id)
                        if run.portfolio_id is not None
                        else None
                    )
                    result: Any = readouts_compute.compute_results(
                        session,
                        run.asof,
                        list(weights),
                        constraint_set,
                        weights,
                        base_currency=(
                            mandate.base_currency if mandate is not None else None
                        ),
                        lookback_days=_recorded_train_size(run.optimizer_config),
                    )
                else:
                    weights = repo.latest_optimizer_weights(run_id)
                    # Risk evaluates the run's authoritative allocation; pair its bounds
                    # with the SAME newest allocator proposal those weights came from
                    # (decisions are decision_index ASC, so scan reversed — mirrors
                    # ``observe.run_metrics``).
                    constraint_set = next(
                        (
                            d.constraint_set
                            for d in reversed(run.decisions)
                            if d.agent == "allocator" and d.step == "optimize_portfolio"
                        ),
                        None,
                    )
                    mandate = (
                        observe.get_mandate(session, run.portfolio_id)
                        if run.portfolio_id is not None
                        else None
                    )
                    result = readouts_compute.compute_risk(
                        weights,
                        constraint_set,
                        None,
                        session=session,
                        eligible=readouts_compute.resolve_eligible(
                            session, mandate, run.asof, constraint_set
                        ),
                    )
                session.rollback()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"recompute failed: {exc}")
            return
        # A newer expand cancels this worker, but a thread cannot be force-stopped; skip
        # the paint so a slow stale result never clobbers the newer one in the view.
        if not get_current_worker().is_cancelled:
            self.call_from_thread(view.paint_readout, run_id, agent, key, result)
            self.call_from_thread(self._set_status, "read-out ready")

    @work(thread=True, group="drive")
    @_round_flight
    def _onboard_worker(
        self,
        portfolio_id: uuid.UUID,
        questionnaire_json: str,
        mandate: Any | None,
        base_currency: str,
    ) -> None:
        """Create (when ``mandate`` is given) + profile a portfolio — off the loop.

        Both mutations funnel through ``apply_action`` on one freshly opened session;
        the caller commits once. A missing model key or any failure surfaces as a
        status line rather than crashing the App.
        """
        from fund.actions import apply_action

        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"model unavailable: {exc}")
            return
        try:
            with self._session_factory() as session:
                if mandate is not None:
                    apply_action(
                        "new_portfolio",
                        session=session,
                        portfolio_id=portfolio_id,
                        mandate=mandate,
                    )
                result = apply_action(
                    "profile",
                    session=session,
                    portfolio_id=portfolio_id,
                    questionnaire=questionnaire_json,
                    model=model,
                    fallback=self._build_fallback(),
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    base_currency=base_currency,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_onboard, result.message)

    @work(thread=True, group="drive")
    @_round_flight
    def _run_worker(self, portfolio_id: uuid.UUID, asof: dt.date | None) -> None:
        """Drive one paper rebalance for ``portfolio_id`` to its gate — off the loop."""
        from fund.actions import apply_action

        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"model unavailable: {exc}")
            return
        try:
            with self._session_factory() as session:
                mandate = observe.get_mandate(session, portfolio_id)
                if mandate is None:
                    self.call_from_thread(
                        self._set_status, "no mandate — onboard the portfolio first"
                    )
                    return
                result = apply_action(
                    "run",
                    session=session,
                    portfolio_id=portfolio_id,
                    mandate=mandate,
                    asof=asof,
                    model=model,
                    fallback=self._build_fallback(),
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    @work(thread=True, group="stream")
    @_round_flight
    def _stream_worker(
        self, portfolio_id: uuid.UUID, asof: dt.date | None, view: DeliberationView
    ) -> None:
        """Stream one paper rebalance into the live Deliberation timeline.

        The live counterpart to :meth:`_run_worker`: it drives ``stream_fund`` and
        marshals each ordered ``StreamEvent`` onto the Deliberation view with
        ``call_from_thread`` (``start_live`` on the first, ``ingest_event`` per step).
        A ``paused`` gate leaves the timeline standing — the adviser just watched it
        build and acts on it in Decisions — while a no-ticket / error end falls back to
        the replay transcript. Like the blocking run it appends the human ``run`` audit
        row in the round's own transaction. ``view`` is resolved by the caller on the
        event loop and passed in — this thread never queries the DOM, touching the view
        only through ``call_from_thread``. The agent stack is imported lazily so the
        module import stays agent-stack-light; one stream per worker (``stream_fund`` is
        not thread-safe). Model/stream errors surface as a status line, never a crash.
        """
        from fund.actions import append_run_audit, halt_run_if_frozen
        from fund.agents.graph import stream_fund

        self._stream_resume.set()  # clear any stale barge-in hold from a prior round
        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"model unavailable: {exc}")
            return
        last_kind: str | None = None
        last_text = ""
        gate_run_id: uuid.UUID | None = None
        halted = False
        try:
            with self._session_factory() as session:
                mandate = observe.get_mandate(session, portfolio_id)
                if mandate is None:
                    self.call_from_thread(
                        self._set_status, "no mandate — onboard the portfolio first"
                    )
                    return
                for event in stream_fund(
                    model,
                    mandate,
                    portfolio_id=portfolio_id,
                    asof=(
                        asof
                        if asof is not None
                        else observe.default_decision_bar(session)
                    ),
                    session=session,
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    fallback=self._build_fallback(),
                ):
                    if last_kind is None:
                        self.call_from_thread(
                            view.start_live, event.run_id, portfolio_id
                        )
                        self.call_from_thread(self._enter_live_runstate)
                    self.call_from_thread(view.ingest_event, event)
                    last_kind = event.kind
                    last_text = event.text
                    gate_run_id = event.run_id
                    # Suspend the graph generator at this step boundary while a barge-in
                    # is open; the overlay re-sets the flag on resolve.
                    if not self._hold_at_boundary():
                        return  # quit / superseded: abandon the round uncommitted
                if last_kind == "paused" and gate_run_id is not None:
                    halted = halt_run_if_frozen(session, gate_run_id, portfolio_id)
                if gate_run_id is not None:
                    append_run_audit(
                        session, run_id=gate_run_id, portfolio_id=portfolio_id
                    )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            failure = f"action failed: {exc}"
            self.call_from_thread(self._set_status, failure)
            self.call_from_thread(view.end_live)
            self.call_from_thread(self._exit_live_runstate)
            self.call_from_thread(self._deliver_queued_directive, None, failure=failure)
            return
        if halted:
            self.call_from_thread(
                self._set_status, "round halted — autonomy was frozen while it ran"
            )
        if last_kind == "incomplete":
            # The replay view that follows has no trace of why the round produced no
            # ticket, and the status bar is the only place left to say so.
            self.call_from_thread(
                self._set_status, f"round ended with no ticket — {last_text}"
            )
        if last_kind != "paused" or halted:
            # A no-ticket / incomplete / halted end has nothing to act on — drop to
            # replay.
            self.call_from_thread(view.end_live)
        self.call_from_thread(self._exit_live_runstate)
        # A halted round left no live gate, so a directive queued for it has nothing to
        # answer; the halt reason becomes the drop notice.
        self.call_from_thread(
            self._deliver_queued_directive,
            gate_run_id if last_kind == "paused" and not halted else None,
            failure="round halted — autonomy was frozen while it ran"
            if halted
            else None,
        )

    def _hold_at_boundary(self) -> bool:
        """Block a stream worker while a barge-in holds it; ``False`` once cancelled.

        Only the overlay's resolve callback re-arms the stream, and that never runs when
        the app quits with the overlay open. A thread worker cannot be killed and
        executor threads are joined at interpreter exit, so an unbounded wait would
        leave the process alive after the UI is gone. Polling the worker's cancel flag
        (Textual sets it on quit) bounds the hold and lets the round be abandoned.
        """
        from textual.worker import get_current_worker

        worker = get_current_worker()
        while not self._stream_resume.wait(_HOLD_POLL_SECONDS):
            if worker.is_cancelled:
                return False
        return not worker.is_cancelled

    @work(thread=True, group="action")
    @_single_flight(
        "steering the PM",
        refused=lambda self, run_id, portfolio_id, directive, remember: (
            f"steer not sent: {directive}"
        ),
    )
    def _barge_in_worker(
        self,
        run_id: uuid.UUID,
        portfolio_id: uuid.UUID | None,
        directive: str,
        remember: bool,
    ) -> None:
        """Thread the adviser's directive to the PM at the gate (``respond``).

        Dispatches ``respond`` through ``apply_action`` for the paused run: the PM
        incorporates ``directive`` and re-gates ``responded`` (still awaiting approval),
        or finalises ``incomplete``. When ``remember`` is set, the directive is also
        upserted into the run portfolio's ``remembered_directives`` (a soft guardrail)
        in the same transaction; a one-shot directive seeds only this round. The agent
        stack is imported lazily; a missing model key or any failure surfaces as a
        status line rather than crashing the App.
        """
        from fund.actions import apply_action

        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"model unavailable: {exc}")
            return
        try:
            with self._session_factory() as session:
                result = apply_action(
                    "respond",
                    session=session,
                    run_id=run_id,
                    portfolio_id=portfolio_id,
                    pm_directive=directive,
                    model=model,
                    fallback=self._build_fallback(),
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        message = result.message
        if remember and portfolio_id is not None:
            # Its own transaction: the gate is already consumed and the respond row
            # committed, so a failure here must not roll the decision's audit back.
            try:
                self._remember_directive(portfolio_id, directive)
            except Exception as exc:  # the directive stands for this round regardless
                logger.warning("remembering a directive failed", exc_info=True)
                message += f" (directive not saved to the mandate: {exc})"
        self.call_from_thread(self._after_action, message)

    def _remember_directive(self, portfolio_id: uuid.UUID, directive: str) -> None:
        """Append ``directive`` to the portfolio mandate's ``remembered_directives``.

        Reads the mandate under the row lock and writes it back changing only the
        directives, so a concurrent edit of any other field is kept rather than
        overwritten with a stale copy. A no-op when the portfolio has no mandate or
        already carries the directive, so a repeated remember never duplicates it.
        Commits its own transaction; waits for a mandate save in flight.
        """
        from fund.actions import apply_action
        from fund.audit.mandate_repository import MandateRepository

        with self._mandate_lock, self._session_factory() as session:
            if MandateRepository(session).get(portfolio_id, for_update=True) is None:
                return
            mandate = observe.get_mandate(session, portfolio_id)
            if mandate is None or directive in mandate.remembered_directives:
                return
            updated = mandate.model_copy(
                update={
                    "remembered_directives": (
                        *mandate.remembered_directives,
                        directive,
                    )
                }
            )
            apply_action(
                "mandate_edit",
                session=session,
                portfolio_id=portfolio_id,
                mandate=updated,
                prev_value=mandate.model_dump(mode="json"),
            )
            session.commit()

    def _enter_live_runstate(self) -> None:
        """Latch the header run-state to ``▶ LIVE`` for the duration of a stream.

        Paints only the in-memory ``▶ LIVE`` glyph (``_run_state_glyph`` short-circuits
        on ``_streaming``, opening no session). Reading autonomy/run state here would
        open a session that collides with the stream worker's open transaction on the
        shared single-connection test engine; the mode glyph is repainted on exit.
        """
        self._streaming = True
        self._runstate = self._run_state_glyph()
        self.query_one("#hdr-runstate", Static).update(self._runstate)

    def _exit_live_runstate(self) -> None:
        """Release the ``▶ LIVE`` latch; the header reverts to the DB-derived glyphs.

        Called after the stream worker's session has closed, so the full
        :meth:`_refresh_runstate` (which opens read sessions) is safe here.
        """
        self._streaming = False
        self._refresh_runstate()

    def _after_onboard(self, message: str) -> None:
        """On the main thread once onboarding committed: refresh the picker + report."""
        self._populate_picker()
        self._after_action(message)


def _recorded_train_size(optimizer_config: dict[str, Any] | None) -> int | None:
    """The backtest training window a run recorded, or ``None`` if it recorded none.

    Runs opened before the key was renamed carry it as ``lookback_days``.
    """
    config = optimizer_config or {}
    return config.get("backtest_train_size", config.get("lookback_days"))


def _initial_mandate(portfolio_id: uuid.UUID, submission: OnboardingSubmission) -> Any:
    """Build a new portfolio's initial run-policy mandate from the onboarding form.

    Capital + base currency come from the questionnaire; the drift trigger and L1
    threshold take conservative defaults the adviser can tune later in Mandate.
    Raises ``pydantic.ValidationError`` on an out-of-contract value (e.g. a
    non-positive capital), which the caller surfaces as a status line.
    """
    from fund.schemas.mandate import PortfolioMandate, RunTriggers

    return PortfolioMandate(
        portfolio_id=str(portfolio_id),
        capital=submission.capital,
        base_currency=submission.answers.base_currency,
        drift_l1_threshold=0.1,
        triggers=RunTriggers(cron=True, drift=True),
    )
