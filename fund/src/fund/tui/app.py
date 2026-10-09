"""``fund-tui`` — the ``ShellApp`` multi-view oversight cockpit.

A six-view mission-control shell (Posture · Deliberation · Decisions · Mandate ·
Audit · Ask) that can land fund-wide or focused on one portfolio. Every data
surface is read from the model-free read model (``fund.observe``); the App owns
all side effects:

* **Polling, not pushing.** ``ShellApp._refresh`` runs on a ``set_interval``
  timer (``poll_interval``, ~2s), opening a *short read-only* session, extracting
  plain dataclasses, and rolling back — the UI never holds a transaction or an ORM
  row across ticks.
* **Mutations off the event loop.** Actions (approve/reject/killswitch/mandate/
  onboard/run) dispatch ``@work(thread=True)`` workers that call
  ``fund.actions.apply_action`` on a background thread, commit once, then marshal
  UI updates back with ``call_from_thread``.
* **Agent-stack-light import.** ``import fund.tui.app`` pulls only Textual + the
  agent-stack-free read model — no ``deepagents`` / ``langgraph`` / model at
  module scope. The chat model is built lazily via the injected ``model_factory``
  only when the adviser acts; the persistence handles (checkpointer + store) are
  built in ``main``.

Every collaborator is injected (``session_factory`` / ``persistence`` /
``model_factory``), so ``App.run_test`` can drive the whole cockpit headlessly
over SQLite + ``MemorySaver`` + a scripted model with zero network.
"""

from __future__ import annotations

import datetime as dt
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from fund import observe
from fund.config import FundConfig, settings
from fund.tui.questionnaire import QuestionnaireScreen
from fund.tui.views import (
    VIEW_SPECS,
    AuditView,
    DecisionsView,
    MandateView,
    ShellView,
    build_views,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType
    from textual.worker import Worker

    from fund.tui.questionnaire import OnboardingSubmission


class ConfirmScreen(ModalScreen[bool]):
    """A modal stating an irreversible consequence, awaiting a deliberate choice.

    Pushed before a destructive, undo-less action (the killswitch): it renders the
    spelled-out consequence over a Confirm | Cancel row and dismisses ``True`` only on
    an explicit Confirm, ``False`` on Cancel or ``escape`` — so nothing irreversible
    fires without a second, deliberate step.

    Args:
        prompt: The consequence text shown above the buttons.
    """

    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    .confirm-dialog {
        width: 60; height: auto; padding: 1 2;
        border: thick $error; background: $surface;
    }
    .confirm-prompt { height: auto; color: $warning; }
    .confirm-actions { height: auto; align-horizontal: center; padding-top: 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        """Yield the consequence prompt above a Confirm | Cancel button row."""
        with Vertical(classes="confirm-dialog"):
            yield Static(self._prompt, classes="confirm-prompt")
            with Horizontal(classes="confirm-actions"):
                yield Button("Confirm", variant="error", id="confirm-yes")
                yield Button("Cancel", variant="primary", id="confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Resolve the dialog: Confirm dismisses ``True``, else ``False``."""
        self.dismiss(event.button.id == "confirm-yes")

    def action_cancel(self) -> None:
        """``escape`` dismisses the dialog as a cancel — a no-op for the caller."""
        self.dismiss(False)


_NARROW_WIDTH = 80  # below this terminal width the sidebar collapses to a rail


class ShellApp(App[None]):
    """Mission-control shell: a sidebar of six views over one ContentSwitcher.

    The multi-view frame the redesign lands on — a left nav rail (Posture ·
    Deliberation · Decisions · Mandate · Audit · Ask) switching a single content
    pane, under a status header (focused portfolio · mode · run state) and a key
    hint bar. It lands **fund-wide** (``portfolio_id=None``) and can focus one
    portfolio; Phase-0 views are placeholders (:class:`~fund.tui.views.ShellView`),
    filled by later phases.

    Every collaborator is injected through ``__init__`` (so ``run_test`` drives it
    headlessly over SQLite + ``MemorySaver`` with zero network); the per-view
    ``_refresh`` is synchronous; and the import stays agent-stack-light (Textual +
    the model-free read model only — no ``deepagents`` / ``langgraph``).
    """

    CSS = """
    .shell-header { height: 1; background: $panel; color: $text; }
    .shell-header Static { width: auto; padding: 0 2; }
    .shell-body { height: 1fr; }
    .sidebar { width: 22; border-right: solid $accent; }
    .sidebar.narrow { width: 5; }
    .content { width: 1fr; padding: 0 1; }
    .view-title { text-style: bold; color: $accent; }
    .view-subtitle { color: $accent; }
    .status-bar { dock: bottom; height: 1; background: $panel; padding: 0 1; }
    .posture-body { height: 1fr; }
    .posture-roster { width: 45%; }
    .posture-detail { width: 1fr; padding: 0 1; color: $warning; }
    .posture-hint { height: 1; color: $accent; }
    .decisions-body { height: 1fr; }
    .decisions-queue { width: 45%; }
    .decisions-gate { width: 1fr; padding: 0 1; color: $warning; }
    .decisions-hint { height: 1; color: $accent; }
    .mandate-body { height: 1fr; }
    .mandate-policy { width: 50%; }
    .mandate-guardrails { width: 1fr; padding: 0 1; }
    .mandate-section { text-style: bold; color: $accent; }
    .mandate-readonly { height: auto; color: $warning; }
    .mandate-hint { height: 1; color: $accent; }
    .audit-body { height: 1fr; }
    .audit-table { width: 55%; }
    .audit-detail { width: 1fr; padding: 0 1; color: $warning; }
    .audit-hint { height: 1; color: $accent; }
    .delib-body { height: 1fr; }
    .delib-left { width: 50%; }
    .delib-runs { height: 40%; }
    .delib-steps { height: 1fr; }
    .delib-detail { width: 1fr; padding: 0 1; color: $warning; }
    .delib-hint { height: 1; color: $accent; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("a", "decide_approve", "Approve"),
        ("r", "decide_reject", "Reject"),
        ("k", "decide_killswitch", "Killswitch"),
        ("ctrl+s", "save_mandate", "Save mandate"),
        ("o", "override_guardrail", "Override guardrail"),
        ("n", "new_portfolio", "New portfolio"),
        ("p", "profile_portfolio", "Profile"),
        ("u", "run_portfolio", "Run now"),
        ("t", "audit_cycle_actor", "Audit actor filter"),
        ("f", "audit_toggle_overrides", "Audit overrides filter"),
        ("g", "audit_toggle_scope", "Audit portfolio scope"),
        ("e", "export_audit", "Export audit"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        portfolio_id: uuid.UUID | None = None,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]],
        persistence: Any,
        model_factory: Callable[[], Any],
        config: FundConfig = settings,
        poll_interval: float = 2.0,
    ) -> None:
        """Initialise the shell, optionally focused on one portfolio.

        Args:
            portfolio_id: Portfolio to focus, or ``None`` for the fund-wide
                landing (the launcher opens the shell before a pick is made).
            session_factory: Callable returning a context-manager-managed
                SQLAlchemy Session for read ticks and the portfolio picker.
            persistence: Langgraph persistence bundle exposing ``.saver`` and
                ``.store`` (used by the data-bearing views in later phases).
            model_factory: Called lazily when the adviser acts; unused by the
                Phase-0 placeholder views.
            config: Application settings; defaults to the process singleton.
            poll_interval: Refresh timer cadence in seconds.
        """
        super().__init__()
        self._portfolio_id = portfolio_id
        self._session_factory = session_factory
        self._persistence = persistence
        self._model_factory = model_factory
        self._config = config
        self._poll_interval = poll_interval
        self.last_status: str = ""
        self.last_worker: Worker[None] | None = None
        # ``_portfolio_ids`` / ``_runstate`` are headless-test hooks mirroring the
        # views' ``render_count``.
        self._portfolio_ids: list[uuid.UUID] = []
        self._runstate = "✓ idle"
        self._narrow = False
        # Notification bus (poll-diff): attention keys already alerted, so a
        # still-pending gate/breach never re-toasts on the next tick.
        self._seen_attention: set[tuple[str, uuid.UUID]] = set()
        self._awaiting_count = 0
        self.last_attention: list[str] = []  # headless-test hook (mirrors last_status)

    # --- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Yield the header, the sidebar | content split, the status bar, and footer."""
        yield Header()
        with Horizontal(id="shell-header", classes="shell-header"):
            yield Select([], prompt="⟨fund-wide⟩", id="hdr-portfolio", allow_blank=True)
            yield Static("● AUTONOMOUS", id="hdr-mode")
            yield Static(self._runstate, id="hdr-runstate")
            yield Static("⏸0 awaiting", id="hdr-attention")
        with Horizontal(id="shell-body", classes="shell-body"):
            yield ListView(
                *(ListItem(Label(title), id=f"nav-{key}") for key, title in VIEW_SPECS),
                id="sidebar",
                classes="sidebar",
            )
            with ContentSwitcher(
                initial="view-posture", id="content", classes="content"
            ):
                yield from build_views()
        yield Static("", id="status-bar", classes="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Seed the portfolio picker, paint the run-state glyph, and start polling.

        The mount-time ``Select.Changed`` the seeding echoes is dropped by
        ``on_select_changed``'s value-equality guard, not a mount-window flag.
        """
        # Seed the picker, then paint once the views have mounted and poll on the
        # cockpit's cadence (the placeholder views' refresh is a no-op counter bump
        # today).
        self._populate_picker()
        self._refresh_runstate()
        if self.size.width:
            self._apply_width(self.size.width)
        self.call_after_refresh(self._refresh)
        self.set_interval(self._poll_interval, self._refresh)

    def on_resize(self, event: Resize) -> None:
        """Collapse/restore the sidebar rail as the terminal crosses the threshold."""
        self._apply_width(event.size.width)

    def _apply_width(self, width: int) -> None:
        """Collapse the sidebar to a single-letter rail below the narrow threshold.

        At narrow widths the sidebar shrinks (``.narrow`` CSS) and each entry's
        label drops to its leading letter, leaving the content pane effectively
        single-pane; above the threshold the full titles and width return.
        """
        self._narrow = width < _NARROW_WIDTH
        try:
            sidebar = self.query_one("#sidebar", ListView)
        except NoMatches:
            return
        sidebar.set_class(self._narrow, "narrow")
        for key, title in VIEW_SPECS:
            self.query_one(f"#nav-{key}", ListItem).query_one(Label).update(
                title[0] if self._narrow else title
            )

    # --- navigation + refresh ----------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Switch the content pane to the picked sidebar entry and repaint it.

        The sidebar ``ListItem`` ids are ``nav-<key>`` and the matching views are
        ``view-<key>`` (see :data:`~fund.tui.views.VIEW_SPECS`), so a pick maps
        straight to the ``ContentSwitcher`` target. An unrecognised id is ignored.
        """
        item_id = event.item.id or ""
        if not item_id.startswith("nav-"):
            return
        self.query_one(ContentSwitcher).current = f"view-{item_id[len('nav-') :]}"
        self._refresh()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Re-focus the shell on the picked portfolio (or fund-wide when blank).

        Scoped to the header picker: content views own other ``Select`` widgets (e.g.
        the Mandate guardrail band), whose values are not portfolio ids — handling them
        here would send a non-UUID down ``uuid.UUID(...)``.

        A no-op pick is dropped: seeding the picker at mount echoes a
        ``Select.Changed`` whose value equals the current focus, and Textual
        dispatches it *after* ``on_mount`` returns — so the guard is value equality,
        not a mount-window flag (async dispatch would already have cleared the flag).
        A real change updates the run-state glyph and repaints. The no-selection
        sentinel is ``Select.NULL`` (the ``NoSelection`` singleton) — *not*
        ``Select.BLANK``, which resolves to ``Widget.BLANK is False`` and would send a
        blank pick down ``uuid.UUID("Select.NULL")``.
        """
        if event.select.id != "hdr-portfolio":
            return
        new_id = None if event.value is Select.NULL else uuid.UUID(str(event.value))
        if new_id == self._portfolio_id:
            return
        self._portfolio_id = new_id
        self._refresh_runstate()
        self._refresh()

    def _refresh(self) -> None:
        """Repaint the active view from the read model — synchronous, test-callable.

        A single sync entry point a headless test can call directly instead of racing
        the poll timer. Runs the cross-portfolio notification bus first (fund-wide,
        independent of the active view), then delegates to the visible
        :class:`~fund.tui.views.ShellView` for its repaint.
        """
        self._poll_attention()
        current = self.query_one(ContentSwitcher).current
        if current is None:
            return
        self.query_one(f"#{current}", ShellView).refresh_view()

    def focus_portfolio(self, portfolio_id: uuid.UUID) -> None:
        """Focus the shell on a portfolio picked inside a content view (Posture roster).

        Sets the same header context the picker does — updates ``_portfolio_id``,
        repaints the run-state glyph, and repaints the active view — and keeps the
        header ``Select`` in sync when the portfolio is a known option (its echoed
        ``Select.Changed`` is dropped by ``on_select_changed``'s value-equality guard,
        since ``_portfolio_id`` already matches).
        """
        self._portfolio_id = portfolio_id
        if portfolio_id in self._portfolio_ids:
            self.query_one("#hdr-portfolio", Select).value = str(portfolio_id)
        self._refresh_runstate()
        self._refresh()

    # --- notification bus (poll-diff → toasts + counters) ------------------

    def _poll_attention(self) -> None:
        """Diff fund-wide attention items against the seen-set; toast the new ones.

        Each tick reads the cross-portfolio awaiting gates + drift breaches, raises a
        persistent (never auto-dismissing) toast for any **newly-appeared** item, and
        bumps the header ``⏸N awaiting`` counter. A still-pending item stays in the
        seen-set, so it is not re-toasted; a resolved item drops out and would re-alert
        only if it genuinely recurs.
        """
        current = self._read_attention()
        if current is None:
            # A read error, not an empty fund: leave the seen-set and counter untouched
            # so a transient DB hiccup neither re-toasts every still-pending gate on the
            # next good tick nor flickers the "awaiting" counter to zero.
            return
        for key, message in current:
            if key not in self._seen_attention:
                self._raise_attention(message)
        self._seen_attention = {key for key, _ in current}
        self._awaiting_count = sum(1 for (kind, _), _ in current if kind == "gate")
        self._paint_attention()

    def _read_attention(self) -> list[tuple[tuple[str, uuid.UUID], str]] | None:
        """The current fund-wide attention items: awaiting gates + drift breaches.

        Each item is ``((kind, id), message)`` where ``kind`` is ``"gate"`` (keyed by
        run id) or ``"breach"`` (keyed by portfolio id). Returns ``None`` on any read
        error — distinct from an empty list — so the poll can leave prior state intact
        rather than mistaking a hiccup for a cleared fund.
        """
        items: list[tuple[tuple[str, uuid.UUID], str]] = []
        try:
            with self._session_factory() as session:
                saver = self._persistence.saver
                for row in observe.pending_hitl(session, saver, None):
                    short = str(row.run_id)[:8]
                    items.append((("gate", row.run_id), f"⏸ gate awaiting · {short}"))
                for pid in observe.list_portfolios(session):
                    if self._is_drift_breach(session, pid):
                        items.append(
                            (("breach", pid), f"◐ drift breach · {str(pid)[:8]}")
                        )
                session.rollback()
        except Exception:  # a transient read error — signal "unknown", not "empty"
            return None
        return items

    def _is_drift_breach(self, session: Session, portfolio_id: uuid.UUID) -> bool:
        """Whether ``portfolio_id``'s L1 drift breaches its mandate threshold.

        Unmandated portfolios (no threshold) never breach; the drift read is skipped
        for them so the bus does not compute state for every roster row needlessly.
        """
        mandate = observe.get_mandate(session, portfolio_id)
        if mandate is None:
            return False
        state = observe.portfolio_state(session, portfolio_id)
        return state.drift_l1 >= mandate.drift_l1_threshold

    def _raise_attention(self, message: str) -> None:
        """Raise a persistent attention toast (never auto-dismisses) + record it."""
        self.last_attention.append(message)
        self.notify(message, timeout=None)

    def _paint_attention(self) -> None:
        """Repaint the header ``⏸N awaiting`` counter from the bus's awaiting count."""
        with suppress(NoMatches):
            self.query_one("#hdr-attention", Static).update(
                f"⏸{self._awaiting_count} awaiting"
            )

    # --- decisions actions (worker thread) ---------------------------------

    def action_decide_approve(self) -> None:
        """Approve the selected Decisions gate (the ``a`` binding)."""
        self._decide("approve")

    def action_decide_reject(self) -> None:
        """Reject the selected Decisions gate (the ``r`` binding)."""
        self._decide("reject")

    def _decide(self, decision: str) -> None:
        """Dispatch ``decision`` for the Decisions view's selected gate on a worker.

        A no-op with a status message unless the Decisions view is active and a
        paused run is selected — the same guard the cockpit uses for a blind approve.
        """
        view = self._active_decisions_view()
        if view is None or view.selected_run_id is None:
            self._set_status("select a paused run first")
            return
        self.last_worker = self._action_worker(
            view.selected_run_id, view.selected_portfolio_id, decision
        )

    def action_decide_killswitch(self) -> None:
        """Killswitch the selected Decisions gate (the ``k`` binding).

        Paused-only by construction: a running langgraph invoke is not interruptible,
        and the Decisions queue lists only paused gates, so a selected run is always
        one. Pushes an explicit-consequence confirm; confirming finalises the run
        rejected and logs a distinct ``killswitch`` row (no undo), cancelling is a
        no-op. Inert — with a status hint — unless Decisions is active and a run is
        selected, the same guard the approve/reject path uses.
        """
        view = self._active_decisions_view()
        if view is None or view.selected_run_id is None:
            self._set_status("select a paused run first")
            return
        run_id = view.selected_run_id
        portfolio_id = view.selected_portfolio_id
        prompt = (
            f"KILLSWITCH run {run_id}\n\n"
            "This finalises the paused run as REJECTED. There is no undo.\n\n"
            "Confirm to kill it, Cancel to leave it paused."
        )

        def _resolved(confirmed: bool | None) -> None:
            if confirmed:
                self.last_worker = self._action_worker(
                    run_id, portfolio_id, "killswitch"
                )
            else:
                self._set_status("killswitch cancelled")

        self.push_screen(ConfirmScreen(prompt), _resolved)

    def _active_decisions_view(self) -> DecisionsView | None:
        """The Decisions view when it is the active content pane, else ``None``."""
        try:
            if self.query_one(ContentSwitcher).current != "view-decisions":
                return None
            view = self.query_one("#view-decisions", ShellView)
        except NoMatches:
            return None
        return view if isinstance(view, DecisionsView) else None

    @work(thread=True, exclusive=True, group="action")
    def _action_worker(
        self, run_id: uuid.UUID, portfolio_id: uuid.UUID | None, decision: str
    ) -> None:
        """Resolve a gate through the unified action path — off the event loop.

        The model is built lazily (a missing key surfaces as a status line, not a
        crash), ``apply_action`` resumes + logs on a freshly opened session, the caller
        commits, and the UI touches marshal back via ``call_from_thread``.
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
                    decision,
                    session=session,
                    run_id=run_id,
                    portfolio_id=portfolio_id,
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    model=model,
                    # Resolving a run is terminal: key on (run, decision) so a repeat
                    # replays the first outcome instead of re-resuming.
                    idempotency_key=f"{run_id}:{decision}",
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    def _after_action(self, message: str) -> None:
        """On the main thread once the worker committed: report + repaint the view."""
        self._set_status(message)
        self._refresh()

    # --- mandate actions (worker thread) -----------------------------------

    def action_save_mandate(self) -> None:
        """Persist the Mandate view's run-policy edits (``ctrl+s``).

        A no-op with a status hint unless the Mandate view is active and has a current
        mandate. The proposed mandate is re-validated here (a bad edit surfaces as a
        status line); the upsert + audit happen on a worker thread.
        """
        view = self._active_mandate_view()
        if view is None or view.mandate is None:
            self._set_status("focus a portfolio with a mandate first")
            return
        try:
            proposed = view.proposed_mandate()
        except Exception as exc:  # pydantic ValidationError on an out-of-contract edit
            self._set_status(f"invalid mandate: {exc}")
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

    @work(thread=True, exclusive=True, group="mandate")
    def _mandate_worker(
        self, portfolio_id: uuid.UUID | None, mandate: Any, prev_value: Any
    ) -> None:
        """Upsert the run-policy mandate through the unified path, off the loop."""
        from fund.actions import apply_action

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
        self.call_from_thread(self._after_action, result.message)

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

    # --- onboarding + drive flows (worker thread) --------------------------

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

        def _submitted(submission: OnboardingSubmission | None) -> None:
            if submission is None:
                self._set_status("onboarding cancelled")
                return
            new_id = uuid.uuid4()
            try:
                mandate = _initial_mandate(new_id, submission)
            except Exception as exc:  # pydantic ValidationError (e.g. bad capital)
                self._set_status(f"invalid onboarding: {exc}")
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

        The ``u`` key passes today's bar; callers may pass an explicit decision bar.
        A no-op with a hint unless Posture is active and a portfolio is focused.
        """
        if not self._posture_active():
            self._set_status("open Posture to run a portfolio")
            return
        pid = self._portfolio_id
        if pid is None:
            self._set_status("focus a portfolio to run")
            return
        self.last_worker = self._run_worker(pid, asof or dt.date.today())

    def _posture_active(self) -> bool:
        """Whether the Posture view is the active content pane (the drive launchpad)."""
        try:
            return self.query_one(ContentSwitcher).current == "view-posture"
        except NoMatches:
            return False

    @work(thread=True, exclusive=True, group="drive")
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
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    base_currency=base_currency,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_onboard, result.message)

    @work(thread=True, exclusive=True, group="drive")
    def _run_worker(self, portfolio_id: uuid.UUID, asof: dt.date) -> None:
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
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    def _after_onboard(self, message: str) -> None:
        """On the main thread once onboarding committed: refresh the picker + report."""
        self._populate_picker()
        self._after_action(message)

    # --- audit filters + export --------------------------------------------

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
        current (filtered) timeline is written; the path is reported on the status bar.
        """
        view = self._active_audit_view()
        if view is None:
            return
        path = f"audit-export-{dt.datetime.now():%Y%m%d-%H%M%S}.csv"
        try:
            written = view.export_csv(path)
        except OSError as exc:  # a write failure surfaces, never crashes the App
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

    # --- header picker + run state -----------------------------------------

    def _populate_picker(self) -> None:
        """Seed the header ``Select`` from the fund's known portfolios (T2 reader).

        Reads ``observe.list_portfolios`` through a short read-only session and
        records the ids on :attr:`_portfolio_ids` (a test hook). When the shell
        launched focused on a known portfolio, that option is pre-selected.
        """
        self._portfolio_ids = self._list_portfolios()
        select = self.query_one("#hdr-portfolio", Select)
        select.set_options(
            (self._short_label(pid), str(pid)) for pid in self._portfolio_ids
        )
        if self._portfolio_id in self._portfolio_ids:
            select.value = str(self._portfolio_id)

    def _list_portfolios(self) -> list[uuid.UUID]:
        """Distinct portfolio ids for the picker; ``[]`` on any read error."""
        try:
            with self._session_factory() as session:
                ids = observe.list_portfolios(session)
                session.rollback()
                return ids
        except Exception:  # keep the shell alive on a read error (mirrors the cockpit)
            return []

    def _refresh_runstate(self) -> None:
        """Repaint the header run-state glyph for the focused portfolio."""
        self._runstate = self._run_state_glyph()
        self.query_one("#hdr-runstate", Static).update(self._runstate)

    def _run_state_glyph(self) -> str:
        """The focused portfolio's latest-run glyph: running / awaiting / idle.

        Fund-wide (no focus), no runs, or a read error all read as ``idle``; a
        ``paused`` or ``responded`` (barge-in re-gate) latest run is ``awaiting`` the
        adviser, ``pending``/``running`` is live, everything else (completed / rejected)
        is idle.
        """
        if self._portfolio_id is None:
            return "✓ idle"
        try:
            with self._session_factory() as session:
                runs = observe.list_portfolio_runs(session, self._portfolio_id)
                session.rollback()
        except Exception:
            return "✓ idle"
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
        self.last_status = message
        self.query_one("#status-bar", Static).update(message)


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


def main() -> None:
    """``fund-tui`` entrypoint: build persistence, launch the shell for one run.

    The portfolio id is **optional**: no arg opens the fund-wide shell (Posture
    landing); ``fund-tui <uuid>`` focuses that portfolio. Needs ``DATABASE_URL``
    (for the checkpointer/store pool); the chat model stays lazy — only built when
    the adviser acts — so a bare launch needs no ``OLLAMA_API_KEY``. The pool is
    always closed on exit. ``typer`` is imported here (not at module top) to keep
    ``import fund.tui.app`` light.
    """
    import typer

    def _launch(portfolio_id: str | None = typer.Argument(default=None)) -> None:
        from fund.agents.model import build_primary
        from fund.audit import setup_langgraph
        from fund.database import get_session

        focus = uuid.UUID(portfolio_id) if portfolio_id else None
        persistence = setup_langgraph(settings)
        try:
            app = ShellApp(
                focus,
                session_factory=get_session,
                persistence=persistence,
                model_factory=lambda: build_primary(settings),
            )
            app.run()
        finally:
            persistence.pool.close()

    typer.run(_launch)


if __name__ == "__main__":
    main()
