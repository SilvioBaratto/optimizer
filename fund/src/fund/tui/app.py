"""``fund-tui`` — the ``ShellApp`` multi-view oversight cockpit.

A six-view mission-control shell (Posture · Deliberation · Decisions · Mandate ·
Audit · Ask) that can land fund-wide or focused on one portfolio. Every data
surface is read from the model-free read model (``fund.observe``); the App owns
all side effects:

* **Polling, not pushing.** ``ShellApp._refresh`` runs on a ``set_interval``
  timer (``poll_interval``, ~2s), opening a *short read-only* session, extracting
  plain dataclasses, and rolling back — the UI never holds a transaction or an ORM
  row across ticks. A view with a heavy read (Performance, Daily brief) instead
  splits its tick via ``ShellView.poll_task`` so the read runs on a worker thread and
  only the paint touches the event loop.
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

The modal screens live in :mod:`fund.tui.screens` and the shell's sections in
:mod:`fund.tui._mixins`. This module keeps ``ShellApp``'s state, layout and
navigation, and re-exports every name it defined before that split, so
existing imports and patches of ``fund.tui.app.<name>`` keep resolving.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.events import Resize
from textual.widgets import (
    ContentSwitcher,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from fund.config import FundConfig, settings
from fund.tui import ask as ask
from fund.tui._mixins.ask import _ASK_CONSEQUENCE as _ASK_CONSEQUENCE
from fund.tui._mixins.ask import _ASK_GATE_VERBS as _ASK_GATE_VERBS
from fund.tui._mixins.ask import (
    _ASK_HISTORY_RETRY_SECONDS as _ASK_HISTORY_RETRY_SECONDS,
)
from fund.tui._mixins.ask import AskMixin
from fund.tui._mixins.ask import _ask_confirm_prompt as _ask_confirm_prompt
from fund.tui._mixins.attention import (
    _ATTENTION_TOAST_TIMEOUT_S as _ATTENTION_TOAST_TIMEOUT_S,
)
from fund.tui._mixins.attention import AttentionMixin, _AttentionKey
from fund.tui._mixins.attention import _AttentionRead as _AttentionRead
from fund.tui._mixins.audit import AuditMixin
from fund.tui._mixins.decisions import DecisionsMixin
from fund.tui._mixins.decisions import _gate_target as _gate_target
from fund.tui._mixins.drive import _HOLD_POLL_SECONDS as _HOLD_POLL_SECONDS
from fund.tui._mixins.drive import DriveMixin, _QueuedDirective
from fund.tui._mixins.drive import _initial_mandate as _initial_mandate
from fund.tui._mixins.drive import _recorded_train_size as _recorded_train_size
from fund.tui._mixins.flight import _P as _P
from fund.tui._mixins.flight import _ROUND_BUSY as _ROUND_BUSY
from fund.tui._mixins.flight import _round_flight as _round_flight
from fund.tui._mixins.flight import _single_flight as _single_flight
from fund.tui._mixins.mandate import MandateMixin
from fund.tui._mixins.picker import PickerMixin
from fund.tui.exports import export_path
from fund.tui.screens import _HELP_TEXT as _HELP_TEXT
from fund.tui.screens import ConfirmScreen as ConfirmScreen
from fund.tui.screens import HelpScreen
from fund.tui.screens import OrderEditScreen as OrderEditScreen
from fund.tui.screens import RationaleConfirmScreen as RationaleConfirmScreen
from fund.tui.screens import ScopePickScreen as ScopePickScreen
from fund.tui.screens import _parse_weights as _parse_weights
from fund.tui.views import (
    RAIL_LABELS,
    VIEW_SPECS,
    AskView,
    DailyBriefView,
    DeliberationView,
    ShellView,
    _gate_text,
    build_views,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType
    from textual.worker import Worker


logger = logging.getLogger(__name__)


_NARROW_WIDTH = 80  # below this terminal width the sidebar collapses to a rail


# Each mixin is a plain class holding one section of the shell. Textual merges
# ``BINDINGS`` and collects ``@on`` handlers only from ``DOMNode`` subclasses,
# so the class attributes Textual reads stay on ``ShellApp``.
class ShellApp(
    AttentionMixin,
    DecisionsMixin,
    MandateMixin,
    DriveMixin,
    AuditMixin,
    AskMixin,
    PickerMixin,
    App[None],
):
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
    /* A bordered Select is 3 rows tall; in the 1-row header its label row would sit
       under the body, so strip the border (focus included) to keep it on one row. */
    .hdr-picker { height: 1; }
    .hdr-picker > SelectCurrent,
    .hdr-picker:focus > SelectCurrent { height: 1; border: none; padding: 0 1; }
    #hdr-feed { display: none; color: $warning; }
    .shell-body { height: 1fr; }
    .sidebar { width: 22; border-right: solid $accent; }
    .sidebar.narrow { width: 5; }
    .content { width: 1fr; padding: 0 1; }
    .view-title { text-style: bold; color: $accent; }
    .view-subtitle { color: $accent; }
    /* Not docked: the Footer owns the bottom dock edge and would paint over it. */
    .status-bar { height: 1; background: $panel; padding: 0 1; }
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
    .mandate-directives { height: 6; border: round $panel; }
    .mandate-hint { height: 1; color: $accent; }
    .audit-body { height: 1fr; }
    .audit-table { width: 55%; }
    .audit-detail { width: 1fr; padding: 0 1; color: $warning; }
    .audit-hint { height: 1; color: $accent; }
    .delib-body { height: 1fr; }
    .delib-left { width: 50%; }
    .delib-runs { height: 40%; }
    .delib-steps { height: 1fr; }
    .delib-live { height: 1fr; }
    .delib-detail-wrap { width: 1fr; }
    .delib-detail { width: 1fr; padding: 0 1; color: $warning; }
    .delib-hint { height: 1; color: $accent; }
    .perf-risk { height: auto; color: $accent; }
    .perf-body { height: 1fr; }
    .perf-holdings { width: 55%; }
    .perf-detail { width: 1fr; padding: 0 1; color: $warning; }
    .perf-hint { height: 1; color: $accent; }
    .ask-log-wrap { height: 1fr; border: round $panel; }
    .ask-log { padding: 0 1; }
    .ask-hint { height: 1; color: $accent; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("a", "decide_approve", "Approve"),
        ("r", "decide_reject", "Reject"),
        ("k", "decide_killswitch", "Killswitch"),
        ("K", "rearm", "Re-arm"),
        ("ctrl+s", "save_mandate", "Save mandate"),
        ("o", "override_guardrail", "Override guardrail"),
        ("n", "new_portfolio", "New portfolio"),
        ("p", "profile_portfolio", "Profile"),
        ("u", "run_portfolio", "Run now"),
        ("b", "barge_in", "Barge in"),
        ("t", "audit_cycle_actor", "Audit actor filter"),
        ("f", "audit_toggle_overrides", "Audit overrides filter"),
        ("g", "audit_toggle_scope", "Audit portfolio scope"),
        ("e", "export_audit", "Export audit"),
        ("question_mark", "help", "Help"),
        ("f1", "help", "Help"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        portfolio_id: uuid.UUID | None = None,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]],
        persistence: Any,
        model_factory: Callable[[], Any],
        fallback_factory: Callable[[], Any] | None = None,
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
            fallback_factory: Called lazily alongside ``model_factory`` to build
                the model the subagents retry on; ``None`` runs without one.
            config: Application settings; defaults to the process singleton.
            poll_interval: Refresh timer cadence in seconds.
        """
        super().__init__()
        self._portfolio_id = portfolio_id
        self._session_factory = session_factory
        self._persistence = persistence
        self._model_factory = model_factory
        self._fallback_factory = fallback_factory
        self._config = config
        self._poll_interval = poll_interval
        self.last_status: str = ""
        self.last_worker: Worker[None] | None = None
        # ``_portfolio_ids`` / ``_runstate`` are headless-test hooks mirroring the
        # views' ``render_count``.
        self._portfolio_ids: list[uuid.UUID] = []
        self._runstate = "✓ idle"
        self._narrow = False
        # Latches the header to ``▶ LIVE`` across poll ticks while a round streams,
        # so a mid-stream refresh does not repaint the DB-derived running glyph.
        self._streaming = False
        # When set, the live stream may advance; when cleared, it is held at a step
        # boundary while the adviser barges in. The stream worker waits on it between
        # steps; the barge-in overlay clears it on open and re-sets it on resolve, so it
        # is never left cleared (which would hang the worker).
        self._stream_resume = threading.Event()
        self._stream_resume.set()
        # Serialises the gate-mutating workers (see ``_single_flight``).
        self._action_lock = threading.Lock()
        # Held by a profile / run / stream thread for as long as it lives (see
        # ``_round_flight``).
        self._round_lock = threading.Lock()
        # Held for the length of one Ask turn: the turns of a scope share one agent
        # thread, and ``exclusive=True`` only flags an older worker cancelled while its
        # thread keeps running.
        self._ask_lock = threading.Lock()
        # Questions dispatched and not yet finished, and whether a clear is running.
        # Both are touched on the event loop only, so a question still building its
        # model, or a clear that has not yet taken the lock, is already counted.
        self._ask_pending = 0
        self._ask_clearing = False
        # Earliest time a failed history load is retried, so a brief outage does not
        # switch saving off for the whole session and a long one is not hammered.
        self._ask_history_retry_at = 0.0
        # Every read, write and delete of the saved conversation runs here, one at a
        # time and in order, so a slow or stalled database never freezes the screen and
        # a clear cannot overtake the messages saved before it.
        self._ask_io = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ask-history"
        )
        # Serialises mandate saves: ``exclusive=True`` only flags the older worker
        # cancelled while its thread keeps running, so two ctrl+s presses would
        # otherwise write the same portfolio at once.
        self._mandate_lock = threading.Lock()
        # A directive sent while the round is still streaming. ``respond`` needs the
        # run row committed and parked at its gate, neither of which holds until the
        # stream worker finishes, so it waits here and ``_deliver_queued_directive``
        # sends it (or drops it) once the stream ends.
        self._queued_directive: _QueuedDirective | None = None
        # Notification bus (poll-diff): attention keys already alerted, so a
        # still-pending gate/breach never re-toasts on the next tick.
        self._seen_attention: set[_AttentionKey] = set()
        self._awaiting_count = 0
        self._attention_degraded = False
        # Failing feed reads already logged, so a read that fails on every ~2s tick
        # writes one line per failure episode instead of one per tick.
        self._logged_attention_failures: set[str] = set()
        self.last_attention: list[str] = []  # headless-test hook (mirrors last_status)
        self._poll_read_inflight = False
        self._attention_read_inflight = False

    # --- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Yield the header, the sidebar | content split, the status bar, and footer."""
        yield Header()
        with Horizontal(id="shell-header", classes="shell-header"):
            yield Select(
                [],
                prompt="⟨fund-wide⟩",
                id="hdr-portfolio",
                classes="hdr-picker",
                allow_blank=True,
            )
            yield Static("● ARMED", id="hdr-mode")
            yield Static(self._runstate, id="hdr-runstate")
            yield Static("⏸0 awaiting", id="hdr-attention")
            yield Static("⚠ feed degraded", id="hdr-feed")
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
        # Status text embeds exception messages (pydantic's "[type=..., input_..]") that
        # Textual would otherwise parse as markup and raise on.
        yield Static("", id="status-bar", classes="status-bar", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Seed the portfolio picker, paint the run-state glyph, and start polling.

        The seeding runs with ``Select.Changed`` prevented (see :meth:`_apply_picker`),
        so it never reaches ``on_select_changed`` as a phantom pick.
        """
        # Seed the picker, then paint once the views have mounted and poll on the
        # cockpit's cadence (the placeholder views' refresh is a no-op counter bump
        # today).
        self._populate_picker()
        self._refresh_runstate()
        if self.size.width:
            self._apply_width(self.size.width)
        self.call_after_refresh(self._paint_landing)
        self.set_interval(self._poll_interval, self._poll)

    def _paint_landing(self) -> None:
        """First paint: repaint the landing view and move focus into it.

        Textual's auto-focus parks on the first focusable widget, the header portfolio
        picker, so without this the landing view's hinted keys (``d`` brief) and arrow
        selection stay dead until the adviser tabs in. The header stays reachable with
        shift+tab.
        """
        self._refresh()
        current = self.query_one(ContentSwitcher).current
        if current is not None:
            self.query_one(f"#{current}", ShellView).focus_primary()

    def on_resize(self, event: Resize) -> None:
        """Collapse/restore the sidebar rail as the terminal crosses the threshold."""
        self._apply_width(event.size.width)

    def action_help(self) -> None:
        """Open the help overlay listing the keymap (bound to ``?`` and ``F1``).

        ``?`` surfaces it for discoverability; ``F1`` fires even while a text input
        (e.g. the Ask input) holds focus, since a function key is not typed into a
        field. Escape dismisses it.
        """
        self.push_screen(HelpScreen())

    def _apply_width(self, width: int) -> None:
        """Collapse the sidebar to a short-label rail below the narrow threshold.

        At narrow widths the sidebar shrinks (``.narrow`` CSS) and each entry's
        label drops to its :data:`~fund.tui.views.RAIL_LABELS` abbreviation, leaving
        the content pane effectively single-pane; above the threshold the full
        titles and width return.
        """
        self._narrow = width < _NARROW_WIDTH
        try:
            sidebar = self.query_one("#sidebar", ListView)
        except NoMatches:
            return
        sidebar.set_class(self._narrow, "narrow")
        for key, title in VIEW_SPECS:
            self.query_one(f"#nav-{key}", ListItem).query_one(Label).update(
                RAIL_LABELS[key] if self._narrow else title
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
        key = item_id[len("nav-") :]
        self.query_one(ContentSwitcher).current = f"view-{key}"
        self._refresh()
        # View-scoped keys and the arrow selection the hints advertise only reach a
        # view once focus is inside it, so a pick must not leave focus on the rail.
        self.query_one(f"#view-{key}", ShellView).focus_primary()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Re-focus the shell on the picked portfolio (or fund-wide when blank).

        Scoped to the header picker: content views own other ``Select`` widgets (e.g.
        the Mandate guardrail band), whose values are not portfolio ids — handling them
        here would send a non-UUID down ``uuid.UUID(...)``.

        A no-op pick is dropped: :meth:`focus_portfolio` assigns the ``Select`` a value
        that already equals the focus, and Textual dispatches the resulting
        ``Select.Changed`` *after* the assigning call returns — so the guard is value
        equality, not an "echo in progress" flag (async dispatch would already have
        cleared it). Re-seeding the options never reaches this handler:
        :meth:`_apply_picker` prevents the message. A real change updates the
        run-state glyph and repaints. The no-selection sentinel is ``Select.NULL``
        (the ``NoSelection`` singleton) — *not* ``Select.BLANK``, which resolves to
        ``Widget.BLANK is False`` and would send a blank pick down
        ``uuid.UUID("Select.NULL")``.
        """
        if event.select.id != "hdr-portfolio":
            return
        new_id = None if event.value is Select.NULL else uuid.UUID(str(event.value))
        if new_id == self._portfolio_id:
            return
        self._portfolio_id = new_id
        self._refresh_runstate()
        self._refresh()

    def _poll(self) -> None:
        """Timer tick: refresh the run-state header, then repaint the active view.

        The autonomy/run-state glyphs are re-read here (not in :meth:`_refresh`) so an
        external freeze/re-arm — from the scheduler or another session — reflects on the
        next tick instead of lingering stale until the next local action, without making
        every manual/test repaint also re-read the autonomy rows. A view that declares a
        :meth:`~fund.tui.views.ShellView.poll_task` reads on a worker thread (see
        :meth:`_poll_read_worker`); every other view repaints in place.
        """
        if not self.is_running:
            # The timer outlives the widgets for a moment on shutdown; Textual clears
            # ``is_running`` before it empties the DOM, so a tick landing in that window
            # has nothing to paint. Any other NoMatches is a real bug and must surface.
            return
        self._refresh_runstate()
        self._sync_picker()
        self._poll_attention_off_loop()
        view = self._current_view()
        task = view.poll_task() if view is not None else None
        if task is None:
            if view is not None:
                view.refresh_view()
            return
        # One read in flight at a time: a read slower than the tick would otherwise
        # pile up threads, each repainting an already-superseded snapshot.
        if self._poll_read_inflight:
            return
        self._poll_read_inflight = True
        self._poll_read_worker(*task)

    @work(thread=True, group="poll")
    def _poll_read_worker(
        self, read: Callable[[], Any], apply: Callable[[Any], None]
    ) -> None:
        """Run a heavy view's read off the event loop, then paint it back on the loop.

        ``read`` and ``apply`` come from :meth:`~fund.tui.views.ShellView.poll_task`;
        this thread never queries the DOM. A failed read keeps the last paint rather
        than crashing the App.
        """
        try:
            result = read()
        except Exception:  # keep the poll loop alive on any read error
            result = None
        self.call_from_thread(self._finish_poll_read, apply, result)

    def _finish_poll_read(self, apply: Callable[[Any], None], result: Any) -> None:
        """Free the poll slot and paint ``result`` (``None`` means the read failed)."""
        self._poll_read_inflight = False
        if result is not None:
            apply(result)

    def _refresh(self) -> None:
        """Repaint the active view from the read model — synchronous, test-callable.

        A single sync entry point a headless test can call directly instead of racing
        the poll timer. Runs the cross-portfolio notification bus first (fund-wide,
        independent of the active view), then delegates to the visible
        :class:`~fund.tui.views.ShellView` for its repaint. A view that hosts editable
        inputs is responsible for keeping an unsaved edit across this repaint.
        """
        self._poll_attention()
        view = self._current_view()
        if view is not None:
            view.refresh_view()
            if isinstance(view, AskView):
                self._sync_ask_history(view)

    def _current_view(self) -> ShellView | None:
        """The visible content pane's view, or ``None`` before one is selected."""
        current = self.query_one(ContentSwitcher).current
        if current is None:
            return None
        return self.query_one(f"#{current}", ShellView)

    def focus_portfolio(self, portfolio_id: uuid.UUID) -> None:
        """Focus the shell on a portfolio picked inside a content view (Posture roster).

        Sets the same header context the picker does — updates ``_portfolio_id``,
        repaints the run-state glyph, and repaints the active view — and keeps the
        header ``Select`` in sync (its echoed ``Select.Changed`` is dropped by
        ``on_select_changed``'s value-equality guard, since ``_portfolio_id`` already
        matches). The roster re-lists portfolios every tick, so a portfolio created
        since the picker was last seeded re-seeds it first; otherwise the header would
        keep showing a different portfolio from the one ``u`` / ``K`` / approvals act
        on, and re-choosing the old one would emit no change.
        """
        self._portfolio_id = portfolio_id
        if portfolio_id not in self._portfolio_ids:
            # ``set_options`` blanks the Select first; that echo would otherwise read
            # as a fund-wide pick and drop the focus just set.
            with self.query_one("#hdr-portfolio", Select).prevent(Select.Changed):
                self._populate_picker()
        if portfolio_id in self._portfolio_ids:
            self.query_one("#hdr-portfolio", Select).value = str(portfolio_id)
        self._refresh_runstate()
        self._refresh()

    def action_open_brief(self) -> None:
        """Open the Daily brief for the focused portfolio (the Posture ``d`` tile).

        A programmatic view switch mirroring the sidebar's
        :meth:`on_list_view_selected`: points the ``ContentSwitcher`` at the brief pane,
        which reads the already-focused ``_portfolio_id``, then repaints. View-scoped to
        Posture (``PostureView.BINDINGS``) so it never clobbers the global ``b``.
        """
        self.query_one(ContentSwitcher).current = "view-brief"
        self._refresh()
        self.query_one("#view-brief", DailyBriefView).focus()

    def _active_brief_view(self) -> DailyBriefView | None:
        """The Daily brief view when it is the active content pane, else ``None``."""
        try:
            if self.query_one(ContentSwitcher).current != "view-brief":
                return None
            view = self.query_one("#view-brief", ShellView)
        except NoMatches:
            return None
        return view if isinstance(view, DailyBriefView) else None

    def action_open_deliberation(self) -> None:
        """Open the brief's day deliberation and preselect its run (the ↵ binding).

        Switches to the Deliberation view and calls
        :meth:`~fund.tui.views.DeliberationView.select_run` with the brief's
        most-recent-on-or-before run (``DailyBrief.deliberation_run_id``). A no-op with
        a hint off the brief or when the day resolves to no run (a never-run portfolio).
        """
        view = self._active_brief_view()
        if view is None:
            return
        run_id = view.brief.deliberation_run_id if view.brief else None
        if run_id is None:
            self._set_status("no deliberation for this day")
            return
        self.query_one(ContentSwitcher).current = "view-deliberation"
        self.query_one("#view-deliberation", DeliberationView).select_run(run_id)

    def action_export_brief(self) -> None:
        """Export the brief to ``daily-brief-<pf>-<asof>.md`` in the export directory.

        Bound to the ``e`` key.

        View-scoped to the brief (``DailyBriefView.BINDINGS``) so it does not collide
        with the global Audit export ``e``. Re-export overwrites (idempotent); a write
        failure surfaces on the status line. A silent no-op off the brief pane.
        """
        view = self._active_brief_view()
        if view is None or view.brief is None:
            return
        brief = view.brief
        name = f"daily-brief-{brief.portfolio_id}-{brief.asof.isoformat()}.md"
        try:
            written = view.export_markdown(str(export_path(name)))
        except OSError as exc:  # write failure → status bar, never crash
            self._set_status(f"export failed: {exc}")
            return
        self._set_status(f"exported brief to {written}")

    # Stays here, not on ``AskMixin``: it reads ``_gate_text`` from this
    # module, the name the tests replace on ``fund.tui.app``.
    def _ask_approve_gate(self, run_id: uuid.UUID) -> str | None:
        """The gate text an Ask approve shows, or ``None`` when it cannot be read.

        Decisions refuses an approve whose gate has not rendered, so the orders are
        always on screen before they execute; Ask must hold itself to the same rule
        instead of confirming a bare gate id. Read on the event loop over a short
        session, like the target resolution that precedes it.
        """
        try:
            with self._session_factory() as session:
                gate = _gate_text(session, self._persistence.saver, run_id)
                session.rollback()
        except Exception:
            logger.warning("ask approve: gate unreadable", exc_info=True)
            return None
        return gate or None


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
        from fund.agents.model import build_fallback, build_primary
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
                fallback_factory=lambda: build_fallback(settings),
            )
            app.run()
        finally:
            persistence.pool.close()

    typer.run(_launch)


if __name__ == "__main__":
    main()
