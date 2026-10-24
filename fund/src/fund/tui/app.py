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
import functools
import logging
import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, ParamSpec

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
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from fund import observe
from fund.config import FundConfig, settings
from fund.tui import ask
from fund.tui.barge_in import BargeInResult, BargeInScreen
from fund.tui.questionnaire import QuestionnaireScreen
from fund.tui.views import (
    RAIL_LABELS,
    VIEW_SPECS,
    AskView,
    AuditView,
    DailyBriefView,
    DecisionsView,
    DeliberationView,
    MandateView,
    ShellView,
    build_views,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session
    from textual.binding import BindingType
    from textual.worker import Worker

    from fund.tui.questionnaire import OnboardingSubmission

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _QueuedDirective:
    """A barge-in directive held back until its run reaches the committed gate."""

    run_id: uuid.UUID
    portfolio_id: uuid.UUID | None
    text: str
    remember: bool


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

    # The dialog opens right after the Enter that submitted the triggering turn, so a
    # key repeat lands on whatever holds focus: that must be the harmless button.
    AUTO_FOCUS: ClassVar[str | None] = "#confirm-no"

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


def _parse_weights(
    raw: str, *, allowed: Collection[str] | None = None
) -> dict[str, float]:
    """Parse a ``"ticker:weight, …"`` edit string into a weights mapping.

    Args:
        raw: Comma-separated ``ticker:weight`` pairs the adviser typed.
        allowed: The tickers an edit may name (the gate's proposed names). When
            given, each typed ticker is resolved to its canonical spelling (exact
            match first, then case-insensitive) and anything outside the set is
            rejected, so a typo can never reach the order path as a phantom holding.

    Returns:
        The parsed ``{ticker: weight}`` mapping (risk invariants are left to
        ``apply_action("edit")``; this guards the textual shape and the ticker set).

    Raises:
        ValueError: A pair is missing its ``ticker``/``weight``, the weight is not a
            float, the ticker is not in ``allowed`` or appears twice — surfaced
            inline so a typo never dismisses the editor.
    """
    known = frozenset(allowed) if allowed is not None else None
    folded = {name.upper(): name for name in known} if known is not None else {}
    weights: dict[str, float] = {}
    for part in raw.split(","):
        ticker, _, weight = part.partition(":")
        ticker = ticker.strip()
        if not ticker or not weight.strip():
            raise ValueError(f"bad entry {part.strip()!r}; use ticker:weight")
        if known is not None and ticker not in known:
            resolved = folded.get(ticker.upper())
            if resolved is None:
                raise ValueError(
                    f"unknown ticker {ticker!r}; not among the proposed tickers"
                )
            ticker = resolved
        if ticker in weights:
            raise ValueError(f"duplicate ticker {ticker!r}")
        try:
            weights[ticker] = float(weight)
        except ValueError:
            raise ValueError(f"bad weight for {ticker!r}") from None
    return weights


class OrderEditScreen(ModalScreen["dict[str, float] | None"]):
    """A modal to edit the gate's proposed target weights before committing.

    Pre-fills the optimizer's proposed ``{ticker: weight}`` as an editable line; on
    Commit (or ``↵``) it parses the text and — if well-formed — dismisses the edited
    mapping, which the caller threads through ``apply_action("edit")``. A parse error
    or a ticker outside the proposed set surfaces inline (the editor stays open);
    Cancel / ``escape`` / empty dismiss ``None``. The proposed tickers are the only
    names an edit may use. Risk invariants (long-only, fully-invested) are enforced
    downstream.

    Args:
        weights: The proposed target weights to pre-fill.
    """

    DEFAULT_CSS = """
    OrderEditScreen { align: center middle; }
    .edit-dialog {
        width: 70; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    .edit-prompt { height: auto; color: $warning; }
    .edit-error { height: auto; color: $error; }
    .edit-actions { height: auto; align-horizontal: center; padding-top: 1; }
    .edit-actions Button { margin: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self._weights = weights

    def compose(self) -> ComposeResult:
        """Yield the prompt, the pre-filled weights input, an error line, actions."""
        with Vertical(classes="edit-dialog"):
            yield Static(
                "edit target weights (ticker:weight, comma-separated)",
                classes="edit-prompt",
            )
            yield Input(
                value=", ".join(
                    f"{ticker}:{weight:.10g}"
                    for ticker, weight in sorted(self._weights.items())
                ),
                id="edit-weights",
            )
            yield Static("", id="edit-error", classes="edit-error")
            with Horizontal(classes="edit-actions"):
                yield Button("Commit", variant="primary", id="edit-commit")
                yield Button("Cancel", id="edit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Commit submits the edited weights; any other button cancels."""
        if event.button.id == "edit-commit":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """``↵`` in the weights field submits the edit."""
        self._submit()

    def action_cancel(self) -> None:
        """``escape`` dismisses the editor as a cancel (a caller no-op)."""
        self.dismiss(None)

    def _submit(self) -> None:
        """Dismiss the parsed weights, surfacing a parse error inline instead."""
        raw = self.query_one("#edit-weights", Input).value.strip()
        if not raw:
            self.dismiss(None)
            return
        try:
            edited = _parse_weights(raw, allowed=self._weights)
        except ValueError as exc:
            self.query_one("#edit-error", Static).update(f"⚠ {exc}")
            return
        self.dismiss(edited)


class ScopePickScreen(ModalScreen["str | None"]):
    """A modal offering a scope choice (each button a deliberate, consequential pick).

    Used by the killswitch (``run`` / ``portfolio`` / ``fund``) and re-arm
    (``portfolio`` / ``fund``): the prompt spells out each scope's consequence and every
    scope button dismisses its value; Cancel / ``escape`` dismiss ``None``.

    Args:
        prompt: The consequence text shown above the scope buttons.
        scopes: ``(label, value)`` pairs; each becomes a button dismissing ``value``.
    """

    DEFAULT_CSS = """
    ScopePickScreen { align: center middle; }
    .scope-dialog {
        width: 70; height: auto; padding: 1 2;
        border: thick $error; background: $surface;
    }
    .scope-prompt { height: auto; color: $warning; }
    .scope-actions { height: auto; align-horizontal: center; padding-top: 1; }
    .scope-actions Button { margin: 0 1; }
    """

    # Every scope button is consequential, so a stray Enter right after the opening
    # keypress must land on Cancel rather than the first scope.
    AUTO_FOCUS: ClassVar[str | None] = "#scope-cancel"

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str, scopes: list[tuple[str, str]]) -> None:
        super().__init__()
        self._prompt = prompt
        self._scopes = scopes

    def compose(self) -> ComposeResult:
        """Yield the consequence prompt above one button per scope, plus Cancel."""
        with Vertical(classes="scope-dialog"):
            yield Static(self._prompt, classes="scope-prompt")
            with Horizontal(classes="scope-actions"):
                for label, value in self._scopes:
                    yield Button(label, variant="error", id=f"scope-{value}")
                yield Button("Cancel", variant="primary", id="scope-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss the picked scope value; Cancel dismisses ``None``."""
        bid = event.button.id or ""
        if bid == "scope-cancel" or not bid.startswith("scope-"):
            self.dismiss(None)
        else:
            self.dismiss(bid[len("scope-") :])

    def action_cancel(self) -> None:
        """``escape`` dismisses the picker as a cancel (a caller no-op)."""
        self.dismiss(None)


class RationaleConfirmScreen(ModalScreen["str | None"]):
    """A confirm that is gated on a non-empty rationale (the fund-re-arm ceremony).

    Confirm stays disabled until the adviser types a rationale; confirming dismisses
    the (stripped) rationale, which the caller records on the audit row. Cancel /
    ``escape`` / an empty confirm dismiss ``None``.

    Args:
        prompt: The consequence text shown above the rationale field.
    """

    DEFAULT_CSS = """
    RationaleConfirmScreen { align: center middle; }
    .rationale-dialog {
        width: 70; height: auto; padding: 1 2;
        border: thick $error; background: $surface;
    }
    .rationale-prompt { height: auto; color: $warning; }
    .rationale-actions { height: auto; align-horizontal: center; padding-top: 1; }
    .rationale-actions Button { margin: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        """Yield the prompt, the required rationale field, and a gated Confirm row."""
        with Vertical(classes="rationale-dialog"):
            yield Static(self._prompt, classes="rationale-prompt")
            yield Input(placeholder="rationale (required)", id="rationale-text")
            with Horizontal(classes="rationale-actions"):
                yield Button(
                    "Confirm", variant="error", id="rationale-yes", disabled=True
                )
                yield Button("Cancel", variant="primary", id="rationale-no")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Enable Confirm only while the rationale field is non-empty."""
        self.query_one("#rationale-yes", Button).disabled = not event.value.strip()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Confirm dismisses the rationale; any other button cancels."""
        if event.button.id == "rationale-yes":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """``↵`` confirms when a rationale is present."""
        self._submit()

    def action_cancel(self) -> None:
        """``escape`` dismisses as a cancel (a caller no-op)."""
        self.dismiss(None)

    def _submit(self) -> None:
        """Dismiss the stripped rationale, or ``None`` when it is empty."""
        rationale = self.query_one("#rationale-text", Input).value.strip()
        self.dismiss(rationale or None)


# The keymap the help overlay lists: the global shortcuts plus the view-scoped ones
# (kept in sync with ShellApp.BINDINGS and each view's BINDINGS).
_HELP_TEXT = """\
GLOBAL
  a approve   r reject   k killswitch   K re-arm
  ctrl+s save mandate    o override guardrail
  n new portfolio   p profile   u run now   b barge in
  t audit actor   f audit overrides   g audit this-pf   e export (Audit)
  ? / F1 help   q quit

VIEW-SCOPED
  Posture       d daily brief
  Deliberation  f follow live   x expand read-out
  Decisions     e edit weights   s steer PM
  Mandate       x remove directive
  Performance   ↵ drill into a holding
  Ask           ↵ ask   e export
  Daily brief   ← / → day   ↵ deliberation   e export"""


class HelpScreen(ModalScreen[None]):
    """A dismissable overlay listing the global + view-scoped keymap.

    Pushed by the help action (``?`` / ``F1``); ``escape`` or the Close button dismisses
    it. A read-only reference — it mutates nothing.
    """

    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    .help-dialog {
        width: 76; height: auto; max-height: 90%; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    .help-title { text-style: bold; color: $accent; }
    .help-body { height: auto; }
    .help-actions { height: auto; align-horizontal: center; padding-top: 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "close_help", "Close")]

    def compose(self) -> ComposeResult:
        """Yield the title, the keymap body, and a Close button."""
        with Vertical(classes="help-dialog"):
            yield Static("fund-tui — keymap", classes="help-title")
            yield Static(_HELP_TEXT, id="help-body", classes="help-body")
            with Horizontal(classes="help-actions"):
                yield Button("Close", variant="primary", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Any button dismisses the overlay."""
        self.dismiss(None)

    def action_close_help(self) -> None:
        """``escape`` dismisses the overlay."""
        self.dismiss(None)


_NARROW_WIDTH = 80  # below this terminal width the sidebar collapses to a rail
# How often a barge-in-held stream worker re-checks whether the app is quitting.
_HOLD_POLL_SECONDS = 0.25

# ``App.notify`` has no "never expire" value: ``timeout=None`` falls back to the
# default (5s). A day outlasts any unattended stretch, and the adviser can still click
# a toast away.
_ATTENTION_TOAST_TIMEOUT_S = 24 * 60 * 60.0

_AttentionKey = tuple[str, uuid.UUID]


@dataclass(frozen=True)
class _AttentionRead:
    """One tick's read of the fund-wide attention feed, with what could not be read.

    Attributes:
        items: ``(key, message)`` for every gate / breach the tick could see.
        unknown_kinds: Item kinds (``"gate"`` / ``"breach"``) whose whole read failed,
            so their prior state is neither confirmed nor cleared.
        unreadable_breaches: Portfolios whose drift check failed on its own while the
            rest of the roster read fine.
    """

    items: list[tuple[_AttentionKey, str]]
    unknown_kinds: frozenset[str] = frozenset()
    unreadable_breaches: frozenset[uuid.UUID] = frozenset()

    @property
    def degraded(self) -> bool:
        """Whether any part of the feed could not be read this tick."""
        return bool(self.unknown_kinds or self.unreadable_breaches)

    def is_unknown(self, key: _AttentionKey) -> bool:
        """Whether ``key``'s state is undetermined, so it must not be cleared."""
        kind, ident = key
        return kind in self.unknown_kinds or (
            kind == "breach" and ident in self.unreadable_breaches
        )


_P = ParamSpec("_P")


def _single_flight(
    label: str,
) -> Callable[
    [Callable[Concatenate[ShellApp, _P], None]],
    Callable[Concatenate[ShellApp, _P], None],
]:
    """Run a gate-mutating thread worker only while no other one is in flight.

    Textual's ``exclusive=True`` merely flags the *older* ``Worker`` cancelled; its
    thread keeps running. Without this guard a repeat or conflicting keypress during
    the seconds a graph resume takes starts a second thread on the same paused run, so
    approve + reject both execute and whichever commits last decides the run's final
    state. A non-blocking lock refuses the late action with a status line rather than
    queueing a decision made against a stale screen, and it is released on every exit
    path, failures included. The portfolio / fund freeze is deliberately not guarded:
    see :meth:`ShellApp._freeze_worker`.

    Args:
        label: Present-tense phrase shown on the status line while the action runs.

    Returns:
        A decorator for a ``ShellApp`` worker body, applied beneath ``@work``.
    """

    def decorate(
        body: Callable[Concatenate[ShellApp, _P], None],
    ) -> Callable[Concatenate[ShellApp, _P], None]:
        @functools.wraps(body)
        def guarded(self: ShellApp, *args: _P.args, **kwargs: _P.kwargs) -> None:
            if not self._action_lock.acquire(blocking=False):
                self.call_from_thread(
                    self._set_status,
                    "another action is still in progress — wait for it to finish",
                )
                return
            try:
                self.call_from_thread(self._set_status, f"{label}…")
                body(self, *args, **kwargs)
            finally:
                self._action_lock.release()

        return guarded

    return decorate


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
        yield Static("", id="status-bar", classes="status-bar")
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
        self.call_after_refresh(self._refresh)
        self.set_interval(self._poll_interval, self._poll)

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
        # Focus a focusable content view so its key bindings fire. Only the brief is
        # focusable, so other views keep focus on the sidebar until a row is picked.
        view = self.query_one(f"#view-{key}", ShellView)
        if view.can_focus:
            view.focus()

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
        every manual/test repaint also re-read the autonomy rows.
        """
        if not self.is_running:
            # The timer outlives the widgets for a moment on shutdown; Textual clears
            # ``is_running`` before it empties the DOM, so a tick landing in that window
            # has nothing to paint. Any other NoMatches is a real bug and must surface.
            return
        self._refresh_runstate()
        self._sync_picker()
        self._refresh()

    def _refresh(self) -> None:
        """Repaint the active view from the read model — synchronous, test-callable.

        A single sync entry point a headless test can call directly instead of racing
        the poll timer. Runs the cross-portfolio notification bus first (fund-wide,
        independent of the active view), then delegates to the visible
        :class:`~fund.tui.views.ShellView` for its repaint. A view that hosts editable
        inputs is responsible for keeping an unsaved edit across this repaint.
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
        header ``Select`` naming the focused portfolio (assigning an existing option
        echoes a ``Select.Changed`` that ``on_select_changed``'s value-equality guard
        drops, since ``_portfolio_id`` already matches). A portfolio created after the
        picker was seeded is added to its options first, so the header never shows a
        different portfolio from the one ``u`` / ``K`` / approvals act on.
        """
        self._portfolio_id = portfolio_id
        if portfolio_id in self._portfolio_ids:
            self.query_one("#hdr-portfolio", Select).value = str(portfolio_id)
        else:
            self._populate_picker()
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
        """Export the brief to ``daily-brief-<pf>-<asof>.md`` in CWD (the ``e`` key).

        View-scoped to the brief (``DailyBriefView.BINDINGS``) so it does not collide
        with the global Audit export ``e``. Re-export overwrites (idempotent); a write
        failure surfaces on the status line. A silent no-op off the brief pane.
        """
        view = self._active_brief_view()
        if view is None or view.brief is None:
            return
        brief = view.brief
        path = f"daily-brief-{brief.portfolio_id}-{brief.asof.isoformat()}.md"
        try:
            written = view.export_markdown(path)
        except OSError as exc:  # write failure → status bar, never crash
            self._set_status(f"export failed: {exc}")
            return
        self._set_status(f"exported brief to {written}")

    # --- notification bus (poll-diff → toasts + counters) ------------------

    def _poll_attention(self) -> None:
        """Diff fund-wide attention items against the seen-set; toast the new ones.

        Each tick reads the cross-portfolio awaiting gates + drift breaches, raises a
        persistent (stays until dismissed) toast for any **newly-appeared** item, and
        bumps the header ``⏸N awaiting`` counter. A still-pending item stays in the
        seen-set, so it is not re-toasted; a resolved item drops out and would re-alert
        only if it genuinely recurs.

        A read that failed is *unknown*, not *empty*: the items it would have covered
        keep their prior seen-state and the awaiting counter keeps its last value, so a
        DB hiccup neither re-toasts every still-pending gate on the next good tick nor
        flickers the counter to zero. Reads fail independently — one portfolio's
        unreadable mandate must not silence the gates or the other portfolios — and
        the header flags the feed as degraded while any of them is failing.
        """
        read = self._read_attention()
        for key, message in read.items:
            if key not in self._seen_attention:
                self._raise_attention(message)
        carried = {key for key in self._seen_attention if read.is_unknown(key)}
        self._seen_attention = {key for key, _ in read.items} | carried
        if "gate" not in read.unknown_kinds:
            self._awaiting_count = sum(
                1 for (kind, _), _ in read.items if kind == "gate"
            )
        self._attention_degraded = read.degraded
        self._paint_attention()

    def _read_attention(self) -> _AttentionRead:
        """The current fund-wide attention items: awaiting gates + drift breaches.

        Each item is ``((kind, id), message)`` where ``kind`` is ``"gate"`` (keyed by
        run id) or ``"breach"`` (keyed by portfolio id). The gate read, the roster read
        and each portfolio's drift check are isolated from one another; whatever
        failed is reported on the result instead of being folded into "no items", and
        logged (once per failure episode) so the header's degraded flag can be traced
        to the read behind it.
        """
        items: list[tuple[_AttentionKey, str]] = []
        unknown: set[str] = set()
        unreadable: set[uuid.UUID] = set()
        failing: set[str] = set()
        try:
            with self._session_factory() as session:
                saver = self._persistence.saver
                try:
                    for row in observe.pending_hitl(session, saver, None):
                        short = str(row.run_id)[:8]
                        items.append(
                            (("gate", row.run_id), f"⏸ gate awaiting · {short}")
                        )
                except Exception:  # unknown gates, not "no gates"
                    session.rollback()
                    unknown.add("gate")
                    self._log_attention_failure(failing, "gate", "pending gates")
                try:
                    roster = observe.list_portfolios(session)
                except Exception:  # unknown breaches, not "no breaches"
                    session.rollback()
                    unknown.add("breach")
                    self._log_attention_failure(failing, "roster", "portfolio roster")
                    roster = []
                for pid in roster:
                    try:
                        if self._is_drift_breach(session, pid):
                            items.append(
                                (("breach", pid), f"◐ drift breach · {str(pid)[:8]}")
                            )
                    except Exception:  # only this portfolio (e.g. a bad mandate)
                        session.rollback()
                        unreadable.add(pid)
                        self._log_attention_failure(
                            failing, f"breach:{pid}", f"drift check for portfolio {pid}"
                        )
                session.rollback()
        except Exception:  # the session itself failed: nothing read can be trusted
            self._log_attention_failure(failing, "session", "database session")
            self._logged_attention_failures &= failing
            return _AttentionRead([], frozenset({"gate", "breach"}))
        self._logged_attention_failures &= failing
        return _AttentionRead(items, frozenset(unknown), frozenset(unreadable))

    def _log_attention_failure(self, failing: set[str], key: str, what: str) -> None:
        """Note a failed feed read for this tick; log it if it just started failing.

        Called from inside the ``except`` that caught the error, so ``exc_info`` carries
        the traceback. A read that stops failing drops out of the logged set (see
        :meth:`_read_attention`), so a recurrence is logged again.

        Args:
            failing: This tick's failing-read keys, extended with ``key``.
            key: Stable identity of the read (the gate read, the roster read, or one
                portfolio's drift check).
            what: Human description of the read, naming the portfolio where relevant.
        """
        failing.add(key)
        if key not in self._logged_attention_failures:
            self._logged_attention_failures.add(key)
            logger.warning("attention feed: %s unreadable", what, exc_info=True)

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
        """Raise an attention toast that stays until dismissed + record it."""
        self.last_attention.append(message)
        self.notify(message, timeout=_ATTENTION_TOAST_TIMEOUT_S)

    def _paint_attention(self) -> None:
        """Repaint the header ``⏸N awaiting`` counter and the feed-degraded flag."""
        with suppress(NoMatches):
            self.query_one("#hdr-attention", Static).update(
                f"⏸{self._awaiting_count} awaiting"
            )
            self.query_one("#hdr-feed", Static).display = self._attention_degraded

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
        """Killswitch the selected Decisions gate at a chosen scope (the ``k`` binding).

        Pushes a scope picker (no undo): ``run`` finalises the paused run REJECTED (the
        original killswitch, logged distinctly); ``portfolio`` / ``fund`` FREEZE the
        autonomy at that scope (``apply_action("freeze")``), halting live runs in scope
        until an explicit re-arm. Cancelling is a no-op. Inert — with a status hint —
        unless Decisions is active and a run is selected (approve/reject's guard).
        """
        view = self._active_decisions_view()
        if view is None or view.selected_run_id is None:
            self._set_status("select a paused run first")
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

    def action_edit_decision(self) -> None:
        """Edit the selected gate's proposed weights before committing (``e`` binding).

        View-scoped to Decisions (``DecisionsView.BINDINGS``) so it does not collide
        with the global Audit export ``e``. Reads the gate's proposed weights, opens the
        editor, and on Commit threads the edit through ``apply_action("edit")``. A no-op
        with a hint off Decisions or without a selected paused gate.
        """
        view = self._active_decisions_view()
        if view is None or view.selected_run_id is None:
            self._set_status("select a paused gate to edit")
            return
        run_id = view.selected_run_id
        portfolio_id = view.selected_portfolio_id
        weights = self._proposed_weights(run_id)
        if not weights:
            self._set_status("no proposed weights to edit")
            return

        def _edited(edited: dict[str, float] | None) -> None:
            if edited is None:
                self._set_status("edit cancelled")
                return
            self.last_worker = self._edit_worker(run_id, portfolio_id, edited)

        self.push_screen(OrderEditScreen(weights), _edited)

    def action_steer_decision(self) -> None:
        """Steer the PM on the selected gate with a directive (the ``s`` binding).

        View-scoped to Decisions; reuses the barge-in overlay and threads the directive
        through ``apply_action("respond")`` so the run re-gates ``responded``. A no-op
        with a hint off Decisions or without a selected paused gate.
        """
        view = self._active_decisions_view()
        if view is None or view.selected_run_id is None:
            self._set_status("select a paused gate to steer")
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

        self.push_screen(BargeInScreen("the selected gate"), _resolved)

    def _proposed_weights(self, run_id: uuid.UUID) -> dict[str, float]:
        """The weights to seed the editor with, or ``{}`` on any read error.

        Needs a pending gate (its ``place_orders`` interrupt). The seed is the run's
        audited optimizer output at full precision when there is one — that is what
        approving actually places — and only otherwise the model's own, possibly
        rounded, ``place_orders`` args. The run repo is imported from its submodule
        (never the ``fund.audit`` root, which pulls the langgraph bootstrap) so this
        stays agent-stack-light.
        """
        from fund.audit.repository import AgentRunRepository

        try:
            with self._session_factory() as session:
                repo = AgentRunRepository(session)
                run = repo.get_run(run_id)
                interrupt = (
                    observe.interrupt_for(self._persistence.saver, run) if run else None
                )
                audited = repo.latest_optimizer_weights(run_id) if interrupt else {}
                session.rollback()
        except Exception:  # keep the shell alive on a read error
            return {}
        if audited:
            return audited
        requests = (interrupt or {}).get("action_requests") or []
        args = (requests[0].get("args") if requests else {}) or {}
        weights = args.get("weights")
        if not isinstance(weights, dict):
            return {}
        return {str(ticker): float(weight) for ticker, weight in weights.items()}

    @work(thread=True, group="action")
    @_single_flight("resolving gate")
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

    @work(thread=True, group="action")
    @_single_flight("committing edit")
    def _edit_worker(
        self,
        run_id: uuid.UUID,
        portfolio_id: uuid.UUID | None,
        weights: dict[str, float],
    ) -> None:
        """Commit adviser-edited weights through the unified path — off the event loop.

        Dispatches ``apply_action("edit", new_value=weights)``: the edit validates the
        risk invariants, appends the override as a fresh allocator decision, and resumes
        the run via ``approve`` so every downstream record reflects the edit. A risk
        violation or any failure surfaces as a status line rather than crashing the App.
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
                    "edit",
                    session=session,
                    run_id=run_id,
                    portfolio_id=portfolio_id,
                    new_value=weights,
                    model=model,
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                    idempotency_key=f"{run_id}:edit",
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

    def _after_action(self, message: str) -> None:
        """On the main thread once a worker committed: report + repaint view/header."""
        self._set_status(message)
        self._refresh()
        self._refresh_runstate()

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

        ``exclusive=True`` only cancels a worker's Textual handle, never its thread, so
        a second launch would run next to the first: both threads feed one live table
        (a duplicate-row error aborts one) and each opens its own run and gate for the
        same bar. The launch is refused instead of relying on that cancel.
        """
        return any(
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
        paused_at = view.live_text.splitlines()[0] if view.live_text else "the gate"

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
                    )
                else:
                    weights = repo.latest_optimizer_weights(run_id)
                    # Risk evaluates the run's authoritative allocation; pair its bounds
                    # with the SAME newest allocator proposal those weights came from
                    # (decisions are decision_index ASC, so scan reversed — mirrors
                    # ``observe._latest_optimizer_metrics``).
                    constraint_set = next(
                        (
                            d.constraint_set
                            for d in reversed(run.decisions)
                            if d.agent == "allocator" and d.step == "optimize_portfolio"
                        ),
                        None,
                    )
                    result = readouts_compute.compute_risk(
                        weights, constraint_set, None
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
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                )
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    @work(thread=True, exclusive=True, group="stream")
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
    @_single_flight("steering the PM")
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
                    checkpointer=self._persistence.saver,
                    store=self._persistence.store,
                )
                if remember and portfolio_id is not None:
                    self._remember_directive(session, portfolio_id, directive)
                session.commit()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"action failed: {exc}")
            return
        self.call_from_thread(self._after_action, result.message)

    def _remember_directive(
        self, session: Session, portfolio_id: uuid.UUID, directive: str
    ) -> None:
        """Upsert ``directive`` into the portfolio mandate's ``remembered_directives``.

        A no-op when the portfolio has no mandate or already carries the directive, so a
        repeated remember never duplicates it. Runs in the caller's transaction (no
        commit here).
        """
        from fund.actions import apply_action

        mandate = observe.get_mandate(session, portfolio_id)
        if mandate is None or directive in mandate.remembered_directives:
            return
        updated = mandate.model_copy(
            update={
                "remembered_directives": (*mandate.remembered_directives, directive)
            }
        )
        apply_action(
            "mandate_edit",
            session=session,
            portfolio_id=portfolio_id,
            mandate=updated,
        )

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

    # --- ask (query summarize + action routing on worker threads) ----------

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
        if view.pending_action is not None:
            self._resolve_pending(view, turn)
        elif ask.classify(turn) == "query":
            self.ask_query(turn)
        else:
            self.ask_action(turn)

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
        self.last_worker = self._ask_worker(turn, view)

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
        with self._session_factory() as session:
            resolution = ask.resolve_pending(
                pending, turn, session, self._persistence.saver, self._portfolio_id
            )
            session.rollback()
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

        def _resolved(ok: bool | None) -> None:
            if ok:
                self.last_worker = self._dispatch_ask_action(view, action)
            else:
                self._ask_cancelled(view, action.verb)

        self.push_screen(ConfirmScreen(_ask_confirm_prompt(action)), _resolved)

    def _dispatch_ask_action(
        self, view: AskView, action: ask.ResolvedAction
    ) -> Worker[None] | None:
        """Dispatch a confirmed ``action`` through the matching action worker."""
        view.append_log("ask", f"{action.verb} confirmed")
        verb = action.verb
        if verb in ("approve", "reject", "killswitch") and action.run_id is not None:
            return self._action_worker(action.run_id, action.portfolio_id, verb)
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

        self.push_screen(BargeInScreen("the selected gate"), _resolved)

    def _ask_cancelled(self, view: AskView, verb: str) -> None:
        """Report a cancelled Ask action on the log + status line (a no-op mutation)."""
        view.append_log("ask", f"{verb} cancelled")
        self._set_status(f"{verb} cancelled")

    def _resolve_ask(self, turn: str) -> ask.TargetResolution:
        """Resolve an action turn's target over a short read session (no execution)."""
        with self._session_factory() as session:
            resolution = ask.resolve_target(
                turn, session, self._persistence.saver, self._portfolio_id
            )
            session.rollback()
        return resolution

    def action_export_ask(self) -> None:
        """Export the Ask conversation to ``ask-<pf>-<timestamp>.md`` in CWD (``e``).

        View-scoped to Ask (``AskView.BINDINGS``) so it does not collide with the global
        Audit export ``e``. Re-export overwrites (idempotent); a write failure surfaces
        on the status line. A silent no-op off the Ask pane.
        """
        view = self._active_ask_view()
        if view is None:
            return
        pid = self._portfolio_id or "fund"
        path = f"ask-{pid}-{dt.datetime.now():%Y%m%d-%H%M%S}.md"
        try:
            written = view.export_markdown(path)
        except OSError as exc:  # write failure → status bar, never crash
            self._set_status(f"export failed: {exc}")
            return
        self._set_status(f"exported ask to {written}")

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
    def _ask_worker(self, turn: str, view: AskView) -> None:
        """Summarize a grounded context into an answer off the event loop (lazy model).

        Builds the grounded context from the agent-stack-free read model, lazily builds
        the chat model via the injected factory, invokes it on this worker thread, and
        marshals the answer back via ``call_from_thread``. ``view`` is resolved by the
        caller on the event loop and passed in — this thread never queries the DOM,
        touching the view only through ``call_from_thread``. A missing/failed model
        degrades to a status line + a log note, never a crash. The langchain message
        types are imported lazily so ``import fund.tui.app`` stays agent-stack-light.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        focus = self._portfolio_id
        try:
            model = self._model_factory()
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"model unavailable: {exc}")
            self.call_from_thread(view.append_log, "ask", f"(model unavailable: {exc})")
            return
        if model is None:
            self.call_from_thread(self._set_status, "no model configured")
            self.call_from_thread(view.append_log, "ask", "(no model configured)")
            return
        try:
            with self._session_factory() as session:
                context = ask.build_context(
                    session, self._persistence.saver, focus, turn
                )
                session.rollback()
            prompt = [
                SystemMessage(content=_ASK_SYSTEM),
                HumanMessage(content=f"{context}\n\nQUESTION: {turn}"),
            ]
            answer = _answer_text(model.invoke(prompt))
        except Exception as exc:  # surface a clean message, never crash the App
            self.call_from_thread(self._set_status, f"ask failed: {exc}")
            self.call_from_thread(view.append_log, "ask", f"(ask failed: {exc})")
            return
        self.call_from_thread(view.set_answer, answer)

    # --- header picker + run state -----------------------------------------

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
        portfolio's own row; a read error falls back to ``● ARMED``. Portfolios frozen
        *other than* the focused one are counted after the glyph (``· ❄ 1 other
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
            return "● ARMED"
        glyph = "❄ FROZEN" if frozen else "● ARMED"
        if others:
            qualifier = "other " if focus is not None else ""
            glyph += f" · ❄ {len(others)} {qualifier}frozen"
        return glyph

    def _run_state_glyph(self) -> str:
        """The focused portfolio's latest-run glyph: live / running / awaiting / idle.

        A streaming round latches ``▶ LIVE`` (poll-stable). Otherwise: fund-wide (no
        focus), no runs, or a read error all read as ``idle``; a ``paused`` or
        ``responded`` (barge-in re-gate) latest run is ``awaiting`` the adviser,
        ``pending``/``running`` is live, everything else (completed / rejected) is idle.
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


# The Ask summarizer's system prompt. Deliberately carries no fund-role marker (unlike
# the PM/subagent prompts) so it never trips a role-aware scripted model into a wrong
# script, and instructs the model to answer only from the grounded context.
_ASK_SYSTEM = (
    "You are the fund supervisor's analyst. Answer the question using only the "
    "grounded FUND STATE context below. Be concise and specific; if the context does "
    "not contain the answer, say so plainly rather than speculating."
)


def _answer_text(message: Any) -> str:
    """Flatten a chat model reply's ``content`` (a str or a list of content blocks)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


# The spelled-out consequence per Ask action verb, for the confirm dialog.
_ASK_CONSEQUENCE: dict[str, str] = {
    "approve": "Approve and execute",
    "reject": "Reject (finalise)",
    "killswitch": "KILLSWITCH — finalise rejected (no undo)",
    "freeze": "FREEZE autonomy (halts runs)",
    "rearm": "Re-arm autonomy",
    "run": "Run a paper rebalance for",
}


def _ask_confirm_prompt(action: Any) -> str:
    """The spelled-out confirm prompt for a resolved Ask ``action``."""
    verb = action.verb
    if action.run_id is not None:
        target = f"gate #{str(action.run_id)[:8]}"
    elif action.scope == "fund":
        target = "the whole fund"
    elif action.portfolio_id is not None:
        target = f"portfolio {str(action.portfolio_id)[:8]}"
    else:
        target = "this"
    return f"{_ASK_CONSEQUENCE.get(verb, verb)} {target}?"


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
