"""The modal screens the ``fund-tui`` shell pushes over its views.

Confirm, order edit, scope pick and rationale confirm guard the consequential
actions; the help overlay lists the keymap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

if TYPE_CHECKING:
    from collections.abc import Collection

    from textual.binding import BindingType


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
        width: 60; height: auto; max-height: 90%; overflow-y: auto; padding: 1 2;
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
            yield Static(self._prompt, classes="confirm-prompt", markup=False)
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
            yield Static("", id="edit-error", classes="edit-error", markup=False)
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
  tab / shift+tab move focus: header, sidebar, view
  ? / F1 help   q quit

VIEW-SCOPED
  Posture       d daily brief
  Deliberation  f follow live   x expand read-out
  Decisions     e edit weights   s steer PM
  Mandate       x remove directive
  Performance   ↵ drill into a holding
  Ask           ↵ ask   /clear delete history   shift+tab log, then e export
  Daily brief   ← / → day   ↵ deliberation   e export   esc sidebar"""


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
