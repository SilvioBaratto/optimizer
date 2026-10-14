"""Barge-in overlay — the adviser's free-text directive to the PM mid-deliberation.

Textual + stdlib only at import time, so ``import fund.tui.barge_in`` drags in neither
``deepagents`` nor ``langgraph`` (the import-light guard in ``test_tui.py``); the agent
stack is dispatched lazily by the App's barge-in worker (``respond`` through
``fund.actions.apply_action``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

if TYPE_CHECKING:
    from textual.binding import BindingType


@dataclass(frozen=True)
class BargeInResult:
    """One submitted barge-in directive and whether to remember it.

    Attributes:
        text: The adviser's free-text directive, threaded to the PM at the gate.
        remember: When ``True``, persist as a soft guardrail
            (``PortfolioMandate.remembered_directives``, wired in T11); when ``False``,
            a one-shot directive that seeds only the current round.
    """

    text: str
    remember: bool


class BargeInScreen(ModalScreen["BargeInResult | None"]):
    """A modal collecting a PM directive, with a one-shot / remember toggle.

    Pushed when the adviser barges into a live round: it shows where the stream paused,
    takes a directive, and — on Send (or ``↵``) with non-empty text — dismisses a
    :class:`BargeInResult`. Cancel, ``escape``, or an empty Send dismiss ``None`` (a
    caller no-op that resumes the stream unchanged).

    Args:
        paused_at: A short label of the step the stream paused on, shown for context.
    """

    DEFAULT_CSS = """
    BargeInScreen { align: center middle; }
    .bargein-dialog {
        width: 70; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    .bargein-where { height: auto; color: $warning; }
    .bargein-actions { height: auto; align-horizontal: center; padding-top: 1; }
    .bargein-actions Button { margin: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def __init__(self, paused_at: str) -> None:
        super().__init__()
        self._paused_at = paused_at

    def compose(self) -> ComposeResult:
        """Yield the paused-at context, the directive input, the remember toggle, and actions."""  # noqa: E501
        with Vertical(classes="bargein-dialog"):
            yield Static(f"⏸ paused at {self._paused_at}", classes="bargein-where")
            yield Input(placeholder="directive to the PM…", id="bargein-text")
            yield Checkbox("remember (soft guardrail)", id="bargein-remember")
            with Horizontal(classes="bargein-actions"):
                yield Button("Send", variant="primary", id="bargein-send")
                yield Button("Cancel", id="bargein-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Send submits the directive; any other button cancels."""
        if event.button.id == "bargein-send":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """``↵`` in the directive field submits (send + resume)."""
        self._submit()

    def action_cancel(self) -> None:
        """``escape`` dismisses the overlay as a cancel (a caller no-op)."""
        self.dismiss(None)

    def _submit(self) -> None:
        """Dismiss a :class:`BargeInResult`, or ``None`` when the directive is empty."""
        text = self.query_one("#bargein-text", Input).value.strip()
        if not text:
            self.dismiss(None)
            return
        remember = self.query_one("#bargein-remember", Checkbox).value
        self.dismiss(BargeInResult(text=text, remember=remember))


__all__ = ["BargeInResult", "BargeInScreen"]
