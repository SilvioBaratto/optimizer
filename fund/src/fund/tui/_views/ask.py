"""The Ask view: the saved conversation and the input the adviser types into."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.content import Content
from textual.widgets import Input, Label, Static

from fund.tui import ask
from fund.tui._views.shell import ShellView
from fund.tui.ask_history import AskHistory
from fund.tui.prose import format_ask_log
from fund.tui.safe_text import defang_markdown, strip_control

if TYPE_CHECKING:
    from collections.abc import Sequence

    from textual.binding import BindingType


CLEAR_COMMAND = "/clear"
# Messages drawn at once. The whole conversation stays saved and is exported in full,
# but re-laying-out hundreds of long answers on every new line made each one slow.
_ASK_RENDER_LIMIT = 100


class AskView(ShellView):
    """Conversational Q&A over the fund state: a log, an input, and a grounded answer.

    Read-first. Each turn is classified ``query | action`` (see
    :func:`fund.tui.ask.classify`); recognized action verbs are the decision verbs only
    (approve / reject / killswitch / freeze / re-arm / run / steer). A query is answered
    from ``fund.observe`` reads summarized by the lazily-built model on a worker thread
    the App owns; an action turn resolves its target and routes through the shared
    confirm + ``apply_action`` path — never silent execution. The view itself only
    decides and hands off; the App owns the model and every mutation.

    At module scope this imports only Textual, the stdlib, and the agent-stack-free
    classify/retrieve helpers (:mod:`fund.tui.ask`), so ``import fund.tui.app`` stays
    agent-stack-light.

    The conversation is saved per portfolio (:mod:`fund.tui.ask_history`): the shell
    binds the focused scope's history when Ask opens or the focus changes, every line
    is saved as it is logged, and ``/clear`` deletes it after a confirmation.

    Attributes:
        last_answer: Mirror of the most recent rendered answer — a headless-test hook.
        pending_action: A resolved-but-unconfirmed action awaiting disambiguation, or
            ``None`` — the cross-turn state the disambiguation flow carries.
        history: The focused scope's saved conversation, or ``None`` when unavailable.
        history_scope: The scope ``history`` belongs to, or ``None`` before first bind.
    """

    # Focusable so the view-scoped ``e`` export reaches this view's binding; ``e`` is
    # view-scoped here (mirrors DailyBriefView) so it exports the Ask conversation and
    # leaves the global Audit export ``e`` untouched.
    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        ("e", "app.export_ask", "Export"),
    ]

    def __init__(self, view_key: str, view_title: str, **kwargs: Any) -> None:
        super().__init__(view_key, view_title, **kwargs)
        self.last_answer: str = ""
        self.pending_action: ask.PendingAction | None = None
        self.history: AskHistory | None = None
        self.history_scope: str | None = None
        self._loading = False
        self._log: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        """Yield the title, the scrollable conversation log, the input, and the hint."""
        yield Label(self.view_title, classes="view-title")
        with VerticalScroll(id="ask-log-wrap", classes="ask-log-wrap"):
            # markup=False: the log holds LLM answers and exception text, whose
            # "[AAPL]" / "[/INST]" fragments the markup parser would drop or reject.
            yield Static(
                "ask about the fund — e.g. “how are we doing?” or “approve the gate”",
                id="ask-log",
                classes="ask-log",
                markup=False,
            )
        yield Input(placeholder="ask…", id="ask-input")
        yield Static(
            "↵ ask · /clear deletes this history · shift+tab to the log, then e export",
            id="ask-hint",
            classes="ask-hint",
        )

    def focus_primary(self) -> None:
        """Focus the question box rather than the bare view container.

        With the container focused, typed text is dispatched as app bindings (``q``
        quits) instead of reaching the input. The view-scoped ``e`` export stays
        reachable from the conversation log (shift+tab from the input), whose key
        bindings bubble up to this view.
        """
        self.query_one("#ask-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """A submitted turn is classified and routed, then the input is cleared."""
        if event.input.id != "ask-input":
            return
        event.stop()
        turn = event.value.strip()
        event.input.value = ""
        if turn:
            self.submit_turn(turn)

    def submit_turn(self, turn: str) -> None:
        """Log a turn and hand it to the App to resolve a pending action or route it.

        The App resolves a carried disambiguation first, else classifies the turn — a
        ``query`` dispatches the lazy summarize worker; an ``action`` routes through the
        shared confirm + ``apply_action`` path. Nothing mutates here — the view only
        logs and hands off, so an action never executes from this call.
        """
        if turn.strip().lower() == CLEAR_COMMAND:
            self.shell.ask_clear()
            return
        if self.shell.ask_turn_refused():
            return
        self.append_log("you", turn)
        self.shell.ask_submit(turn)

    def bind_history(
        self,
        history: AskHistory | None,
        scope: str,
        rows: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Show ``scope``'s saved conversation and keep saving to it.

        Called when Ask opens and whenever the focused portfolio changes, so the log on
        screen is always the conversation of the portfolio in the header. A carried
        disambiguation belongs to the previous conversation and is dropped.

        Args:
            history: The scope's saved conversation, or ``None`` when it cannot be read
                (the log then lives in memory only, as it always did).
            scope: The scope the log now belongs to.
            rows: The messages already read from ``history``, oldest first.
        """
        self.history = history
        self.history_scope = scope
        self._log = list(rows)
        self._loading = False
        self.last_answer = ""
        self.pending_action = None
        self._repaint_log()

    def begin_binding(self, history: AskHistory, scope: str) -> None:
        """Switch to ``scope`` now and show a placeholder while its messages are read.

        The read happens off the event loop. Lines typed meanwhile are kept and saved,
        and :meth:`finish_binding` puts the read messages in front of them.
        """
        self.bind_history(history, scope)
        self._loading = True
        self._repaint_log()

    def finish_binding(self, rows: Sequence[tuple[str, str]]) -> None:
        """Show the messages that were read, ahead of any typed while reading."""
        self._log = [*rows, *self._log]
        self._loading = False
        self._repaint_log()

    def clear_log(self) -> None:
        """Empty the log on screen after its saved conversation was deleted."""
        self._log = []
        self.last_answer = ""
        self.pending_action = None
        self._repaint_log()

    def append_log(self, speaker: str, text: str, *, save: bool = True) -> None:
        """Append a ``(speaker, text)`` line to the conversation and repaint the log.

        The line is also saved, so it is there next time Ask opens, unless ``save`` is
        false: an error note belongs on screen, not in the record of what was said. A
        save runs off the event loop; one that fails is logged and dropped, since losing
        a line of history must not end the cockpit.
        """
        self._log.append((speaker, text))
        if save and self.history is not None:
            self.shell.ask_persist(self.history, speaker, text)
        self._repaint_log()

    def set_answer(self, answer: str) -> None:
        """Record the model's answer (the ``last_answer`` hook) and log it."""
        self.last_answer = answer
        self.append_log("ask", answer)

    def refresh_view(self) -> None:
        """Bump the counter; the log is push-driven by submit/answer, not polled."""
        self.render_count += 1

    def _repaint_log(self) -> None:
        log = self.query_one("#ask-log", Static)
        placeholder = (
            "loading the saved conversation…"
            if self._loading
            else "ask about the fund…"
        )
        log.update(self._log_content() if self._log else placeholder)
        # The new text changes the log's height only on the next layout pass, so
        # scrolling now would stop short of the newest message.
        self.call_after_refresh(self._scroll_log_to_end)

    def _log_content(self) -> Content:
        """The newest messages as styled content, noting any older ones left out."""
        shown = self._log[-_ASK_RENDER_LIMIT:]
        content = format_ask_log(shown)
        hidden = len(self._log) - len(shown)
        if hidden <= 0:
            return content
        note = (
            f"… {hidden} earlier messages are saved but not shown (e exports them all)"
        )
        return Content.assemble((note, "dim"), "\n\n", content)

    def _scroll_log_to_end(self) -> None:
        """Scroll the conversation to its newest line."""
        self.query_one("#ask-log-wrap", VerticalScroll).scroll_end(animate=False)

    def export_markdown(self, path: str) -> str:
        """Write the conversation to ``path`` as markdown; return ``path``.

        Re-export overwrites (idempotent). Each turn renders as a bolded speaker label
        over its text, quoted line by line. Any write failure surfaces as ``OSError``
        for the caller to report on the status line.
        """
        from fund.tui.exports import open_private

        lines = ["# Ask conversation", ""]
        for speaker, text in self._log:
            lines.append(f"**{speaker}:**")
            lines.append("")
            # Model text is free prose: quote every line, blanks included, so a "#" or
            # "-" in an answer cannot become a heading or bullet of the exported file.
            rows = defang_markdown(strip_control(str(text))).splitlines() or [""]
            lines.extend(f"> {row}" if row else ">" for row in rows)
            lines.append("")
        with open_private(path) as handle:
            handle.write("\n".join(lines))
        return path
