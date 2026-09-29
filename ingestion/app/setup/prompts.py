"""Prompt seam for the portopt install wizard.

All user interaction goes through a ``Prompter`` so the wizard is testable
without a TTY (questionary needs one and cannot run under Typer's CliRunner).
Runtime prompters are chosen by ``make_prompter()``: ``QuestionaryPrompter``
where prompt_toolkit can bind a real console, and the plain-stdin
``FallbackPrompter`` under Git Bash mintty (where prompt_toolkit crashes with
``NoConsoleScreenBufferError``). ``NonInteractivePrompter`` serves
``--non-interactive`` / CI, answering from a preset map.
"""

from __future__ import annotations

import getpass
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import questionary
from rich.console import Console


class PromptError(Exception):
    """Raised when a prompt cannot produce an answer (e.g. the user aborted)."""


class PromptUnavailableError(PromptError):
    """Raised when a non-interactive prompter has no preset answer."""


class Prompter(Protocol):  # pragma: no cover - stub-only interface
    """The interface the wizard depends on; swap implementations in tests/CI."""

    def text(self, message: str, *, default: str | None = None) -> str: ...

    def password(self, message: str) -> str: ...

    def select(self, message: str, choices: Sequence[str]) -> str: ...

    def confirm(self, message: str, *, default: bool = False) -> bool: ...

    def error(self, message: str) -> None: ...


class QuestionaryPrompter:
    """Interactive prompter backed by questionary (requires a TTY)."""

    def text(self, message: str, *, default: str | None = None) -> str:
        return self._ask(questionary.text(message, default=default or ""))

    def password(self, message: str) -> str:
        return self._ask(questionary.password(message))

    def select(self, message: str, choices: Sequence[str]) -> str:
        return self._ask(questionary.select(message, choices=list(choices)))

    def confirm(self, message: str, *, default: bool = False) -> bool:
        return bool(self._ask(questionary.confirm(message, default=default)))

    def error(self, message: str) -> None:
        questionary.print(message, style="bold fg:red")

    @staticmethod
    def _ask(question: Any) -> Any:
        answer = question.ask()
        if answer is None:
            raise PromptError("Prompt aborted")
        return answer


class NonInteractivePrompter:
    """Answers from a preset map; raises if an answer is missing (CI / flags)."""

    def __init__(self, answers: Mapping[str, Any]) -> None:
        self._answers = dict(answers)

    def _get(self, message: str) -> Any:
        if message not in self._answers:
            raise PromptUnavailableError(f"No preset answer for prompt: {message!r}")
        return self._answers[message]

    def text(self, message: str, *, default: str | None = None) -> str:
        return str(self._get(message))

    def password(self, message: str) -> str:
        return str(self._get(message))

    def select(self, message: str, choices: Sequence[str]) -> str:
        return str(self._get(message))

    def confirm(self, message: str, *, default: bool = False) -> bool:
        return bool(self._get(message))

    def error(self, message: str) -> None:
        return None


class FallbackPrompter:
    """Plain-stdin prompter for terminals where prompt_toolkit crashes.

    Git Bash mintty has no Windows console screen buffer, so questionary/
    prompt_toolkit raise ``NoConsoleScreenBufferError`` there. This prompter
    renders with rich and reads with the builtin ``input()`` / ``getpass`` so it
    keeps working. Menus are numbered; an empty answer takes the offered default.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console if console is not None else Console()

    def text(self, message: str, *, default: str | None = None) -> str:
        suffix = f" [{default}]" if default else ""
        answer = self._input(f"{message}{suffix} ").strip()
        if not answer and default is not None:
            return default
        return answer

    def password(self, message: str) -> str:
        try:
            return getpass.getpass(f"{message} ")
        except EOFError as exc:
            raise PromptError("Prompt aborted (no input)") from exc

    def select(self, message: str, choices: Sequence[str]) -> str:
        options = list(choices)
        if not options:
            raise PromptError("select() needs at least one choice")
        self._console.print(message)
        for index, choice in enumerate(options, start=1):
            self._console.print(f"  {index}) {choice}")
        while True:
            answer = self._input(
                f"Enter a number [1-{len(options)}], default 1: "
            ).strip()
            if not answer:
                return options[0]
            try:
                picked = int(answer)
            except ValueError:
                self.error(f"Not a number: {answer!r}")
                continue
            if 1 <= picked <= len(options):
                return options[picked - 1]
            self.error(f"Choose a number between 1 and {len(options)}.")

    def confirm(self, message: str, *, default: bool = False) -> bool:
        hint = "Y/n" if default else "y/N"
        while True:
            answer = self._input(f"{message} [{hint}] ").strip().lower()
            if not answer:
                return default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            self.error("Please answer y or n.")

    def error(self, message: str) -> None:
        self._console.print(message, style="bold red")

    @staticmethod
    def _input(prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError as exc:
            raise PromptError("Prompt aborted (no input)") from exc


def _under_mintty() -> bool:
    """True on Windows Git Bash / MSYS2 / mintty.

    prompt_toolkit's Win32 backend crashes there (no console screen buffer), and
    mintty also makes ``isatty()`` report no TTY — so ``make_prompter()``
    checks this first and hands back a ``FallbackPrompter``. ``MSYSTEM``
    (e.g. ``MINGW64``) or an ``xterm*`` ``TERM`` is the reliable signal.
    """
    if os.name != "nt":
        return False
    if os.environ.get("MSYSTEM"):
        return True
    return os.environ.get("TERM", "").startswith("xterm")


def _has_tty() -> bool:
    """True when stdin and stdout are both attached to a real terminal."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (ValueError, OSError):
        return False


def _console_bindable() -> bool:
    """Probe whether prompt_toolkit can bind a real console output.

    It raises ``NoConsoleScreenBufferError`` when no Windows console screen
    buffer is available; any failure here means the interactive backend would
    crash, so we treat it as "not bindable" and fall back to plain stdin.
    """
    try:
        from prompt_toolkit.output.defaults import create_output

        create_output(stdout=sys.stdout)
    except Exception:
        return False
    return True


def make_prompter() -> Prompter:
    """Pick the interactive prompter that will actually work in this terminal.

    Order matters: mintty is detected first because it crashes prompt_toolkit
    *and* makes ``isatty()`` lie, so it must win over the no-TTY guard. A
    genuinely non-interactive stream (pipe / headless) raises ``PromptError`` —
    callers wanting CI behaviour must supply a ``NonInteractivePrompter``.
    """
    if _under_mintty():
        return FallbackPrompter()
    if not _has_tty():
        raise PromptError(
            "No interactive terminal available; run with --non-interactive."
        )
    if not _console_bindable():
        return FallbackPrompter()
    return QuestionaryPrompter()
