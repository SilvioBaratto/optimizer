"""Prompt seam for the portopt install wizard.

All user interaction goes through a ``Prompter`` so the wizard is testable
without a TTY (questionary needs one and cannot run under Typer's CliRunner).
The interactive ``QuestionaryPrompter`` is used at runtime; ``NonInteractive
Prompter`` serves ``--non-interactive`` / CI, answering from a preset map.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import questionary


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
