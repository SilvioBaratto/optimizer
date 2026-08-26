"""Prompt seam contract (task T2).

The wizard talks to the user only through a `Prompter` seam so tests can inject
answers (questionary needs a real TTY and cannot run under CliRunner). Two
implementations: interactive `QuestionaryPrompter` and `NonInteractivePrompter`
for CI / `--non-interactive`.
"""

import pytest

from app.setup import prompts


def test_noninteractive_returns_canned_answers() -> None:
    p = prompts.NonInteractivePrompter({"Provider:": "openai", "Connect?": True})
    assert p.select("Provider:", ["openai", "anthropic"]) == "openai"
    assert p.confirm("Connect?") is True
    assert p.error("ignored") is None  # error is a no-op without a TTY


def test_noninteractive_missing_answer_raises() -> None:
    p = prompts.NonInteractivePrompter({})
    with pytest.raises(prompts.PromptUnavailableError):
        p.password("API key:")


def test_noninteractive_text_honours_preset_over_default() -> None:
    p = prompts.NonInteractivePrompter({"Name:": "silvio", "Base:": "https://x"})
    assert p.text("Name:") == "silvio"
    assert p.text("Base:", default="https://fallback") == "https://x"


def test_questionary_prompter_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Q:
        def __init__(self, val: object) -> None:
            self._val = val

        def ask(self) -> object:
            return self._val

    monkeypatch.setattr(prompts.questionary, "text", lambda m, **k: _Q("typed"))
    monkeypatch.setattr(prompts.questionary, "password", lambda m, **k: _Q("secret"))
    monkeypatch.setattr(
        prompts.questionary, "select", lambda m, choices, **k: _Q(choices[0])
    )
    monkeypatch.setattr(prompts.questionary, "confirm", lambda m, **k: _Q(True))
    monkeypatch.setattr(prompts.questionary, "print", lambda *a, **k: None)

    p = prompts.QuestionaryPrompter()
    assert p.text("t") == "typed"
    assert p.password("p") == "secret"
    assert p.select("s", ["a", "b"]) == "a"
    assert p.confirm("c") is True
    assert p.error("boom") is None


def test_questionary_prompter_aborted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Aborted:
        def ask(self) -> None:
            return None

    monkeypatch.setattr(prompts.questionary, "password", lambda m, **k: _Aborted())
    p = prompts.QuestionaryPrompter()
    with pytest.raises(prompts.PromptError):
        p.password("p")
