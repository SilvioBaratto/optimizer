"""Prompt seam contract (task T2).

The wizard talks to the user only through a `Prompter` seam so tests can inject
answers (questionary needs a real TTY and cannot run under CliRunner). Two
implementations: interactive `QuestionaryPrompter` and `NonInteractivePrompter`
for CI / `--non-interactive`.
"""

import pytest

from app.setup import prompts


def test_noninteractive_returns_canned_answers() -> None:
    p = prompts.NonInteractivePrompter({"Choice:": "alpha", "Connect?": True})
    assert p.select("Choice:", ["alpha", "beta"]) == "alpha"
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


# --- FallbackPrompter (plain stdin; survives Git Bash mintty) --------------


class _RecordingConsole:
    """Captures rich `.print` calls so tests assert output without rendering."""

    def __init__(self) -> None:
        self.messages: list[tuple[object, dict[str, object]]] = []

    def print(self, message: object = "", **kwargs: object) -> None:
        self.messages.append((message, kwargs))


def _scripted_input(*answers: str):
    """Return an `input`-shaped callable that yields `answers` in order."""
    it = iter(answers)

    def _fake(_prompt: str = "") -> str:
        return next(it)

    return _fake


def _raise_eof(_prompt: str = "") -> str:
    raise EOFError


def test_fallback_implements_prompter_protocol() -> None:
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    for name in ("text", "password", "select", "confirm", "error"):
        assert callable(getattr(p, name))


def test_fallback_text_returns_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", _scripted_input("typed"))
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    assert p.text("Name:") == "typed"


def test_fallback_text_empty_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", _scripted_input(""))
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    assert p.text("Base:", default="https://example.test") == "https://example.test"


def test_fallback_password_uses_getpass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts.getpass, "getpass", lambda prompt="": "s3cret")
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    assert p.password("API key:") == "s3cret"


def test_fallback_select_returns_chosen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", _scripted_input("2"))
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    assert p.select("Provider:", ["alpha", "beta", "gamma"]) == "beta"


def test_fallback_select_empty_takes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", _scripted_input(""))
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    assert p.select("Provider:", ["alpha", "beta"]) == "alpha"


def test_fallback_select_reprompts_on_bad_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", _scripted_input("abc", "9", "2"))
    console = _RecordingConsole()
    p = prompts.FallbackPrompter(console=console)
    assert p.select("Provider:", ["a", "b", "c"]) == "b"
    # two rejected answers -> two error prints (bold red)
    errors = [m for m, k in console.messages if k.get("style") == "bold red"]
    assert len(errors) == 2


def test_fallback_select_no_choices_raises() -> None:
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    with pytest.raises(prompts.PromptError):
        p.select("Pick:", [])


def test_fallback_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    monkeypatch.setattr("builtins.input", _scripted_input("y"))
    assert p.confirm("Go?") is True
    monkeypatch.setattr("builtins.input", _scripted_input("n"))
    assert p.confirm("Go?", default=True) is False
    monkeypatch.setattr("builtins.input", _scripted_input(""))
    assert p.confirm("Go?", default=True) is True


def test_fallback_confirm_reprompts_on_bad_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", _scripted_input("maybe", "yes"))
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    assert p.confirm("Go?") is True


def test_fallback_error_prints_red() -> None:
    console = _RecordingConsole()
    p = prompts.FallbackPrompter(console=console)
    assert p.error("boom") is None
    assert console.messages == [("boom", {"style": "bold red"})]


def test_fallback_input_eof_raises_prompt_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", _raise_eof)
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    with pytest.raises(prompts.PromptError):
        p.text("Name:")


def test_fallback_password_eof_raises_prompt_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(prompts.getpass, "getpass", _boom)
    p = prompts.FallbackPrompter(console=_RecordingConsole())
    with pytest.raises(prompts.PromptError):
        p.password("API key:")


# --- detection helpers -----------------------------------------------------


def test_under_mintty_false_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts.os, "name", "posix")
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    assert prompts._under_mintty() is False


def test_under_mintty_detects_msystem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts.os, "name", "nt")
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    monkeypatch.delenv("TERM", raising=False)
    assert prompts._under_mintty() is True


def test_under_mintty_detects_xterm_term(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts.os, "name", "nt")
    monkeypatch.delenv("MSYSTEM", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert prompts._under_mintty() is True


def test_under_mintty_false_on_plain_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts.os, "name", "nt")
    monkeypatch.delenv("MSYSTEM", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    assert prompts._under_mintty() is False


def test_console_bindable_false_when_output_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prompt_toolkit.output import defaults as pt_defaults

    def _boom(**_kwargs: object) -> object:
        raise RuntimeError("no console screen buffer")

    monkeypatch.setattr(pt_defaults, "create_output", _boom)
    assert prompts._console_bindable() is False


def test_console_bindable_true_when_output_creates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prompt_toolkit.output import defaults as pt_defaults

    monkeypatch.setattr(pt_defaults, "create_output", lambda **k: object())
    assert prompts._console_bindable() is True


# --- make_prompter() factory ----------------------------------------------


def test_make_prompter_no_tty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts, "_under_mintty", lambda: False)
    monkeypatch.setattr(prompts, "_has_tty", lambda: False)
    with pytest.raises(prompts.PromptError):
        prompts.make_prompter()


def test_make_prompter_mintty_returns_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # mintty makes isatty() lie, so mintty detection must win over no-TTY.
    monkeypatch.setattr(prompts, "_under_mintty", lambda: True)
    monkeypatch.setattr(prompts, "_has_tty", lambda: False)
    assert isinstance(prompts.make_prompter(), prompts.FallbackPrompter)


def test_make_prompter_console_bind_failure_returns_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "_under_mintty", lambda: False)
    monkeypatch.setattr(prompts, "_has_tty", lambda: True)
    monkeypatch.setattr(prompts, "_console_bindable", lambda: False)
    assert isinstance(prompts.make_prompter(), prompts.FallbackPrompter)


def test_make_prompter_returns_questionary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "_under_mintty", lambda: False)
    monkeypatch.setattr(prompts, "_has_tty", lambda: True)
    monkeypatch.setattr(prompts, "_console_bindable", lambda: True)
    assert isinstance(prompts.make_prompter(), prompts.QuestionaryPrompter)
