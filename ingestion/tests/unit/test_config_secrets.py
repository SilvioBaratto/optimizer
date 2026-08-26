"""Config contract for cloud LLM provider fields + Docker-secret file reads (T3).

- `llm_provider` is cloud-only (openai/anthropic), case-normalized, rejects local.
- Secret fields are overridden by mounted compose secret files
  (`$PORTOPT_SECRETS_DIR/<field>`, default `/run/secrets`), stripped, file-wins;
  a missing file falls back to env; missing both yields "" (preserves the
  T212-absent skip).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

_SECRET_ENV = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "FRED_API_KEY",
    "TRADING_212_API_KEY",
    "TRADING_212_SECRET_KEY",
    "PORTOPT_SECRETS_DIR",
)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key in _SECRET_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_llm_provider_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings(monkeypatch).llm_provider == "openai"


def test_llm_provider_normalizes_case(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings(monkeypatch, LLM_PROVIDER="Anthropic").llm_provider == "anthropic"


@pytest.mark.parametrize("bad", ["ollama", "local", "llama3", ""])
def test_llm_provider_rejects_non_cloud(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    with pytest.raises(ValidationError):
        _settings(monkeypatch, LLM_PROVIDER=bad)


def test_secret_file_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "openai_api_key").write_text("file-key\n", encoding="utf-8")
    s = _settings(
        monkeypatch, OPENAI_API_KEY="env-key", PORTOPT_SECRETS_DIR=str(tmp_path)
    )
    assert s.openai_api_key == "file-key"  # file wins, trailing newline stripped


def test_missing_secret_file_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = _settings(
        monkeypatch, OPENAI_API_KEY="env-key", PORTOPT_SECRETS_DIR=str(tmp_path)
    )
    assert s.openai_api_key == "env-key"


def test_missing_secret_and_env_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = _settings(monkeypatch, PORTOPT_SECRETS_DIR=str(tmp_path))
    assert s.openai_api_key == ""
    assert s.trading_212_api_key == ""  # T212-absent skip stays intact
