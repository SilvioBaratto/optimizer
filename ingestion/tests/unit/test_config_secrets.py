"""Config contract for Docker-secret file reads (T3).

Secret fields are overridden by mounted compose secret files
(`$PORTOPT_SECRETS_DIR/<field>`, default `/run/secrets`), stripped, file-wins;
a missing or empty file falls back to env; missing both yields "" (preserves the
T212-absent skip).
"""

import logging
from pathlib import Path

import pytest

from app.config import Settings

_SECRET_ENV = (
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


def test_secret_file_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "fred_api_key").write_text("file-key\n", encoding="utf-8")
    s = _settings(
        monkeypatch, FRED_API_KEY="env-key", PORTOPT_SECRETS_DIR=str(tmp_path)
    )
    assert s.fred_api_key == "file-key"  # file wins, trailing newline stripped


def test_missing_secret_file_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = _settings(
        monkeypatch, FRED_API_KEY="env-key", PORTOPT_SECRETS_DIR=str(tmp_path)
    )
    assert s.fred_api_key == "env-key"


def test_missing_secret_and_env_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = _settings(monkeypatch, PORTOPT_SECRETS_DIR=str(tmp_path))
    assert s.fred_api_key == ""
    assert s.trading_212_api_key == ""  # T212-absent skip stays intact


@pytest.mark.parametrize("placeholder", ["", "\n", "  \r\n"])
def test_empty_secret_file_keeps_env_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, placeholder: str
) -> None:
    # compose_secrets.render writes "" for every secret the store lacks, so an
    # empty file means "unset" and must not shadow a key supplied via env/.env.
    (tmp_path / "fred_api_key").write_text(placeholder, encoding="utf-8")
    s = _settings(
        monkeypatch, FRED_API_KEY="env-key", PORTOPT_SECRETS_DIR=str(tmp_path)
    )
    assert s.fred_api_key == "env-key"


def test_empty_secret_file_with_no_env_stays_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "trading_212_api_key").write_text("", encoding="utf-8")
    s = _settings(monkeypatch, PORTOPT_SECRETS_DIR=str(tmp_path))
    assert s.trading_212_api_key == ""


def test_empty_placeholder_does_not_block_other_secret_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "fred_api_key").write_text("", encoding="utf-8")
    (tmp_path / "trading_212_api_key").write_text("t212-file\n", encoding="utf-8")
    s = _settings(
        monkeypatch, FRED_API_KEY="env-key", PORTOPT_SECRETS_DIR=str(tmp_path)
    )
    assert (s.fred_api_key, s.trading_212_api_key) == ("env-key", "t212-file")


def _deny_read(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    # chmod 000 is a no-op for root and on Windows; patching read_text reproduces
    # the container case (0600 host file owned by another uid) on every platform.
    real_read_text = Path.read_text

    def read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == name:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)


def test_unreadable_secret_file_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "fred_api_key").write_text("file-key", encoding="utf-8")
    _deny_read(monkeypatch, "fred_api_key")
    s = _settings(
        monkeypatch, FRED_API_KEY="env-key", PORTOPT_SECRETS_DIR=str(tmp_path)
    )
    assert s.fred_api_key == "env-key"


def test_unreadable_secret_file_does_not_hide_readable_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "fred_api_key").write_text("file-key", encoding="utf-8")
    (tmp_path / "trading_212_api_key").write_text("t212-file", encoding="utf-8")
    _deny_read(monkeypatch, "fred_api_key")
    s = _settings(monkeypatch, PORTOPT_SECRETS_DIR=str(tmp_path))
    assert (s.fred_api_key, s.trading_212_api_key) == ("", "t212-file")


def test_unreadable_secret_file_is_reported_without_leaking_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "fred_api_key").write_text("file-key", encoding="utf-8")
    _deny_read(monkeypatch, "fred_api_key")
    with caplog.at_level(logging.WARNING, logger="app.config"):
        _settings(monkeypatch, PORTOPT_SECRETS_DIR=str(tmp_path))
    assert "fred_api_key" in caplog.text
    assert "file-key" not in caplog.text
