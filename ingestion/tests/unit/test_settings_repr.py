"""Daemon settings hold broker, FRED and database credentials; none may be printed."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings

_SENTINEL = "S3NTINEL"


def test_settings_repr_hides_every_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Neither ``repr`` nor ``str`` of the settings shows a key or the DB password."""
    # Mounted secret files win over env vars, so point the lookup at an empty
    # directory or a host with /run/secrets populated would change what loads.
    monkeypatch.setenv("PORTOPT_SECRETS_DIR", str(tmp_path))
    values = {
        "TRADING_212_API_KEY": _SENTINEL + "-broker-key",
        "TRADING_212_SECRET_KEY": _SENTINEL + "-broker-secret",
        "FRED_API_KEY": _SENTINEL + "-fred",
        "DATABASE_URL": "postgresql://user:" + _SENTINEL + "-db@host/db",
        "NOTIFICATION_WEBHOOK_URL": "https://hooks.example.test/"
        + _SENTINEL
        + "-token",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    # Without this the test would also pass if the aliases stopped loading the values.
    loaded_from = {
        "trading_212_api_key": "TRADING_212_API_KEY",
        "trading_212_secret_key": "TRADING_212_SECRET_KEY",
        "fred_api_key": "FRED_API_KEY",
        "database_url": "DATABASE_URL",
        "notification_webhook_url": "NOTIFICATION_WEBHOOK_URL",
    }
    for attribute, env_name in loaded_from.items():
        assert getattr(settings, attribute) == values[env_name]

    assert _SENTINEL not in repr(settings)
    assert _SENTINEL not in str(settings)
