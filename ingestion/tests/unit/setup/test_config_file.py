"""Non-secret config file contract (SPEC D6/wizard, task T2).

`~/.portopt/config.toml` holds only non-secret settings. These tests pin the
round-trip, the empty-on-missing behaviour, and the guard that refuses to write
secret-looking keys (so a wizard bug can never leak a key into plaintext TOML).
"""

from pathlib import Path

import pytest

from app.setup import config_file


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = {
        "llm_provider": "openai",
        "require_full_coverage": True,
        "exchanges": ["NMS", "LSE"],
        "workers": 4,
    }
    config_file.save_config(cfg, path=path)
    assert config_file.load_config(path=path) == cfg


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert config_file.load_config(path=tmp_path / "nope.toml") == {}


@pytest.mark.parametrize(
    "bad_key",
    [
        "openai_api_key",
        "TRADING_212_SECRET_KEY",
        "passphrase",
        "auth_token",
        "db_password",
    ],
)
def test_rejects_secret_looking_keys(tmp_path: Path, bad_key: str) -> None:
    with pytest.raises(ValueError, match="secret"):
        config_file.save_config({bad_key: "x"}, path=tmp_path / "config.toml")


def test_string_values_are_escaped(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    value = 'has "quotes" and \\ backslash'
    config_file.save_config({"note": value}, path=path)
    assert config_file.load_config(path=path)["note"] == value


def test_rejects_unsupported_value_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        config_file.save_config({"nested": {"a": 1}}, path=tmp_path / "config.toml")
