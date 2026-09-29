"""Non-secret config file contract (SPEC D6/wizard).

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
        "universe_source": "yfinance",
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
        "fred_api_key",
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


def test_update_config_merges_into_existing(tmp_path: Path) -> None:
    """update_config adds a key without clobbering the keys already on disk."""
    path = tmp_path / "config.toml"
    config_file.save_config({"llm_provider": "openai"}, path=path)
    config_file.update_config({"repo_path": "/home/u/optimizer"}, path=path)
    assert config_file.load_config(path=path) == {
        "llm_provider": "openai",
        "repo_path": "/home/u/optimizer",
    }


def test_update_config_creates_file_when_absent(tmp_path: Path) -> None:
    """update_config on a missing file writes just the update."""
    path = tmp_path / "config.toml"
    config_file.update_config({"repo_path": "/repo"}, path=path)
    assert config_file.load_config(path=path) == {"repo_path": "/repo"}


def test_update_config_still_rejects_secret_keys(tmp_path: Path) -> None:
    """The secret-key guard applies to merged updates too."""
    with pytest.raises(ValueError, match="secret"):
        config_file.update_config({"fred_api_key": "x"}, path=tmp_path / "config.toml")
