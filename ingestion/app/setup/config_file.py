"""Non-secret config file for the portopt install wizard.

Stores only non-secret settings (provider name, model, exchange scope, ...) in
``~/.portopt/config.toml``. A guard refuses secret-looking keys so a wizard bug
can never leak a credential into plaintext. Reading uses stdlib ``tomllib``
(Python 3.11+); writing uses a minimal serialiser for the flat
str/bool/int/float/list values the wizard needs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomllib

DEFAULT_CONFIG_PATH = Path.home() / ".portopt" / "config.toml"

_SECRET_MARKERS = ("key", "secret", "token", "password", "passphrase")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _format_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_string(str(item)) for item in value) + "]"
    raise TypeError(f"Unsupported config value type: {type(value).__name__}")


def save_config(config: Mapping[str, Any], *, path: Path | None = None) -> Path:
    """Write non-secret ``config`` to ``path`` as TOML. Rejects secret-looking keys."""
    path = path or DEFAULT_CONFIG_PATH
    secret_keys = [k for k in config if _is_secret_key(k)]
    if secret_keys:
        raise ValueError(
            f"Refusing to write secret-looking keys to plaintext config: {secret_keys}"
        )
    lines = [f"{key} = {_format_value(value)}" for key, value in config.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_config(*, path: Path | None = None) -> dict[str, Any]:
    """Load config from ``path``; return an empty dict if it does not exist."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)
