"""Render decrypted secrets into Docker-compose secret files (SPEC D6).

`portopt start` decrypts `~/.portopt/secrets.enc` and calls `render` to write
each value to `./secrets/<name>` (owner-only 0600), which docker-compose mounts
at `/run/secrets/<name>`. Every declared secret is written — empty when
unconfigured — so `docker compose up` never fails on a missing file. `portopt
stop` calls `cleanup` to remove the plaintext files.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path

# Must match the top-level `secrets:` keys in docker-compose.yml.
SECRET_NAMES = (
    "trading_212_api_key",
    "trading_212_secret_key",
    "fred_api_key",
    "openai_api_key",
    "anthropic_api_key",
)

DEFAULT_SECRETS_DIR = Path("secrets")


def render(
    secrets: Mapping[str, str], *, secrets_dir: Path | None = None
) -> list[Path]:
    """Write every declared secret to `<secrets_dir>/<name>` (0600). Returns paths."""
    target = secrets_dir or DEFAULT_SECRETS_DIR
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in SECRET_NAMES:
        path = target / name
        path.write_text(secrets.get(name, ""), encoding="utf-8")
        path.chmod(0o600)
        written.append(path)
    return written


def cleanup(*, secrets_dir: Path | None = None) -> None:
    """Remove rendered plaintext secret files (best effort)."""
    target = secrets_dir or DEFAULT_SECRETS_DIR
    for name in SECRET_NAMES:
        with contextlib.suppress(OSError):
            (target / name).unlink(missing_ok=True)
