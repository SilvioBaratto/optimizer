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
# The `portopt setup` wizard also configures the fund's switchable-LLM backend
# (SPEC "Switchable LLM Backends"): each provider's auth key is a file-based
# docker secret the `fund` service reads via `<NAME>_FILE`. These are config
# identifiers, not an LLM-stack import — the boundary (ingestion ⊬ agent stack /
# optimizer) is unchanged. `aws` is deliberately absent: it authenticates via
# IAM/region, not a secret file.
SECRET_NAMES = (
    # ingestion daemon secrets (Trading212 universe + FRED macro)
    "trading_212_api_key",
    "trading_212_secret_key",
    "fred_api_key",
    # fund LLM-provider auth keys (name-agree with fund.config._read_secret)
    "ollama_api_key",
    "openrouter_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "groq_api_key",
    "nvidia_api_key",
    "huggingfacehub_api_token",
    "azure_openai_api_key",
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
