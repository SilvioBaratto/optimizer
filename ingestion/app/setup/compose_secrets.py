"""Render decrypted secrets into Docker-compose secret files (SPEC D6).

`portopt start` decrypts `~/.portopt/secrets.enc` and calls `render` to write
each value to `<repo>/secrets/<name>`, which docker-compose mounts at
`/run/secrets/<name>`. The directory is owner-only (0700) and holds the files at
0644 so the containers' non-root user can read them. Every declared secret is
written — empty when unconfigured — so `docker compose up` never fails on a
missing file. `portopt stop` calls `cleanup` for the secrets no still-running
service mounts (the fund's LLM keys); `portopt stop --all` and a failed setup
remove every plaintext file.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Iterable, Mapping
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

# The secrets the scheduler mounts (docker-compose.yml `scheduler.secrets`; parity-
# tested). It restarts on its own (`restart: unless-stopped`), so these files must
# outlive a fund-only stop; every other secret is consumed by the fund alone.
DAEMON_SECRET_NAMES = (
    "trading_212_api_key",
    "trading_212_secret_key",
    "fred_api_key",
)
FUND_ONLY_SECRET_NAMES = tuple(
    name for name in SECRET_NAMES if name not in DAEMON_SECRET_NAMES
)

# Relative to the compose project directory (the repo root), never the cwd: callers
# join it onto the checkout so compose mounts exactly what was rendered.
DEFAULT_SECRETS_DIR = Path("secrets")


def _restrict_dir(target: Path) -> bool:
    """Create ``target`` owner-only; report whether that actually gates access.

    A git checkout creates ``secrets/`` 0755, and some filesystems (exFAT, drvfs
    mounts) silently ignore a directory chmod, so the resulting mode is re-read
    rather than assumed.
    """
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        target.chmod(0o700)
    return target.stat().st_mode & 0o077 == 0


def _write_secret(path: Path, value: str, mode: int) -> None:
    """Write ``value`` to ``path``, created with ``mode`` rather than tightened later.

    A symlink at ``path`` is replaced, never followed, and ``O_NOFOLLOW`` turns a
    link swapped in after that check into an error instead of a write through it.
    The trailing chmod only matters for a umask that narrowed the creation mode.
    """
    if path.is_symlink():
        path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        # A leftover keeps the mode it had, possibly looser than *mode*: settle the
        # final mode before the value lands, in place so a running container's bind
        # mount keeps its inode.
        if hasattr(os, "fchmod"):  # POSIX only
            os.fchmod(fd, mode)
        handle.write(value)
    path.chmod(mode)


def render(secrets: Mapping[str, str], *, secrets_dir: Path) -> list[Path]:
    """Write every declared secret to `<secrets_dir>/<name>`. Returns the paths.

    ``secrets_dir`` has no default on purpose: a cwd-relative one would plant the
    keys wherever the operator happened to run, not where compose mounts them.

    The directory is made owner-only (0700). Files are then 0644 so the containers'
    uid-1000 user can read them: a compose `file:` secret is a plain bind mount, so
    the host owner and mode apply inside the container, and a 0600 file owned by a
    different host uid is unreadable there. When the directory cannot be made
    owner-only the files stay 0600 rather than exposing the keys to other users.
    """
    file_mode = 0o644 if _restrict_dir(secrets_dir) else 0o600
    written: list[Path] = []
    for name in SECRET_NAMES:
        path = secrets_dir / name
        # Docker creates a missing file-mount source as a directory when the stack
        # comes up before the secret is rendered, and a later write then fails with a
        # directory error. Clear any such placeholder so the value lands as a file.
        if path.is_dir():
            shutil.rmtree(path)
        _write_secret(path, secrets.get(name, ""), file_mode)
        written.append(path)
    return written


def cleanup(*, secrets_dir: Path, names: Iterable[str] | None = None) -> None:
    """Remove rendered plaintext secret files (best effort).

    Args:
        secrets_dir: Directory the files were rendered into; required for the same
            reason as in :func:`render`.
        names: Secrets to remove; every declared secret when omitted.
    """
    for name in SECRET_NAMES if names is None else names:
        with contextlib.suppress(OSError):
            (secrets_dir / name).unlink(missing_ok=True)
