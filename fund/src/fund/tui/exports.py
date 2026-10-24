"""Where the TUI writes exports that hold client data.

Audit CSVs, Ask logs and daily briefs carry capital, mandates and adviser free text,
so they go to one dedicated directory instead of the working directory: that is the
container's writable layer (lost on recreate) or, on a host run, the repo root where
a stray ``git add -A`` would publish them. Stdlib only, so the TUI import stays light.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

EXPORT_DIR_ENV = "FUND_EXPORT_DIR"


def export_directory() -> Path:
    """The directory exports are written to.

    ``FUND_EXPORT_DIR`` when set (the compose service points it at a mounted host
    directory); otherwise ``fund/exports`` under ``XDG_DATA_HOME`` or
    ``~/.local/share``.
    """
    configured = os.environ.get(EXPORT_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "fund" / "exports"


def export_path(name: str) -> Path:
    """The absolute path for an export called ``name``, creating the directory.

    The directory is created owner-only. An existing one keeps whatever mode its
    owner gave it.

    Raises:
        OSError: The directory cannot be created.
    """
    directory = export_directory().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory / name


@contextmanager
def open_private(path: str | Path, *, newline: str | None = None) -> Iterator[IO[str]]:
    """Open ``path`` for writing as an owner-only UTF-8 text file.

    The mode is forced to 0600 whatever the umask or the file's previous mode, and a
    symlink at ``path`` is refused rather than followed, so a link planted in a shared
    directory cannot redirect the write. An existing file is truncated, which keeps a
    re-export idempotent.

    Args:
        path: The file to write.
        newline: Passed to the text wrapper (``""`` for ``csv`` writers).

    Raises:
        OSError: The file cannot be opened, or ``path`` is a symlink.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        handle = os.fdopen(fd, "w", encoding="utf-8", newline=newline)
    except BaseException:
        os.close(fd)
        raise
    with handle:
        yield handle
