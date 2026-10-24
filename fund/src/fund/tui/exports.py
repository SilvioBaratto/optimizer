"""Where the TUI writes exports that hold client data.

Audit CSVs, Ask logs and daily briefs carry capital, mandates and adviser free text,
so they go to one dedicated directory instead of the working directory: that is the
container's writable layer (lost on recreate) or, on a host run, the repo root where
a stray ``git add -A`` would publish them. Stdlib only, so the TUI import stays light.
"""

from __future__ import annotations

import os
from pathlib import Path

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
