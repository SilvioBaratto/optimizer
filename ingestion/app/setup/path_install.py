r"""Install the ``optimizer`` launcher onto the user's PATH (SPEC D9, task T12).

POSIX writes a *symlink* ``~/.local/bin/optimizer`` -> ``scripts/optimizer`` (a
symlink, not a copy, so a ``git pull`` updates the launcher in place). Windows writes
``%USERPROFILE%\.local\bin\optimizer.cmd`` with the repo path baked in and puts that
directory on the User PATH via ``userpath`` (never ``setx`` — it truncates PATH at
1024 chars). The step is idempotent: it skips the PATH edit when the directory already
resolves and repoints a stale symlink instead of erroring.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import userpath

from app.setup import config_file

_logger = logging.getLogger(__name__)

_BIN_DIR = Path.home() / ".local" / "bin"

# cmd.exe decodes a batch file with the console's code page (OEM by default), never
# UTF-8, so the baked repo path has to be written in that encoding to survive a
# non-ASCII directory name. "oem" is a Windows-only codec.
_CMD_ENCODING = "oem" if sys.platform == "win32" else "utf-8"

# The env var the front doors (setup.sh/.ps1/.cmd) export so an out-of-repo tool-venv
# install of `portopt` (uv tool install --from .) can still locate the checkout.
_REPO_ENV = "OPTIMIZER_REPO"


class PathInstallError(RuntimeError):
    """Raised when the launcher cannot be installed onto the user's PATH."""


def _repo_root() -> Path:
    """Return the repo root (path_install.py -> setup -> app -> ingestion -> root)."""
    return Path(__file__).resolve().parents[3]


def is_repo(path: Path) -> bool:
    """Whether ``path`` is an optimizer checkout (it owns ``docker-compose.yml``)."""
    return (path / "docker-compose.yml").is_file()


def resolve_repo(repo: Path | None = None) -> Path:
    """Locate the repo checkout, in precedence order.

    ``uv tool install --from .`` installs ``portopt`` into an out-of-repo tool venv,
    so ``_repo_root()`` (derived from ``__file__``) no longer points at the checkout.
    Resolution therefore prefers, in order: an explicit ``repo`` argument, the
    ``OPTIMIZER_REPO`` env the front doors export, the current directory or one of
    its parents (a direct ``portopt`` run from inside a checkout, so a worktree is
    never overridden by the other checkout recorded in config), the ``repo_path``
    persisted to ``~/.portopt/config.toml`` (the moved-repo self-heal, for a run from
    outside any checkout), then the ``__file__`` root (correct only for an in-repo
    editable install).

    Args:
        repo: An explicit checkout path that short-circuits resolution.

    Returns:
        The best available repo root; falls back to the ``__file__``-derived root.
    """
    if repo is not None:
        return repo
    env = os.environ.get(_REPO_ENV)
    if env and is_repo(Path(env)):
        return Path(env)
    cwd = Path.cwd()
    for base in (cwd, *cwd.parents):
        if is_repo(base):
            return base
    configured = config_file.load_config().get("repo_path")
    if configured and is_repo(Path(str(configured))):
        return Path(str(configured))
    return _repo_root()


def locate_repo() -> Path | None:
    """Return the checkout that owns ``docker-compose.yml``, or ``None`` without one.

    Unlike ``resolve_repo`` this never hands back a path that has no compose file: the
    ``__file__``-derived fallback points into site-packages for a ``uv tool install``
    with no clone, where compose can only fail with "no configuration file provided".
    """
    repo = resolve_repo()
    return repo if is_repo(repo) else None


def install_launcher(*, repo: Path | None = None, bin_dir: Path | None = None) -> Path:
    """Install the ``optimizer`` launcher onto the user's PATH.

    Chooses the POSIX (symlink) or Windows (``.cmd`` + User PATH) strategy by
    ``os.name`` and is idempotent — a re-run repoints a stale symlink and skips the
    PATH edit when ``bin_dir`` already resolves. The resolved checkout is persisted to
    config ``repo_path`` so the launcher self-heals a moved repo on a later run.

    Args:
        repo: The checkout to point the launcher at; resolved via ``resolve_repo()``
            when omitted.
        bin_dir: Directory to install into; defaults to ``~/.local/bin`` (which uv
            already exposes on PATH on POSIX). Created if absent.

    Returns:
        The created symlink (POSIX) or ``.cmd`` file (Windows).

    Raises:
        PathInstallError: If the symlink/``.cmd`` cannot be written or the PATH edit
            fails. Callers treat this as non-fatal — the launcher install is the last,
            least-critical setup step.
    """
    try:
        target_dir = bin_dir or _BIN_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved = resolve_repo(repo)
        _persist_repo_path(resolved)
        installed = (
            _install_windows(target_dir, resolved)
            if os.name == "nt"
            else _install_posix(target_dir, resolved)
        )
        _ensure_on_path(target_dir)
    except Exception as exc:
        # Deliberately broad: this is the last, non-critical setup step, and userpath
        # can raise beyond OSError (e.g. UnicodeDecodeError on a malformed Windows PATH
        # value). Any failure must degrade to a warning at the call site, never a crash.
        raise PathInstallError(str(exc)) from exc
    return installed


def _persist_repo_path(repo: Path) -> None:
    """Record ``repo`` as config ``repo_path`` for the launcher's moved-repo self-heal.

    Best-effort: a config-write failure must not fail the launcher install itself.
    """
    try:
        config_file.update_config({"repo_path": str(repo)})
    except (OSError, ValueError) as exc:
        _logger.warning("Could not persist repo_path to config: %s", exc)


def _install_posix(bin_dir: Path, repo: Path) -> Path:
    """Symlink ``bin_dir/optimizer`` at the repo launcher, repointing a stale link."""
    link = bin_dir / "optimizer"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(repo / "scripts" / "optimizer")
    return link


def _install_windows(bin_dir: Path, repo: Path) -> Path:
    """Write ``bin_dir/optimizer.cmd`` with the repo path baked in (not ``%~dp0``).

    Raises:
        PathInstallError: If the repo path has characters the console code page
            cannot represent. The existing launcher is left untouched.
    """
    template = (repo / "scripts" / "optimizer.cmd").read_text(encoding="utf-8")
    launcher = bin_dir / "optimizer.cmd"
    baked = template.replace("%~dp0..", str(repo)).replace("\n", "\r\n")
    try:
        payload = baked.encode(_CMD_ENCODING)
    except UnicodeEncodeError as exc:
        raise PathInstallError(
            f"{repo} has characters the console code page ({_CMD_ENCODING}) cannot "
            "represent; move the checkout to an ASCII-only path."
        ) from exc
    launcher.write_bytes(payload)
    return launcher


def _ensure_on_path(bin_dir: Path) -> None:
    """Add ``bin_dir`` to the User PATH via userpath unless it already resolves.

    A userpath.append that reports failure (returns falsey) is a warning, not an
    error: the launcher file is still installed and the user can add the dir manually.
    """
    location = str(bin_dir)
    if userpath.in_current_path(location) or userpath.in_new_path(location):
        return
    if not userpath.append(location, "optimizer"):
        _logger.warning(
            "Could not add %s to PATH automatically; add it to PATH manually.",
            location,
        )
