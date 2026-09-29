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
from pathlib import Path

import userpath

from app.setup import config_file

_logger = logging.getLogger(__name__)

_BIN_DIR = Path.home() / ".local" / "bin"

# The env var the front doors (setup.sh/.ps1/.cmd) export so an out-of-repo tool-venv
# install of `portopt` (uv tool install --from .) can still locate the checkout.
_REPO_ENV = "OPTIMIZER_REPO"


class PathInstallError(RuntimeError):
    """Raised when the launcher cannot be installed onto the user's PATH."""


def _repo_root() -> Path:
    """Return the repo root (path_install.py -> setup -> app -> ingestion -> root)."""
    return Path(__file__).resolve().parents[3]


def _is_repo(path: Path) -> bool:
    return (path / "docker-compose.yml").is_file()


def resolve_repo(repo: Path | None = None) -> Path:
    """Locate the repo checkout, in precedence order.

    ``uv tool install --from .`` installs ``portopt`` into an out-of-repo tool venv,
    so ``_repo_root()`` (derived from ``__file__``) no longer points at the checkout.
    Resolution therefore prefers, in order: an explicit ``repo`` argument, the
    ``OPTIMIZER_REPO`` env the front doors export, the ``repo_path`` persisted to
    ``~/.portopt/config.toml`` (the moved-repo self-heal), then the ``__file__`` root
    (correct only for an in-repo editable install).

    Args:
        repo: An explicit checkout path that short-circuits resolution.

    Returns:
        The best available repo root; falls back to the ``__file__``-derived root.
    """
    if repo is not None:
        return repo
    env = os.environ.get(_REPO_ENV)
    if env and _is_repo(Path(env)):
        return Path(env)
    configured = config_file.load_config().get("repo_path")
    if configured and _is_repo(Path(str(configured))):
        return Path(str(configured))
    return _repo_root()


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
    """Write ``bin_dir/optimizer.cmd`` with the repo path baked in (not ``%~dp0``)."""
    template = (repo / "scripts" / "optimizer.cmd").read_text(encoding="utf-8")
    launcher = bin_dir / "optimizer.cmd"
    launcher.write_text(template.replace("%~dp0..", str(repo)), encoding="utf-8")
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
