"""Theory- and skills-staged virtual filesystem backend (SPEC D2, Risk R1).

deepagents reads and writes files through a pluggable *backend*. Phase 7 roots
that backend at a runtime workdir under ``fund/`` and — when
``config.preload_theory`` is set — pre-populates it with the agent's read-only
knowledge:

* the canonical ``optimizerwiki/`` monograph, so both knowledge layers resolve as
  the agent reads: the curated ``optimizer-theory/openwiki/`` map (``index.md`` +
  ``quickstart.md`` routing + per-topic pages) the consultation protocol steers
  every role to first, and the raw ``optimizer-theory/docs/NN_*.md`` chapters the
  Phase-6 skills cite by ``NN:line``;
* the per-role skills tree (``skills/<role>/<skill-name>/SKILL.md``). deepagents'
  ``SkillsMiddleware`` loads skills **through this backend** (no direct filesystem
  access), and the backend runs in ``virtual_mode`` — which blocks any absolute
  path outside its root. So the skills every role loads must live *under* the
  staged root and be addressed by the root-relative
  :func:`fund.agents.skills.skill_sources` (Risk R1). The per-role layout lets one
  source dir (``skills/<role>``) load only that role's skills.

The theory tree is the *committed* in-repo ``optimizerwiki/`` canonical source;
the skills tree is the *committed* ``fund/src/fund/skills`` resource tree. The
runtime workdir is reconciled against both on every call so it never drifts and is
never hand-edited: a workdir that already matches canonical is left untouched, and
only a stale or hand-edited one is rewritten. Every agent build stages the one shared
workdir while other runs (worker sweep, TUI, CLI) read it lazily through their own
backend, so the reconcile runs under a cross-process lock and never wipes a tree that
is already correct. ``fund/.runtime/`` is gitignored — nothing staged here is committed.

``deepagents`` is imported **inside** :func:`build_backend` so a bare
``import fund.agents.backend`` drags in no agent stack and needs no environment
(mirrors :mod:`fund.agents.model`). :func:`stage_theory` is pure stdlib.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING

from fund.agents.skills import SKILLS_BY_AGENT, SKILLS_DIR, STAGED_SKILLS_ROOT
from fund.config import FundConfig, settings

if TYPE_CHECKING:
    from deepagents import FilesystemPermission
    from deepagents.backends import FilesystemBackend

__all__ = ["build_backend", "read_only_permissions", "stage_theory"]

# ``…/fund/src/fund/agents/backend.py`` → parents[3] = ``fund/``, .parent = repo root.
_FUND_DIR: Path = Path(__file__).resolve().parents[3]
_REPO_ROOT: Path = _FUND_DIR.parent

# Canonical (committed) in-repo theory source and the regenerated, gitignored
# runtime workdir it is staged into.
_THEORY_SRC: Path = _REPO_ROOT / "optimizerwiki"
_AGENT_FS_ROOT: Path = _FUND_DIR / ".runtime" / "agent-fs"

# Canonical (committed) skills source — the per-role staging reads each skill dir
# from here (:data:`fund.agents.skills.SKILLS_DIR`).
_SKILLS_SRC: Path = SKILLS_DIR

# Fixed prefix the skill citations hardcode; the staged subtree keeps this name
# regardless of the source directory's own name.
_STAGE_SUBDIR = "optimizer-theory"

# Editor/VCS/CI cruft never copied into the staged tree (the OpenWiki repo carries
# a ``.github`` workflow dir and ``.obsidian`` vault config that are not theory).
_STAGE_IGNORE = shutil.ignore_patterns(".git", ".github", ".obsidian")

# Lives in the workdir root, which staging never wipes (only the two subtrees below
# it), so the lock survives every restage.
_LOCK_NAME = ".stage.lock"

_DIR_MARK = "<dir>"

if sys.platform == "win32":
    import msvcrt

    def _acquire(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)

    def _release(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    # ``flock`` (not ``lockf``): the lock belongs to the open file description, so
    # two threads of one process that each open the file exclude each other too.
    def _acquire(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def _release(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _staging_lock(root: Path) -> Iterator[None]:
    """Serialise stagers of one workdir across threads and processes."""
    with (root / _LOCK_NAME).open("a+b") as handle:
        _acquire(handle)
        try:
            yield
        finally:
            _release(handle)


def _layout(
    top: Path,
    prefix: str,
    ignore: Callable[[str, list[str]], set[str]] | None,
) -> dict[str, str]:
    """Map every dir and file under ``top`` to ``<dir>`` or its content digest.

    Keys are ``prefix``-relative POSIX paths. ``ignore`` mirrors the ``copytree``
    filter so a source layout compares equal to the staged copy it produces; the
    staged side passes ``None`` so a stray cruft entry still counts as drift.
    """
    entries = {prefix: _DIR_MARK}
    for dirpath, dirnames, filenames in os.walk(top, followlinks=True):
        skipped = ignore(dirpath, [*dirnames, *filenames]) if ignore else set()
        dirnames[:] = [name for name in dirnames if name not in skipped]
        base = Path(prefix) / Path(dirpath).relative_to(top)
        for name in dirnames:
            entries[(base / name).as_posix()] = _DIR_MARK
        for name in filenames:
            if name not in skipped:
                digest = hashlib.sha256((Path(dirpath) / name).read_bytes())
                entries[(base / name).as_posix()] = digest.hexdigest()
    return entries


def _expected_layout(src: Path, skills_src: Path) -> dict[str, str]:
    """The workdir layout a fresh staging of ``src`` and ``skills_src`` produces."""
    layout = _layout(src, _STAGE_SUBDIR, _STAGE_IGNORE)
    for role, names in SKILLS_BY_AGENT.items():
        for name in names:
            # ``copytree`` creates the missing parent dirs of each skill copy.
            layout[STAGED_SKILLS_ROOT] = _DIR_MARK
            layout[f"{STAGED_SKILLS_ROOT}/{role}"] = _DIR_MARK
            layout.update(
                _layout(
                    skills_src / name,
                    f"{STAGED_SKILLS_ROOT}/{role}/{name}",
                    _STAGE_IGNORE,
                )
            )
    return layout


def _staged_layout(root: Path) -> dict[str, str]:
    """The layout currently on disk under the two staged subtrees of ``root``."""
    layout: dict[str, str] = {}
    for top in (_STAGE_SUBDIR, STAGED_SKILLS_ROOT):
        if (root / top).is_dir():
            layout.update(_layout(root / top, top, None))
    return layout


def _stage_skills(root: Path, skills_src: Path) -> None:
    """Regenerate the per-role skills tree under ``<root>/skills``.

    Copies each registered skill dir (``<skills_src>/<skill-name>``) into its
    owning role's folder (``<root>/skills/<role>/<skill-name>``) so a single
    root-relative source (``skills/<role>``) loads only that role's skills through
    the backend (Risk R1). The whole ``skills`` subtree is wiped first so a
    hand-edited or stale staged skill never survives (drift-free).
    """
    staged_root = root / STAGED_SKILLS_ROOT
    if staged_root.exists():
        shutil.rmtree(staged_root)
    for role, names in SKILLS_BY_AGENT.items():
        for name in names:
            shutil.copytree(
                skills_src / name,
                staged_root / role / name,
                ignore=_STAGE_IGNORE,
            )


def stage_theory(
    config: FundConfig = settings,
    *,
    source: Path | None = None,
    dest: Path | None = None,
    skills_source: Path | None = None,
) -> Path:
    """Reconcile the runtime agent workdir with canonical and return its root path.

    When ``config.preload_theory`` is set, the canonical ``optimizerwiki/``
    tree (``source``) is copied under ``<dest>/optimizer-theory`` — the
    ``optimizer-theory/docs/…`` citation prefix is preserved verbatim — and the
    per-role skills tree is staged under ``<dest>/skills`` (Risk R1). A workdir
    that already matches canonical byte for byte is left untouched, because other
    live runs may be reading it; one that drifted (hand edit, stray file, changed
    canonical, interrupted copy) has both subtrees wiped and recopied. The whole
    reconcile runs under an exclusive cross-process lock. When preload is
    disabled, the workdir is created empty and left untouched.

    Args:
        config: Frozen run config; ``preload_theory`` gates the copy.
        source: Canonical theory tree. Defaults to the repo ``optimizerwiki/``.
        dest: Runtime workdir root. Defaults to ``fund/.runtime/agent-fs``.
        skills_source: Canonical skills tree. Defaults to the committed
            ``fund/src/fund/skills`` (:data:`fund.agents.skills.SKILLS_DIR`).

    Returns:
        The workdir root (``dest``) — the ``root_dir`` a ``FilesystemBackend``
        is rooted at.

    Raises:
        FileNotFoundError: If preloading is on but ``source`` (or the skills
            source) does not exist.
        OSError: If the staging lock cannot be acquired.
    """
    src = _THEORY_SRC if source is None else source
    root = _AGENT_FS_ROOT if dest is None else dest
    skills_src = _SKILLS_SRC if skills_source is None else skills_source

    root.mkdir(parents=True, exist_ok=True)

    if not config.preload_theory:
        return root

    if not src.is_dir():
        raise FileNotFoundError(
            f"Canonical optimizerwiki tree not found at {src}; cannot preload "
            "(set preload_theory=False to run without it)."
        )

    if not skills_src.is_dir():
        raise FileNotFoundError(
            f"Canonical skills tree not found at {skills_src}; cannot stage skills."
        )

    with _staging_lock(root):
        if _staged_layout(root) == _expected_layout(src, skills_src):
            return root
        staged = root / _STAGE_SUBDIR
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(src, staged, ignore=_STAGE_IGNORE)
        _stage_skills(root, skills_src)

    return root


def build_backend(config: FundConfig = settings) -> FilesystemBackend:
    """Build the ``virtual_mode`` ``FilesystemBackend`` rooted at the staged workdir.

    ``deepagents`` is imported here (not at module top) so importing this module
    needs no agent stack; ``virtual_mode`` blocks path traversal outside the root.

    Args:
        config: Frozen run config; ``preload_theory`` gates theory/skills staging
            and ``agent_virtual_mode`` controls path-traversal enforcement.

    Returns:
        A ``FilesystemBackend`` rooted at the staged workdir, ready to pass to a
        deepagents role constructor.
    """
    from deepagents.backends import FilesystemBackend

    return FilesystemBackend(
        root_dir=stage_theory(config),
        virtual_mode=config.agent_virtual_mode,
    )


def read_only_permissions() -> list[FilesystemPermission]:
    """Filesystem rules that make the whole agent workdir read-only for the roles.

    The theory and the per-role skills are reference material every run (and every
    role) reads as instructions, and the workdir root is shared across runs and
    portfolios, so a write by any role (a prompt-injected economist, the
    auto-added general-purpose subagent) would poison later reads or persist across
    runs. A prompt asking agents not to write is not a control; these rules deny
    ``write_file`` / ``edit_file`` / ``delete`` at the tool level. Passed as
    ``create_deep_agent(permissions=…)``, which subagents inherit unless they set
    their own. The roles have no legitimate write: they exchange data through tool
    results and the audit trail, and deepagents' own offloading (summarised history,
    large tool results) writes through the backend, which these tool-level rules do
    not gate. ``deepagents`` is imported here to keep the module import light.
    """
    from deepagents import FilesystemPermission

    return [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]
