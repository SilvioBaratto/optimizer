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
runtime workdir is regenerated from both on every call so it never drifts and is
never hand-edited. ``fund/.runtime/`` is gitignored — nothing staged here is committed.

``deepagents`` is imported **inside** :func:`build_backend` so a bare
``import fund.agents.backend`` drags in no agent stack and needs no environment
(mirrors :mod:`fund.agents.model`). :func:`stage_theory` is pure stdlib.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from fund.agents.skills import SKILLS_BY_AGENT, SKILLS_DIR, STAGED_SKILLS_ROOT
from fund.config import FundConfig, settings

if TYPE_CHECKING:
    from deepagents.backends import FilesystemBackend

__all__ = ["build_backend", "stage_theory"]

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


def _stage_skills(root: Path, skills_src: Path) -> None:
    """Regenerate the per-role skills tree under ``<root>/skills``.

    Copies each registered skill dir (``<skills_src>/<skill-name>``) into its
    owning role's folder (``<root>/skills/<role>/<skill-name>``) so a single
    root-relative source (``skills/<role>``) loads only that role's skills through
    the backend (Risk R1). The whole ``skills`` subtree is wiped first so a
    hand-edited or stale staged skill never survives (drift-free).
    """
    if not skills_src.is_dir():
        raise FileNotFoundError(
            f"Canonical skills tree not found at {skills_src}; cannot stage skills."
        )
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
    """Regenerate the runtime agent workdir and return its root path.

    When ``config.preload_theory`` is set, the canonical ``optimizerwiki/``
    tree (``source``) is copied fresh under ``<dest>/optimizer-theory`` — wiping
    any prior copy first so the result never drifts from canonical and the
    ``optimizer-theory/docs/…`` citation prefix is preserved verbatim — and the
    per-role skills tree is staged under ``<dest>/skills`` (Risk R1). When preload
    is disabled, the workdir is created empty and left untouched.

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
    """
    from deepagents.backends import FilesystemBackend

    return FilesystemBackend(
        root_dir=stage_theory(config),
        virtual_mode=config.agent_virtual_mode,
    )
