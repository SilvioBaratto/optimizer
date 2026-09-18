"""Per-agent skill registry (SPEC Phase 6 §4c).

The single source of truth for *which* deepagents skill each fund role gets. Phase
7 consumes this to wire ``create_deep_agent(..., skills=skill_paths(role))`` — and
because deepagents subagents do **not** inherit skills, the assignment must be
explicit per role.

Skills themselves are ``SKILL.md`` (+ ``reference.md``) directories under
``fund/src/fund/skills/``; they are text resources loaded on demand by the agent
backend, never imported as Python. This module only maps role → directory path.

Cheap to import (stdlib only): reading the registry in a test drags in no model,
no DB, no optimizer.
"""

from __future__ import annotations

from pathlib import Path

# Absolute path to the skills resource tree (``fund/src/fund/skills``).
SKILLS_DIR: Path = Path(__file__).resolve().parent.parent / "skills"

# Backend-root-relative subdir the Phase-7 backend stages the per-role skills into
# (``fund.agents.backend.stage_theory``). deepagents' ``SkillsMiddleware`` reads
# skill sources **through the backend**, and the backend runs in ``virtual_mode``
# (blocks any absolute path outside its root) — so a skill source MUST be
# root-relative, never the absolute :func:`skill_paths` (Risk R1, verified). The
# staged layout is per-role (``skills/<role>/<skill-name>/SKILL.md``) so a single
# source dir per role loads ONLY that role's skills (the loader scans a source for
# its child skill dirs).
STAGED_SKILLS_ROOT = "skills"

# Canonical role keys — Phase 7 subagents MUST adopt these names verbatim.
# Value = the skill directory names that role loads, in load order.
SKILLS_BY_AGENT: dict[str, tuple[str, ...]] = {
    "orchestrator": ("fund-orchestration",),
    "profiler": ("mifid-profiling",),
    "economist": ("macro-regime-read", "views-construction"),
    "allocator": ("universe-preselection", "optimization-objective-map"),
    "risk": ("risk-limits-check",),
    "executor": ("rebalancing-execution",),
}


def skill_paths(agent: str) -> list[str]:
    """Absolute skill-directory paths for ``agent`` (in load order).

    Args:
        agent: One of the canonical role keys in :data:`SKILLS_BY_AGENT`.

    Returns:
        Absolute directory paths, ready to pass to ``create_deep_agent(skills=…)``.

    Raises:
        KeyError: If ``agent`` is not a known role.
    """
    return [str(SKILLS_DIR / name) for name in SKILLS_BY_AGENT[agent]]


def skill_sources(agent: str) -> list[str]:
    """Backend-root-relative skill *source* dirs for ``agent`` (Phase-7 wiring).

    Unlike :func:`skill_paths` (absolute, for direct filesystem use), these are the
    root-relative sources handed to ``create_deep_agent(skills=…)`` /
    ``SubAgent["skills"]``: deepagents loads them through the ``virtual_mode``
    backend, which blocks absolute paths outside its root (Risk R1). Returns the
    single per-role source dir (``skills/<role>``) the backend stages the role's
    skills under; deepagents scans it for the role's child skill dirs.

    Args:
        agent: One of the canonical role keys in :data:`SKILLS_BY_AGENT`.

    Returns:
        A one-element list with the role's staged source dir, root-relative.

    Raises:
        KeyError: If ``agent`` is not a known role.
    """
    if agent not in SKILLS_BY_AGENT:
        raise KeyError(agent)
    return [f"{STAGED_SKILLS_ROOT}/{agent}"]


__all__ = [
    "SKILLS_BY_AGENT",
    "SKILLS_DIR",
    "STAGED_SKILLS_ROOT",
    "skill_paths",
    "skill_sources",
]
