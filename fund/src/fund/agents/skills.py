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


__all__ = ["SKILLS_BY_AGENT", "SKILLS_DIR", "skill_paths"]
