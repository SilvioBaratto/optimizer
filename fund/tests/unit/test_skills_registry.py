"""Phase 6 B2 — per-agent skill registry ↔ filesystem agreement.

Asserts the registry (``fund.agents.skills``) and the on-disk skill tree are an
exact bijection, and that ``skill_paths`` resolves. Static: no model, no DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fund.agents.skills import (
    SKILLS_BY_AGENT,
    SKILLS_DIR,
    STAGED_SKILLS_ROOT,
    skill_paths,
    skill_sources,
)

# Directories under skills/ that actually hold a SKILL.md.
_ON_DISK = {
    d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
}


def test_skills_dir_points_at_the_resource_tree():
    assert SKILLS_DIR.is_dir()
    assert SKILLS_DIR.name == "skills"
    # Path resolves under the fund package, not the top-level repo.
    assert SKILLS_DIR.parent.name == "fund"


def test_registry_is_a_bijection_with_on_disk_skills():
    assigned = [name for names in SKILLS_BY_AGENT.values() for name in names]
    # No skill assigned to two roles (would double-load).
    assert len(assigned) == len(set(assigned)), "a skill is assigned to >1 role"
    # Every assigned skill exists on disk, and every on-disk skill is assigned.
    assert set(assigned) == _ON_DISK


def test_expected_eight_skills():
    assert len(_ON_DISK) == 8
    assert "mifid-profiling" in _ON_DISK


@pytest.mark.parametrize("role", sorted(SKILLS_BY_AGENT))
def test_skill_paths_resolve_to_existing_dirs(role: str):
    paths = skill_paths(role)
    assert paths, f"role {role} has no skills"
    for p in paths:
        assert Path(p).is_dir()


def test_skill_paths_unknown_role_raises():
    with pytest.raises(KeyError):
        skill_paths("no-such-role")


# --- Phase 7 (Risk R1): backend-root-relative skill sources --------------------
# deepagents' SkillsMiddleware reads sources through the (virtual_mode) backend, so
# a skill source MUST be root-relative — an absolute path outside the staged root
# is blocked. ``skill_sources`` returns the per-role staged source dir.


@pytest.mark.parametrize("role", sorted(SKILLS_BY_AGENT))
def test_skill_sources_are_root_relative_per_role(role: str):
    sources = skill_sources(role)
    assert sources == [f"{STAGED_SKILLS_ROOT}/{role}"]
    for src in sources:
        # Root-relative: no drive/anchor, never an absolute path.
        assert not Path(src).is_absolute()
        assert src.startswith(f"{STAGED_SKILLS_ROOT}/")


def test_skill_sources_unknown_role_raises():
    with pytest.raises(KeyError):
        skill_sources("no-such-role")


def test_staged_skills_root_is_a_bare_relative_name():
    # The staged skills subdir the backend populates; a plain name, not a path.
    assert STAGED_SKILLS_ROOT == "skills"
    assert "/" not in STAGED_SKILLS_ROOT
    assert not Path(STAGED_SKILLS_ROOT).is_absolute()
