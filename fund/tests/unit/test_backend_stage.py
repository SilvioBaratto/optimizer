"""Task 2 — ``fund.agents.backend`` theory-staged virtual backend (unit slice).

Staging is pure stdlib (``shutil``/``pathlib``): it regenerates a runtime workdir
from the canonical (gitignored) ``optimizer-theory/`` tree, preserving the
``optimizer-theory/docs/…`` citation prefix so the Phase-6 skills resolve
unchanged. Because the canonical tree is gitignored (absent in CI), every test
injects a temp ``source``/``dest`` — nothing touches the real ``fund/.runtime``.
``build_backend`` roots a ``virtual_mode`` deepagents ``FilesystemBackend`` at the
staged workdir; construction only, no agent invocation and no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fund.agents import backend
from fund.config import FundConfig

_PRELOAD = FundConfig(preload_theory=True)
_NO_PRELOAD = FundConfig(preload_theory=False)


def _fake_theory(root: Path) -> Path:
    """Build a minimal ``optimizer-theory``-shaped source tree under ``root``."""
    src = root / "optimizer-theory"
    docs = src / "docs"
    (docs / "architecture").mkdir(parents=True)
    (docs / "00 Introduzione e scopo.md").write_text("intro", encoding="utf-8")
    (docs / "architecture" / "OPTIMIZER-OBLIGATIONS.md").write_text(
        "obligations", encoding="utf-8"
    )
    return src


def _fake_skills(root: Path) -> Path:
    """Build a minimal skills source tree (one ``SKILL.md`` per registered skill)."""
    from fund.agents.skills import SKILLS_BY_AGENT

    src = root / "skills"
    src.mkdir(parents=True)
    for names in SKILLS_BY_AGENT.values():
        for name in names:
            skill = src / name
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                f"---\nname: {name}\n---\n{name} body", encoding="utf-8"
            )
            (skill / "reference.md").write_text("ref", encoding="utf-8")
    return src


def _stage(config, tmp_path):
    """Stage a fake theory + fake skills tree into a temp workdir; return the root.

    The fake source trees are built once per ``tmp_path`` so re-staging (the
    idempotency check) reuses the same canonical sources.
    """
    theory = tmp_path / "canonical" / "optimizer-theory"
    skills = tmp_path / "canonical-skills" / "skills"
    if not theory.exists():
        _fake_theory(tmp_path / "canonical")
    if not skills.exists():
        _fake_skills(tmp_path / "canonical-skills")
    dest = tmp_path / "agent-fs"
    root = backend.stage_theory(config, source=theory, dest=dest, skills_source=skills)
    return root, skills


def test_import_needs_no_env_and_defers_deepagents_to_build_time():
    # Importing the module (done at file top) must not need env or the agent
    # stack; stage_theory is stdlib, build_backend imports deepagents lazily.
    assert callable(backend.stage_theory)
    assert callable(backend.build_backend)


def test_stage_theory_returns_workdir_and_preserves_citation_prefix(tmp_path):
    src = _fake_theory(tmp_path / "canonical")
    dest = tmp_path / "agent-fs"

    root = backend.stage_theory(_PRELOAD, source=src, dest=dest)

    assert root == dest
    obligations = (
        root / "optimizer-theory" / "docs" / "architecture" / "OPTIMIZER-OBLIGATIONS.md"
    )
    assert obligations.read_text(encoding="utf-8") == "obligations"
    intro = root / "optimizer-theory" / "docs" / "00 Introduzione e scopo.md"
    assert intro.read_text(encoding="utf-8") == "intro"


def test_stage_theory_is_idempotent_and_drift_free(tmp_path):
    src = _fake_theory(tmp_path / "canonical")
    dest = tmp_path / "agent-fs"

    backend.stage_theory(_PRELOAD, source=src, dest=dest)

    # Simulate drift: hand-edit a staged file and drop a stray one.
    staged_docs = dest / "optimizer-theory" / "docs"
    (staged_docs / "00 Introduzione e scopo.md").write_text("HACKED", encoding="utf-8")
    (staged_docs / "stray.md").write_text("junk", encoding="utf-8")

    backend.stage_theory(_PRELOAD, source=src, dest=dest)

    # Regenerated from canonical: edit reverted, stray gone.
    assert (staged_docs / "00 Introduzione e scopo.md").read_text(
        encoding="utf-8"
    ) == "intro"
    assert not (staged_docs / "stray.md").exists()


def test_stage_theory_omits_git_and_obsidian_cruft(tmp_path):
    src = _fake_theory(tmp_path / "canonical")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    (src / "docs" / ".obsidian").mkdir()
    (src / "docs" / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "agent-fs"

    root = backend.stage_theory(_PRELOAD, source=src, dest=dest)

    staged = root / "optimizer-theory"
    assert not (staged / ".git").exists()
    assert not (staged / "docs" / ".obsidian").exists()
    assert (staged / "docs" / "00 Introduzione e scopo.md").exists()


def test_stage_theory_preload_gate_creates_empty_workdir(tmp_path):
    src = _fake_theory(tmp_path / "canonical")
    dest = tmp_path / "agent-fs"

    root = backend.stage_theory(_NO_PRELOAD, source=src, dest=dest)

    assert root == dest
    assert dest.is_dir()
    # Gated off: canonical tree is not copied even though the source exists.
    assert not (dest / "optimizer-theory").exists()


def test_stage_theory_missing_source_raises_when_preloading(tmp_path):
    missing = tmp_path / "does-not-exist"
    dest = tmp_path / "agent-fs"

    with pytest.raises(FileNotFoundError, match="optimizer-theory tree not found"):
        backend.stage_theory(_PRELOAD, source=missing, dest=dest)


def test_stage_theory_missing_skills_source_raises_when_preloading(tmp_path):
    theory = _fake_theory(tmp_path / "canonical")
    dest = tmp_path / "agent-fs"

    with pytest.raises(FileNotFoundError, match="skills tree not found"):
        backend.stage_theory(
            _PRELOAD,
            source=theory,
            dest=dest,
            skills_source=tmp_path / "no-skills-here",
        )


def test_build_backend_roots_virtual_filesystembackend_at_staged_workdir(
    tmp_path, monkeypatch
):
    src = _fake_theory(tmp_path / "canonical")
    dest = tmp_path / "agent-fs"
    monkeypatch.setattr(backend, "_THEORY_SRC", src)
    monkeypatch.setattr(backend, "_AGENT_FS_ROOT", dest)

    fs = backend.build_backend(_PRELOAD)

    assert type(fs).__name__ == "FilesystemBackend"
    assert Path(fs.cwd).resolve() == dest.resolve()
    assert fs.virtual_mode is True
    obligations = (
        dest / "optimizer-theory" / "docs" / "architecture" / "OPTIMIZER-OBLIGATIONS.md"
    )
    assert obligations.exists()


def test_build_backend_threads_virtual_mode_from_config(tmp_path, monkeypatch):
    src = _fake_theory(tmp_path / "canonical")
    dest = tmp_path / "agent-fs"
    monkeypatch.setattr(backend, "_THEORY_SRC", src)
    monkeypatch.setattr(backend, "_AGENT_FS_ROOT", dest)

    cfg = FundConfig(preload_theory=True, agent_virtual_mode=False)
    fs = backend.build_backend(cfg)

    assert fs.virtual_mode is False


# --- Risk R1: per-role skills staged under the backend root --------------------
# deepagents loads skills THROUGH the backend, so per-role skills must live under
# the staged root as ``skills/<role>/<skill-name>/SKILL.md`` and be reachable by a
# root-relative source (``skills/<role>``) under ``virtual_mode``.


def test_stage_theory_stages_per_role_skills(tmp_path):
    from fund.agents.skills import SKILLS_BY_AGENT, STAGED_SKILLS_ROOT

    root, _ = _stage(_PRELOAD, tmp_path)

    for role, names in SKILLS_BY_AGENT.items():
        for name in names:
            skill_md = root / STAGED_SKILLS_ROOT / role / name / "SKILL.md"
            assert skill_md.is_file(), f"{role}/{name} not staged"
            # The immediate parent of SKILL.md is the skill name (deepagents
            # validates the frontmatter name == that dir name).
            assert skill_md.parent.name == name


def test_staged_skills_resolve_through_virtual_backend(tmp_path):
    """The R1 check: each role's root-relative source resolves under the staged,
    virtual-mode backend (an absolute path outside the root would be blocked)."""
    from deepagents.backends import FilesystemBackend

    from fund.agents.skills import SKILLS_BY_AGENT, skill_sources

    root, _ = _stage(_PRELOAD, tmp_path)
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    for role, names in SKILLS_BY_AGENT.items():
        (source,) = skill_sources(role)
        listing = fs.ls(source)
        assert listing.error is None, f"{role}: {listing.error}"
        found = {
            entry["path"].rstrip("/").rsplit("/", 1)[-1]
            for entry in listing.entries
            if entry["is_dir"]
        }
        assert found == set(names), f"{role}: {found} != {set(names)}"
        # Every listed skill's SKILL.md downloads through the backend.
        responses = fs.download_files([f"{source}/{name}/SKILL.md" for name in names])
        for resp in responses:
            assert resp.error is None


def test_stage_skills_is_idempotent_and_drift_free(tmp_path):
    from fund.agents.skills import STAGED_SKILLS_ROOT

    root, _ = _stage(_PRELOAD, tmp_path)
    staged_skills = root / STAGED_SKILLS_ROOT

    # Simulate drift: hand-edit a staged SKILL.md and drop a stray skill dir.
    victim = next(staged_skills.rglob("SKILL.md"))
    victim.write_text("HACKED", encoding="utf-8")
    (staged_skills / "stray").mkdir()

    _stage(_PRELOAD, tmp_path)

    assert victim.read_text(encoding="utf-8") != "HACKED"
    assert not (staged_skills / "stray").exists()


def test_stage_theory_preload_gate_skips_skills(tmp_path):
    from fund.agents.skills import STAGED_SKILLS_ROOT

    root, _ = _stage(_NO_PRELOAD, tmp_path)

    assert root.is_dir()
    # Gated off: neither theory docs nor skills are staged.
    assert not (root / "optimizer-theory").exists()
    assert not (root / STAGED_SKILLS_ROOT).exists()
