"""T6 — profiler system prompt + ``mifid-profiling`` skill (frontmatter + smoke).

The LLM half of Fase 5 needs an English, ESMA-2022-derived system prompt and a
routing skill. This phase creates them; wiring the agent is Task 7. These tests
assert:

* ``PROFILER_SYSTEM_PROMPT`` imports (no heavy deps), is English, walks the four
  ESMA pillars, and states the load-bearing boundary — the LLM interprets answers
  into typed inputs and **never** invents optimizer knobs.
* ``skills/mifid-profiling/SKILL.md`` has valid YAML frontmatter (``name`` +
  a specific ``description``) and a body that routes to the deterministic mapping.
* ``skills/mifid-profiling/reference.md`` carries the pillar→knob table + citations.
* No ``deepagents`` / ``langchain`` hard-import leaks into ``fund/schemas/`` (the
  schemas stay a pure, importable data layer — SPEC §6 "Never").
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from fund.agents.prompts import PROFILER_SYSTEM_PROMPT

_FUND_ROOT = Path(__file__).resolve().parents[2]
_SKILL_DIR = _FUND_ROOT / "src" / "fund" / "skills" / "mifid-profiling"
_SCHEMAS_DIR = _FUND_ROOT / "src" / "fund" / "schemas"


def _frontmatter(text: str) -> dict[str, object]:
    """Parse the leading ``---`` YAML frontmatter block into a dict."""
    assert text.startswith("---\n"), "SKILL.md must open with a --- frontmatter block"
    _, fm, _body = text.split("---\n", 2)
    parsed = yaml.safe_load(fm)
    assert isinstance(parsed, dict)
    return parsed


# --- system prompt ----------------------------------------------------------


def test_profiler_prompt_is_english_and_substantial():
    assert isinstance(PROFILER_SYSTEM_PROMPT, str)
    assert len(PROFILER_SYSTEM_PROMPT) > 200
    lower = PROFILER_SYSTEM_PROMPT.lower()
    assert "mifid" in lower
    assert "esma" in lower


def test_profiler_prompt_walks_the_four_esma_pillars():
    lower = PROFILER_SYSTEM_PROMPT.lower()
    # The four ESMA suitability pillars.
    assert "knowledge" in lower  # + experience
    assert "capacity" in lower  # financial situation / loss capacity
    assert "objectives" in lower  # incl. horizon + risk tolerance
    assert "esg" in lower


def test_profiler_prompt_states_the_deterministic_mapping_boundary():
    lower = PROFILER_SYSTEM_PROMPT.lower()
    # Load-bearing rule: the LLM emits typed inputs, never a knob.
    assert "knob" in lower
    assert "never" in lower
    assert "typed" in lower


def test_profiler_prompt_flags_anti_overconfidence():
    assert "overconfiden" in PROFILER_SYSTEM_PROMPT.lower()


# --- skill: frontmatter + body ----------------------------------------------


def test_skill_md_has_valid_frontmatter():
    text = (_SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    fm = _frontmatter(text)

    assert fm.get("name") == "mifid-profiling"
    description = fm.get("description")
    assert isinstance(description, str)
    assert len(description) > 40  # specific, not a one-word stub
    assert "mifid" in description.lower()


def test_skill_md_body_routes_to_the_deterministic_mapping():
    text = (_SKILL_DIR / "SKILL.md").read_text(encoding="utf-8").lower()
    assert "build_constraint_set" in text
    assert "pillar" in text
    # Points to the theory, never copies it (SPEC §3).
    assert "deep_agent" in text


def test_reference_md_carries_the_pillar_knob_table():
    text = (_SKILL_DIR / "reference.md").read_text(encoding="utf-8")
    lower = text.lower()
    assert "a_gamma" in lower
    assert "min(tolerance, capacity)" in lower
    assert "risk_measure" in lower
    assert "universe_filters" in lower
    assert "esg" in lower
    # Theory citations preserved (deep_agent.md line refs).
    assert "01:142" in text or "deep_agent" in lower


# --- hygiene: schemas stay a pure data layer --------------------------------


def _imported_roots(source: str) -> set[str]:
    """Top-level module roots imported by ``source`` (AST — ignores docstrings)."""
    roots: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_schemas_do_not_hard_import_deepagents_or_langchain():
    # "hard-import" — a real import statement, not a mention in a docstring; parse
    # the AST so prose like "wired at the call site (langchain_ollama)" is ignored.
    forbidden = {"deepagents", "langchain", "langchain_core", "langchain_ollama"}
    for py in sorted(_SCHEMAS_DIR.glob("*.py")):
        roots = _imported_roots(py.read_text(encoding="utf-8"))
        leaked = roots & forbidden
        assert not leaked, f"{py.name} hard-imports {leaked}"
