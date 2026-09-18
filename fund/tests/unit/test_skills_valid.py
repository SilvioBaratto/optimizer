"""Phase 6 D1 — static validation of every deepagents skill.

The machine-checked half of SPEC §5. For each ``skills/<dir>/`` it asserts the
``SKILL.md`` frontmatter shape, the load-bearing rule, a non-empty ``reference.md``,
that theory citations resolve to real ``optimizer-theory/docs`` chapters, and that
the skill routes to its expected real ``@tool``s. No model, no DB, no optimizer
import (``__all__`` is read via AST).

TDD note: until the 7 stub skills are authored, their parametrised cases are RED;
``mifid-profiling`` (Phase 5) is GREEN and proves the rules are mifid-compatible.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

_FUND_ROOT = Path(__file__).resolve().parents[2]  # …/fund
_FUND_SRC = _FUND_ROOT / "src" / "fund"
_SKILLS_DIR = _FUND_SRC / "skills"
_THEORY_DOCS = _FUND_ROOT.parent / "optimizer-theory" / "docs"

# deepagents built-in tools a skill body may legitimately name.
_BUILTINS = frozenset(
    {
        "task",
        "read_file",
        "write_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
        "write_todos",
    }
)

# The 8 real @tools (SPEC §1); kept in sync with fund.tools.__all__ by a test below.
_REAL_TOOLS = frozenset(
    {
        "get_prices",
        "get_macro_series",
        "estimate_moments",
        "optimize_portfolio",
        "universe_filter",
        "risk_check",
        "backtest",
        "place_orders",
    }
)

# Tools each skill MUST reference (positive routing check). Empty = no mandatory
# tool (the skill emits a schema / calls a deterministic mapping instead).
_EXPECTED_TOOLS: dict[str, frozenset[str]] = {
    "fund-orchestration": frozenset({"task"}),
    "mifid-profiling": frozenset(),
    "macro-regime-read": frozenset({"get_macro_series"}),
    "views-construction": frozenset(),
    "universe-preselection": frozenset({"universe_filter", "estimate_moments"}),
    "optimization-objective-map": frozenset({"optimize_portfolio"}),
    "risk-limits-check": frozenset({"risk_check", "backtest"}),
    "rebalancing-execution": frozenset({"place_orders"}),
}

_SKILL_DIRS = sorted(d.name for d in _SKILLS_DIR.iterdir() if d.is_dir())
_CITATION_RE = re.compile(r"^\d{2}:\d+(?:-\d+)?$")


def _frontmatter(text: str) -> dict[str, object]:
    """Parse the leading ``---`` YAML frontmatter block into a dict."""
    assert text.startswith("---\n"), "SKILL.md must open with a --- frontmatter block"
    _, fm, _body = text.split("---\n", 2)
    parsed = yaml.safe_load(fm)
    assert isinstance(parsed, dict)
    return parsed


def _read(name: str, fname: str) -> str:
    return (_SKILLS_DIR / name / fname).read_text(encoding="utf-8")


def _inline_code_spans(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def _tools_all_from_source() -> set[str]:
    """Read ``fund.tools.__all__`` via AST (no import → no optimizer)."""
    src = (_FUND_SRC / "tools" / "__init__.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("fund/tools/__init__.py defines no __all__")


# --- sanity: the test's own tables agree with the codebase --------------------


def test_expected_tools_table_is_a_subset_of_real_tools():
    for name, tools in _EXPECTED_TOOLS.items():
        assert tools <= (_REAL_TOOLS | _BUILTINS), f"{name}: unknown expected tool"


def test_real_tools_match_fund_tools_all():
    assert _tools_all_from_source() == _REAL_TOOLS


def test_every_skill_dir_is_in_the_expected_table():
    # Guards against an unregistered skill dir slipping past validation.
    assert set(_SKILL_DIRS) == set(_EXPECTED_TOOLS)


# --- per-skill validation -----------------------------------------------------


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_skill_md_frontmatter(name: str):
    fm = _frontmatter(_read(name, "SKILL.md"))
    assert fm.get("name") == name, "frontmatter name must equal the directory name"
    description = fm.get("description")
    assert isinstance(description, str) and len(description) > 40
    assert "when" in description.lower(), "description needs a routing cue"


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_skill_md_states_load_bearing_rule(name: str):
    lower = _read(name, "SKILL.md").lower()
    assert "never" in lower and "weight" in lower, "missing load-bearing rule"


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_reference_md_present_and_nonempty(name: str):
    assert _read(name, "reference.md").strip(), "reference.md must be non-empty"


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_theory_citations_resolve(name: str):
    text = _read(name, "SKILL.md") + "\n" + _read(name, "reference.md")
    citations = [t for t in _inline_code_spans(text) if _CITATION_RE.match(t)]
    for cite in citations:
        chapter = cite.split(":")[0]
        assert list(_THEORY_DOCS.glob(f"{chapter} *.md")), (
            f"{name}: citation `{cite}` → no optimizer-theory/docs/{chapter} *.md"
        )


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_skill_references_expected_tools(name: str):
    body = _read(name, "SKILL.md")
    for tool in _EXPECTED_TOOLS[name]:
        assert tool in body, f"{name}: SKILL.md must reference tool `{tool}`"


def test_optimization_objective_map_has_factories():
    factories = _SKILLS_DIR / "optimization-objective-map" / "factories.py"
    assert factories.read_text(encoding="utf-8").strip(), "empty factories.py"
