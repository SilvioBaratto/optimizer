"""Static validation of every deepagents skill.

For each ``skills/<dir>/`` it asserts the ``SKILL.md`` frontmatter shape, the
load-bearing rule, a non-empty ``reference.md``, that theory citations resolve to
real ``optimizer-theory/docs`` chapters, and that the skill routes to its expected
real ``@tool``s. No model, no DB, no optimizer import (``__all__`` is read via AST).
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
_THEORY_DOCS = _FUND_ROOT.parent / "optimizerwiki" / "docs"

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

# The 11 real @tools; kept in sync with fund.tools.__all__ by a test below.
# ``get_recent_events`` / ``get_portfolio_history`` are the Phase-4 history tools
# (economist / orchestrator) — available but not mandated by any skill prompt, so
# they are absent from ``_EXPECTED_TOOLS``.
_REAL_TOOLS = frozenset(
    {
        "get_prices",
        "get_macro_series",
        "get_recent_events",
        "get_regime_stats",
        "estimate_moments",
        "optimize_portfolio",
        "universe_filter",
        "risk_check",
        "backtest",
        "place_orders",
        "get_portfolio_history",
    }
)

# Tools each skill MUST reference (positive routing check). Empty = no mandatory
# tool (the skill emits a schema / calls a deterministic mapping instead).
_EXPECTED_TOOLS: dict[str, frozenset[str]] = {
    "fund-orchestration": frozenset({"task"}),
    "mifid-profiling": frozenset(),
    "macro-regime-read": frozenset({"get_macro_series", "get_regime_stats"}),
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
        # The canonical docs are underscore-named (``00_introduction_and_scope.md``);
        # tolerate a legacy space-named tree so the check tracks the real filenames.
        assert list(_THEORY_DOCS.glob(f"{chapter}_*.md")) or list(
            _THEORY_DOCS.glob(f"{chapter} *.md")
        ), f"{name}: citation `{cite}` → no optimizer-theory/docs/{chapter}_*.md"


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_skill_references_expected_tools(name: str):
    body = _read(name, "SKILL.md")
    for tool in _EXPECTED_TOOLS[name]:
        assert tool in body, f"{name}: SKILL.md must reference tool `{tool}`"


# No tool builds a Black-Litterman prior from the economist's ViewSet (the views are
# recorded in the allocator's decision for the audit trail), so no skill the agents
# load may promise that the optimizer prices them.
_PHANTOM_VIEW_PRICING = (
    r"optimizer (folds|runs black-litterman|computes the posterior)",
    r"price[sd]? (the )?(views? )?into the (black-litterman )?(prior|posterior)",
    r"views? (as|into) the (prior|posterior)",
)


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_skills_do_not_claim_the_optimizer_prices_views(name: str):
    text = _read(name, "SKILL.md") + " " + _read(name, "reference.md")
    flat = " ".join(text.lower().split())
    for pattern in _PHANTOM_VIEW_PRICING:
        assert not re.search(pattern, flat), f"{name}: stale claim /{pattern}/"


def test_optimization_objective_map_has_factories():
    factories = _SKILLS_DIR / "optimization-objective-map" / "factories.py"
    assert factories.read_text(encoding="utf-8").strip(), "empty factories.py"


# --- fund-orchestration: only promise what the PM can actually see -----------


def test_orchestration_reads_mandate_and_limits_from_the_run_instruction():
    # The ConstraintSet lives in the langgraph Store and no PM tool reads it, so a
    # skill that tells the PM to "read the ConstraintSet" invites invented limits.
    # The only legitimate source is the digest the run instruction carries.
    body = _read("fund-orchestration", "SKILL.md")

    assert "run instruction" in body
    assert "verbatim" in body
    assert "Confirm an active `ConstraintSet` exists" not in body


def _orchestration_round_cap_texts() -> dict[str, str]:
    skill = _read("fund-orchestration", "SKILL.md")
    reference = _read("fund-orchestration", "reference.md")
    step = re.search(r"3\. [^\n]*round cap.*?(?=\n4\.)", skill, re.DOTALL)
    guard = re.search(r"- Round cap[^\n]*(?:\n {2}[^\n]*)*", reference)
    assert step and guard
    return {"SKILL.md step 3": step.group(0), "reference.md guardrail": guard.group(0)}


def test_orchestration_round_cap_promises_no_hitl_checkpoint():
    # place_orders is the PM's only gated tool, so "raise a HITL checkpoint" at the
    # cap is only satisfiable by committing the latest (possibly risk-blocked)
    # weights; the harness finalises the run INCOMPLETE on its own.
    for where, text in _orchestration_round_cap_texts().items():
        assert "hitl" not in text.lower(), where
        assert "INCOMPLETE" in text, where


def test_orchestration_forbids_place_orders_before_risk_passes():
    body = _read("fund-orchestration", "SKILL.md")

    assert "Never call `place_orders` unless the risk gate passed" in body


# --- risk gate: skills must match the tools' real (weight-free) signatures ----


def test_orchestration_requires_a_fresh_risk_pass_after_each_allocator_run():
    # A risk pass is bound to one weight vector; re-running the allocator replaces
    # the weights place_orders would commit, so the pass must be earned again.
    body = _read("fund-orchestration", "SKILL.md")

    assert "delegate to risk again" in body
    assert "refuses weights whose newest recorded risk verdict" in body


_WEIGHT_FREE_TOOL_CALL_RE = re.compile(r"`(risk_check|backtest)\(([^)]*)\)`")


@pytest.mark.parametrize("name", _SKILL_DIRS)
def test_skills_call_risk_tools_without_a_weights_argument(name: str):
    # ``risk_check`` / ``backtest`` judge the allocator's audited weights and accept
    # no arguments; a skill that tells the risk officer to pass ``weights`` makes it
    # relay (and round) a vector the gate must never trust.
    text = _read(name, "SKILL.md") + "\n" + _read(name, "reference.md")

    for tool, args in _WEIGHT_FREE_TOOL_CALL_RE.findall(text):
        assert args.strip() == "", f"{name}: `{tool}({args})` takes no arguments"


def test_risk_skill_records_a_verdict_from_the_audited_weights():
    body = _read("risk-limits-check", "SKILL.md")

    assert "`risk_check()`" in body
    assert "audited" in body


# --- executor inputs: one name for the band, and a source for current holdings ---


def test_skills_name_the_band_the_way_the_run_instruction_does():
    # The seed calls the mandate threshold "drift band (L1)". A skill that sends the
    # executor to "the profile's no-trade band" names a limit it is never shown.
    rebalancing = _read("rebalancing-execution", "SKILL.md")
    objective_map = _read("optimization-objective-map", "SKILL.md")

    assert "drift band (L1)" in rebalancing
    assert "drift band" in objective_map
    assert "no-trade band" not in rebalancing
    assert "no-trade band" not in objective_map


def test_executor_is_told_where_current_holdings_come_from():
    # The executor binds no tools, so the drift test needs the holdings relayed: the
    # PM reads them (get_portfolio_history) and both skills must say so.
    orchestration = _read("fund-orchestration", "SKILL.md")
    rebalancing = _read("rebalancing-execution", "SKILL.md")

    for body in (orchestration, rebalancing):
        assert "current_holdings" in body
        assert "get_portfolio_history" in body
