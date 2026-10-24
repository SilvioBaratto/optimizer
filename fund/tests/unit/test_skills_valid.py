"""Static validation of every deepagents skill.

For each ``skills/<dir>/`` it asserts the ``SKILL.md`` frontmatter shape, the
load-bearing rule, a non-empty ``reference.md``, that theory citations resolve to
real ``optimizer-theory/docs`` chapters, and that the skill routes to its expected
real ``@tool``s. No model, no DB, and no optimizer import at module scope
(``__all__`` is read via AST); the few tests that check a skill against the real
bound tool or optimizer API import them inside the test body.
"""

from __future__ import annotations

import ast
import datetime as dt
import re
import uuid
from pathlib import Path

import pytest
import yaml

_FUND_ROOT = Path(__file__).resolve().parents[2]  # …/fund
_FUND_SRC = _FUND_ROOT / "src" / "fund"
_SKILLS_DIR = _FUND_SRC / "skills"
_THEORY_DOCS = _FUND_ROOT.parent / "optimizerwiki" / "docs"
_TOOLSETS_SRC = _FUND_SRC / "agents" / "toolsets.py"

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
        "list_universe",
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
    "universe-preselection": frozenset(
        {"list_universe", "universe_filter", "estimate_moments"}
    ),
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


def _bound_tool_params(tool: str) -> list[str]:
    """Parameter names of the ``@tool`` that ``toolsets._bind_<tool>`` exposes.

    Read via AST so the check needs no optimizer import and tracks the schema the
    LLM actually sees (a skill documenting any other argument misleads it).
    """
    tree = ast.parse(_TOOLSETS_SRC.read_text(encoding="utf-8"))
    for binder in ast.walk(tree):
        if isinstance(binder, ast.FunctionDef) and binder.name == f"_bind_{tool}":
            for inner in ast.walk(binder):
                if isinstance(inner, ast.FunctionDef) and inner.name == tool:
                    return [a.arg for a in inner.args.args]
    raise AssertionError(f"toolsets.py binds no `{tool}` tool")


def _documented_call_args(text: str, tool: str) -> list[list[str]]:
    """Argument names of every inline ``tool(a, b)`` call form in ``text``."""
    calls = re.findall(rf"`{tool}\(([^`)]*)\)`", text)
    return [[a.strip() for a in call.split(",") if a.strip()] for call in calls]


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


# --- optimization-objective-map vs the code it documents ---------------------

_OBJECTIVE_MAP_SKILL = "optimization-objective-map"


def _section(text: str, heading: str) -> str:
    """Body of the ``## <heading>`` section, up to the next ``## `` heading."""
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.M
    )
    assert match, f"no `## {heading}` section"
    return match.group(1)


# --- universe-preselection documents the surface the allocator really has ----

# Selection knobs no bound tool or ``PreSelectionConfig`` field implements: the filter
# has no ADV/price, factor or sector-cap input, so a skill presenting them as levers
# makes the LLM believe a screen ran. (The ESG and K&E instrument screens are real: the
# bound tool applies them from the ConstraintSet.)
_INERT_KNOB_RE = re.compile(
    r"liquidity|\bADV\b|factor[- ](?:score|tilt)|sector[- ]cap", re.IGNORECASE
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|cannot|unavailable|absent|ignored)\b", re.IGNORECASE
)


@pytest.mark.parametrize("tool", ["universe_filter", "estimate_moments"])
def test_universe_preselection_calls_match_the_bound_tool_signature(tool: str):
    skill = _read("universe-preselection", "SKILL.md")
    reference = _read("universe-preselection", "reference.md")
    real = set(_bound_tool_params(tool))
    skill_calls = _documented_call_args(skill, tool)
    assert skill_calls, f"SKILL.md must show the `{tool}(...)` call form"
    for args in skill_calls + _documented_call_args(reference, tool):
        assert set(args) <= real, f"`{tool}({', '.join(args)})` vs real {sorted(real)}"


def test_universe_preselection_does_not_advertise_inert_selection_knobs():
    text = _read("universe-preselection", "SKILL.md") + "\n"
    text += _read("universe-preselection", "reference.md")
    for line in text.splitlines():
        if _INERT_KNOB_RE.search(line):
            assert _NEGATION_RE.search(line), f"presented as a live knob: {line!r}"


def _module_str_constant(path: Path, name: str) -> str:
    """Value of a module-level string constant, read via AST (no agent-stack import)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} defines no `{name}` string constant")


def _sentences(text: str) -> list[str]:
    """Soft-wrap-insensitive sentences, so a claim split over two lines stays whole."""
    return re.split(r"(?<=[.;])\s+", " ".join(text.split()))


def _bound_optimize_tool_args() -> set[str]:
    """Model-facing arg names of the allocator's real bound ``optimize_portfolio``."""
    from fund.agents.toolsets import RunContext, bind_toolset
    from fund.config import settings

    ctx = RunContext(
        session=None,  # type: ignore[arg-type]  # closure only inspected, never run
        asof=dt.date(2024, 1, 2),
        store=None,
        config=settings,
        run_id=uuid.uuid4(),
        portfolio_id=uuid.uuid4(),
    )
    tools = {t.name: t for t in bind_toolset("allocator", ctx)}
    return set(tools["optimize_portfolio"].args)


def test_objective_map_skill_documents_the_real_optimize_call():
    body = _read(_OBJECTIVE_MAP_SKILL, "SKILL.md")
    real_args = _bound_optimize_tool_args()
    calls = re.findall(r"optimize_portfolio\(([^)]*)\)", body)
    assert calls, "SKILL.md must show the optimize_portfolio call"
    for raw in calls:
        documented = {a.split("=")[0].strip() for a in raw.split(",") if a.strip()}
        assert documented == real_args, (
            f"SKILL.md calls optimize_portfolio({raw}) but the bound tool takes "
            f"{sorted(real_args)}"
        )


def test_objective_map_skill_states_what_the_tool_actually_applies():
    from fund.tools.optimize import _CONSTRAINT_FIELDS, _resolve_config

    body = _read(_OBJECTIVE_MAP_SKILL, "SKILL.md")
    applied = _resolve_config(None)
    assert applied.objective.name in body
    assert applied.risk_measure.name in body
    for field in _CONSTRAINT_FIELDS:
        assert field in body, f"SKILL.md must name the honoured bound `{field}`"


def test_objective_map_procedure_does_not_instruct_knobs_the_tool_ignores():
    procedure = _section(_read(_OBJECTIVE_MAP_SKILL, "SKILL.md"), "Procedure").lower()
    for knob in ("robust", "turnover"):
        assert knob not in procedure, (
            f"Procedure tells the allocator to apply `{knob}`, which "
            "optimize_portfolio never receives"
        )


def _objective_table_rows() -> list[tuple[set[str], str]]:
    """Rows of reference.md's objective table as ``(first-cell words, row text)``."""
    lines = _read(_OBJECTIVE_MAP_SKILL, "reference.md").splitlines()
    header = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("|") and "ConstraintSet.objective" in line
        ),
        None,
    )
    assert header is not None, "reference.md has no `ConstraintSet.objective` table"
    rows: list[tuple[set[str], str]] = []
    for line in lines[header + 2 :]:  # skip the header and its divider row
        if not line.startswith("|"):
            break
        first_cell = line.split("|")[1]
        rows.append((set(re.findall(r"[a-z_]+", first_cell)), line))
    return rows


def test_objective_map_reference_table_is_keyed_on_the_profile_objectives():
    from fund.schemas.enums import ObjectiveChoice

    keyed = set().union(*(words for words, _ in _objective_table_rows()))
    assert keyed == {o.value for o in ObjectiveChoice}


def test_objective_map_reference_table_matches_the_constraint_set_mapping():
    from fund.schemas.constraint_set import _OBJECTIVE_MAP

    rows = _objective_table_rows()
    for objective, target in _OBJECTIVE_MAP.items():
        matching = [text for words, text in rows if objective.value in words]
        assert len(matching) == 1, f"`{objective.value}` needs exactly one table row"
        assert target.upper() in matching[0], (
            f"`{objective.value}` row lacks {target.upper()}"
        )


def test_objective_map_skill_description_uses_the_profile_objective_names():
    from fund.schemas.enums import ObjectiveChoice

    description = str(
        _frontmatter(_read(_OBJECTIVE_MAP_SKILL, "SKILL.md"))["description"]
    )
    words = set(re.findall(r"[a-z_]+", description))
    for objective in ObjectiveChoice:
        assert objective.value in words
    for stale in ("min-risk", "max-utility", "max-ratio", "risk-budget"):
        assert stale not in description


_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _skeleton_resource() -> Path:
    """The factory-skeleton file that SKILL.md tells the allocator to open."""
    body = _read(_OBJECTIVE_MAP_SKILL, "SKILL.md")
    names = [
        span
        for span in _inline_code_spans(body)
        if re.fullmatch(r"factories\.\w+", span)
    ]
    assert len(names) == 1, f"SKILL.md should name exactly one skeleton file: {names}"
    return _SKILLS_DIR / _OBJECTIVE_MAP_SKILL / names[0]


def _skeleton_sections() -> list[tuple[str, list[str]]]:
    """``(heading, python blocks)`` for each ``## `` section holding code."""
    text = _skeleton_resource().read_text(encoding="utf-8")
    sections = re.findall(r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.M)
    return [
        (heading, _PYTHON_BLOCK_RE.findall(body))
        for heading, body in sections
        if _PYTHON_BLOCK_RE.search(body)
    ]


def test_objective_map_factory_skeletons_equal_the_constraint_set_mapping():
    from fund.schemas.constraint_set import _OBJECTIVE_MAP, Bounds, ConstraintSet
    from fund.schemas.enums import Horizon, RiskMeasureChoice

    sections = [
        (heading, blocks) for heading, blocks in _skeleton_sections() if "→" in heading
    ]
    for objective, target in _OBJECTIVE_MAP.items():
        owners = [
            (heading, blocks)
            for heading, blocks in sections
            if objective.value in re.findall(r"[a-z_]+", heading.split("→")[0])
        ]
        assert len(owners) == 1, f"`{objective.value}` needs exactly one skeleton"
        heading, blocks = owners[0]
        assert target.upper() in heading
        assert len(blocks) == 1
        namespace: dict[str, object] = {
            "a_gamma": 3.0,
            "beta": 0.9,  # distinct from MeanRiskConfig's 0.95 default
            "cap": 0.25,
            "cardinality": None,
        }
        exec(compile(blocks[0], heading, "exec"), namespace)  # noqa: S102
        config = namespace["config"]
        expected = ConstraintSet(
            portfolio_id="p",
            base_currency="EUR",
            a_gamma=3.0,
            objective=objective,
            risk_measure=RiskMeasureChoice(config.risk_measure.value),
            beta=0.9,
            nu1=0.1,
            nu2=0.2,
            nu3=0.3,
            horizon=Horizon.MEDIUM,
            bounds=Bounds(max_weights=0.25),
            cardinality=None,
        ).to_mean_risk_config()
        assert config == expected, f"`{heading}` drifted from to_mean_risk_config()"


def test_objective_map_factory_skeletons_run_against_the_real_optimizer():
    import numpy as np

    skeletons = _skeleton_resource()
    blocks = _PYTHON_BLOCK_RE.findall(skeletons.read_text(encoding="utf-8"))
    assert len(blocks) >= 4, f"{skeletons.name} holds no runnable skeletons"
    for block in blocks:
        namespace: dict[str, object] = {
            "a_gamma": 3.0,
            "beta": 0.95,
            "cap": 0.25,
            # A cardinality cap needs a mixed-integer solver, which the default
            # config does not carry; the profiler leaves the cap unset too.
            "cardinality": None,
            "previous_weights": np.full(3, 1 / 3),
        }
        exec(compile(block, skeletons.name, "exec"), namespace)  # noqa: S102


def test_skill_local_file_references_resolve():
    for name in _SKILL_DIRS:
        text = _read(name, "SKILL.md")
        for span in _inline_code_spans(text):
            if re.fullmatch(r"[\w-]+\.(?:md|py)", span):
                assert (_SKILLS_DIR / name / span).is_file(), (
                    f"{name}: SKILL.md points at `{span}` which does not exist"
                )


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


# The allocator's system prompt is read on every run, the skill only on demand, so
# the prompt may not claim a screen the skill says is absent.
def test_allocator_prompt_does_not_claim_screens_the_filter_lacks():
    from fund.agents.prompts import ALLOCATOR_SYSTEM_PROMPT

    for sentence in _sentences(ALLOCATOR_SYSTEM_PROMPT):
        if _INERT_KNOB_RE.search(sentence):
            assert _NEGATION_RE.search(sentence), f"presented as applied: {sentence!r}"


def test_allocator_is_described_as_pre_selecting_handed_over_tickers():
    from fund.agents.prompts import ALLOCATOR_SYSTEM_PROMPT, PM_SYSTEM_PROMPT

    pm_bullet = re.search(r"\*\*allocator\*\*(.*?)\n3\.", PM_SYSTEM_PROMPT, re.S)
    assert pm_bullet, "PM prompt lost its allocator bullet"
    always_loaded = {
        "allocator prompt": ALLOCATOR_SYSTEM_PROMPT,
        "PM prompt bullet": pm_bullet.group(1),
        "graph description": _module_str_constant(
            _FUND_SRC / "agents" / "graph.py", "_ALLOCATOR_DESCRIPTION"
        ),
        "orchestration reference": next(
            row
            for row in _read("fund-orchestration", "reference.md").splitlines()
            if "| allocator |" in row
        ),
    }
    for label, text in always_loaded.items():
        assert "pre-select" in text.lower(), f"{label} still says the allocator filters"


def test_universe_preselection_missing_has_both_causes_and_is_not_dropped_by_hand():
    skill = _read("universe-preselection", "SKILL.md")
    # The load-bearing rule bars the LLM from dropping a ticker, so no procedure
    # step may tell it to; the tools already skip what they flag as ``missing``.
    assert re.search(r"never adds, drops or invents", skill)
    steps = re.split(r"^(?=\d+\. )", _section(skill, "Procedure"), flags=re.M)
    step = " ".join(next(s for s in steps if "`missing`" in s).split())
    assert "no instrument" in step and "no priced" in step, step
    assert "drop" not in step.lower(), step


def test_fund_orchestration_skill_documents_the_adviser_directive():
    body = " ".join(
        _read("fund-orchestration", "SKILL.md").lower().replace("`", "").split()
    )
    assert "adviser directive" in body
    assert "not executed" in body
    assert "place_orders again" in body


# --- rebalancing-execution: the doc must match what the code does ------------


def test_rebalancing_execution_reference_documents_the_no_fill_bar_outcome():
    # The next-close fill needs a priced bar strictly after ``asof``; without one
    # ``place_orders`` writes no ticket and the approved run ends ``incomplete``.
    # The executor's reference must say so instead of implying the ticket always lands.
    text = _read("rebalancing-execution", "reference.md").lower()
    assert "no fill bar" in text
    assert "incomplete" in text
    assert "holdings" in text
