"""Task 4 — the five role system prompts for the Phase-7 deep agent.

The PM/orchestrator and its four subagents (economist → allocator → risk →
executor) each run under an English, plain-string system prompt living in
``fund/agents/prompts.py`` (no ``deepagents`` / ``langchain`` import — the prompts
stay a cheap, agent-stack-free text layer; the agent is assembled in
``agents/graph.py``, Task 5).

These tests assert the SPEC/plan contract for Task 4:

* every role prompt is a non-empty English string that states the **load-bearing
  rule** — the LLM chooses structured inputs, the optimizer computes the weights,
  and no agent ever emits a weight;
* each prompt names its own tools/skills;
* the **PM** names the fixed delegation order, the **blocking** risk gate, and the
  **round cap** (→ incomplete + HITL when hit);
* the **risk** officer reports pass/fail with explicit violations and hard-blocks a
  MiFID/ESG/validation breach;
* the **executor** treats ``place_orders`` as paper-only, next-close, idempotent,
  HITL-gated, and never fabricates weights;
* ``__all__`` exports all six prompts.
"""

from __future__ import annotations

from fund.agents import prompts
from fund.agents.prompts import (
    ALLOCATOR_SYSTEM_PROMPT,
    ECONOMIST_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    PROFILER_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
)

_ROLE_PROMPTS = {
    "PM_SYSTEM_PROMPT": PM_SYSTEM_PROMPT,
    "ECONOMIST_SYSTEM_PROMPT": ECONOMIST_SYSTEM_PROMPT,
    "ALLOCATOR_SYSTEM_PROMPT": ALLOCATOR_SYSTEM_PROMPT,
    "RISK_SYSTEM_PROMPT": RISK_SYSTEM_PROMPT,
    "EXECUTOR_SYSTEM_PROMPT": EXECUTOR_SYSTEM_PROMPT,
}


# --- shared shape: every role prompt is substantial English --------------------


def test_every_role_prompt_is_a_substantial_english_string():
    for name, prompt in _ROLE_PROMPTS.items():
        assert isinstance(prompt, str), name
        assert len(prompt) > 200, name
        assert "you" in prompt.lower(), name


def test_every_role_prompt_states_the_load_bearing_rule():
    # Load-bearing invariant (whole architecture): the LLM chooses structured
    # inputs; the optimizer computes weights; no agent ever emits a weight.
    for name, prompt in _ROLE_PROMPTS.items():
        lower = prompt.lower()
        assert "optimizer" in lower, name
        assert "weight" in lower, name
        assert "never" in lower, name


# --- PM: delegation order, blocking gate, round cap ----------------------------


def test_pm_prompt_names_the_fixed_delegation_order():
    lower = PM_SYSTEM_PROMPT.lower()
    assert "economist → allocator → risk → executor" in lower
    # Delegation is via the built-in ``task`` tool.
    assert "task" in lower
    # Routing skill.
    assert "fund-orchestration" in lower


def test_pm_prompt_marks_risk_as_a_blocking_gate():
    lower = PM_SYSTEM_PROMPT.lower()
    assert "blocking" in lower
    assert "gate" in lower


def test_pm_prompt_respects_the_round_cap():
    lower = PM_SYSTEM_PROMPT.lower()
    assert "round" in lower
    assert "incomplete" in lower


# --- economist: narrative regime read, its tools + skills ----------------------


def test_economist_prompt_names_its_tools_and_skills():
    lower = ECONOMIST_SYSTEM_PROMPT.lower()
    assert "get_macro_series" in lower
    assert "get_prices" in lower
    assert "macro-regime-read" in lower
    assert "views-construction" in lower


# --- allocator: universe → moments → optimizer, its tools + skills -------------


def test_allocator_prompt_names_its_tools_and_skills():
    lower = ALLOCATOR_SYSTEM_PROMPT.lower()
    assert "universe_filter" in lower
    assert "estimate_moments" in lower
    assert "optimize_portfolio" in lower
    assert "universe-preselection" in lower
    assert "optimization-objective-map" in lower


# --- risk: pass/fail with violations; breaches hard-block ----------------------


def test_risk_prompt_reports_pass_fail_with_explicit_violations():
    lower = RISK_SYSTEM_PROMPT.lower()
    assert "pass" in lower
    assert "fail" in lower
    assert "violation" in lower
    assert "risk_check" in lower
    assert "backtest" in lower
    assert "risk-limits-check" in lower


def test_risk_prompt_hard_blocks_mifid_esg_validation_breach():
    lower = RISK_SYSTEM_PROMPT.lower()
    assert "hard" in lower and "block" in lower
    assert "mifid" in lower
    assert "esg" in lower
    assert "validation" in lower


# --- executor: paper-only, next-close, idempotent, HITL, no fabricated weights -


def test_executor_prompt_describes_paper_orders_semantics():
    lower = EXECUTOR_SYSTEM_PROMPT.lower()
    assert "place_orders" in lower
    assert "paper" in lower
    assert "next" in lower and "close" in lower
    assert "idempotent" in lower
    assert "rebalancing-execution" in lower


def test_executor_prompt_forbids_fabricating_weights():
    lower = EXECUTOR_SYSTEM_PROMPT.lower()
    assert "fabricate" in lower
    assert "weight" in lower


# --- public surface ------------------------------------------------------------


def test_all_exports_the_six_prompts():
    expected = {
        "PROFILER_SYSTEM_PROMPT",
        "PM_SYSTEM_PROMPT",
        "ECONOMIST_SYSTEM_PROMPT",
        "ALLOCATOR_SYSTEM_PROMPT",
        "RISK_SYSTEM_PROMPT",
        "EXECUTOR_SYSTEM_PROMPT",
    }
    assert expected <= set(prompts.__all__)
    # The pre-existing profiler prompt is untouched and still exported.
    assert isinstance(PROFILER_SYSTEM_PROMPT, str)
