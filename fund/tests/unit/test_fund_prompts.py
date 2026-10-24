"""System prompts for the five fund-agent roles (PM + four subagents).

The PM/orchestrator and its four subagents (economist → allocator → risk →
executor) each run under an English, plain-string system prompt living in
``fund/agents/prompts.py`` (no ``deepagents`` / ``langchain`` import — the prompts
stay a cheap, agent-stack-free text layer; the agent is assembled in
``agents/graph.py``).

These tests assert the prompt contract:

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
    THEORY_CONSULTATION_PROTOCOL,
)

_ROLE_PROMPTS = {
    "PM_SYSTEM_PROMPT": PM_SYSTEM_PROMPT,
    "ECONOMIST_SYSTEM_PROMPT": ECONOMIST_SYSTEM_PROMPT,
    "ALLOCATOR_SYSTEM_PROMPT": ALLOCATOR_SYSTEM_PROMPT,
    "RISK_SYSTEM_PROMPT": RISK_SYSTEM_PROMPT,
    "EXECUTOR_SYSTEM_PROMPT": EXECUTOR_SYSTEM_PROMPT,
}

# The profiler (step 0) is bound by the same theory-consultation rule as the five
# orchestration roles, so it joins them for the consultation-protocol checks.
_ALL_SIX_PROMPTS = {**_ROLE_PROMPTS, "PROFILER_SYSTEM_PROMPT": PROFILER_SYSTEM_PROMPT}


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


def test_pm_prompt_names_the_fixed_delegation_order():
    lower = PM_SYSTEM_PROMPT.lower()
    assert "economist → allocator → risk → executor" in lower
    # Delegation is via the built-in ``task`` tool.
    assert "task" in lower
    assert "fund-orchestration" in lower


def test_pm_prompt_marks_risk_as_a_blocking_gate():
    lower = PM_SYSTEM_PROMPT.lower()
    assert "blocking" in lower
    assert "gate" in lower


def test_pm_prompt_respects_the_round_cap():
    lower = PM_SYSTEM_PROMPT.lower()
    assert "round" in lower
    assert "incomplete" in lower


def test_economist_prompt_names_its_tools_and_skills():
    lower = ECONOMIST_SYSTEM_PROMPT.lower()
    assert "get_macro_series" in lower
    assert "get_prices" in lower
    assert "macro-regime-read" in lower
    assert "views-construction" in lower


def _tool_bullet(prompt: str, tool: str) -> str:
    """The prompt's ``- `tool` — ...`` bullet, up to the next bullet or blank line."""
    start = prompt.index(f"- `{tool}`")
    end = prompt.index("\n\n", start)
    next_bullet = prompt.find("\n- `", start + 1)
    return prompt[start : min(end, next_bullet) if next_bullet != -1 else end]


def test_economist_prompt_describes_only_what_get_prices_returns():
    # get_prices returns a bounded per-ticker statistics block, never the raw price
    # matrix, so the prompt must name those figures and must not promise more.
    bullet = _tool_bullet(ECONOMIST_SYSTEM_PROMPT, "get_prices").lower()

    for figure in ("last close", "realised volatility", "drawdown"):
        assert figure in bullet
    assert "history" not in bullet


def test_allocator_prompt_names_its_tools_and_skills():
    lower = ALLOCATOR_SYSTEM_PROMPT.lower()
    assert "universe_filter" in lower
    assert "estimate_moments" in lower
    assert "optimize_portfolio" in lower
    assert "universe-preselection" in lower
    assert "optimization-objective-map" in lower


def test_prompts_do_not_claim_views_feed_the_optimizer_prior():
    # ``optimize_portfolio`` takes only the ConstraintSet; no tool builds a
    # Black-Litterman prior from the economist's ViewSet.
    for prompt in (ECONOMIST_SYSTEM_PROMPT, ALLOCATOR_SYSTEM_PROMPT):
        flat = " ".join(prompt.lower().split())
        assert "turn them into a prior" not in flat
        assert "views as the prior" not in flat


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


def test_executor_prompt_describes_paper_orders_semantics():
    lower = EXECUTOR_SYSTEM_PROMPT.lower()
    assert "place_orders" in lower
    assert "paper" in lower
    assert "next" in lower and "close" in lower
    assert "idempotent" in lower
    assert "rebalancing-execution" in lower


def test_pm_prompt_hands_the_executor_the_current_holdings():
    # The executor binds no tool, so the PM is the only agent that can read the
    # current holdings and it must pass them on with the target weights.
    assert "get_portfolio_history" in PM_SYSTEM_PROMPT
    assert "current_holdings" in PM_SYSTEM_PROMPT


def test_executor_prompt_takes_the_current_holdings_from_the_pm():
    lower = EXECUTOR_SYSTEM_PROMPT.lower()
    assert "current_holdings" in lower
    assert "no tool" in lower


def test_pm_prompt_hands_the_executor_the_no_trade_band():
    # The band is the mandate's drift threshold, stated in the run instruction; the
    # executor has no tool to look it up, so the PM must relay it.
    flat = " ".join(PM_SYSTEM_PROMPT.lower().split())
    assert "no-trade band" in flat
    assert "run instruction" in flat


def test_executor_prompt_takes_the_no_trade_band_from_the_pm():
    lower = EXECUTOR_SYSTEM_PROMPT.lower()
    assert "no-trade band" in lower
    assert "never infer" in lower or "never invent" in lower


def test_executor_prompt_forbids_fabricating_weights():
    lower = EXECUTOR_SYSTEM_PROMPT.lower()
    assert "fabricate" in lower
    assert "weight" in lower


def test_every_role_prompt_directs_to_the_staged_theory():
    # The second load-bearing principle (todo/deep_agent.md line 7): an agent must
    # consult the staged theory, not decide from parametric memory. Every role —
    # including the profiler — carries the consultation protocol.
    for name, prompt in _ALL_SIX_PROMPTS.items():
        assert THEORY_CONSULTATION_PROTOCOL in prompt, name
        lower = prompt.lower()
        # Points at the mounted tree and the openwiki entry map.
        assert "optimizer-theory/" in lower, name
        assert "openwiki" in lower, name
        # Tells the agent to navigate it with the filesystem tools.
        assert "read_file" in lower or "grep" in lower, name


def test_theory_protocol_forbids_writing_under_the_theory_tree():
    lower = THEORY_CONSULTATION_PROTOCOL.lower()
    # Read-only: the tree is reference, never edited.
    assert "read-only" in lower
    assert "never" in lower and "write_file" in lower
    # Steers to the routing map first, then the raw chapters.
    assert "quickstart.md" in lower
    assert "docs/nn" in lower or "nn:line" in lower


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
    assert isinstance(PROFILER_SYSTEM_PROMPT, str)
