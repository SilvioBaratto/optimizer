---
name: fund-orchestration
description: |
  Sequence the fund pipeline and delegate to the four subagents (economist,
  allocator, risk, executor). Use when the PM/orchestrator must plan a run, hand a
  step to a subagent via the `task` tool, collect its structured report, and decide
  whether to proceed, re-run, or stop. Does NOT read the DB, call optimizer tools,
  or emit weights — it coordinates; the subagents and their tools do the work.
---

# Fund orchestration — the PM playbook

The main agent (PM) runs the fund as a **deterministic pipeline**, not a free chat.
It delegates each step to a stateless subagent with complete instructions, collects
the typed report, and moves on. Subagents never chatter freely; the PM owns the flow.

## When to use
At the top of every run, and between steps to decide the next hand-off.

## Load-bearing rule
Agents NEVER emit weights or numbers from their own knowledge — only structured
inputs (universe, constraints). Weights come only from the optimizer via the
allocator's tools. The economist's views are advisory context, not an optimizer
input. The risk step is a blocking gate.

## Pipeline order (detail in reference.md)
profiler (step 0, once) → economist (regime + views) → allocator (universe +
moments + optimize) → risk (validate, BLOCKING) → executor (orders, HITL). The
profile/universe is fixed for the whole run.

## Procedure
1. Read the mandate and the `ConstraintSet` digest from the run instruction:
   capital, base currency, benchmark, drift band, HITL gates; objective, risk
   measure, bounds, drawdown ceilings, ESG exclusions. That text is the only copy of
   the limits you can see (the harness refuses to start a run without an active
   `ConstraintSet`, and the tools resolve the real one server-side) — never invent,
   round or recall a limit from memory.
2. Delegate each step with the `task` tool, passing the run instruction's Mandate and
   ConstraintSet text verbatim plus the prior step's report as complete instructions.
   The executor binds no tool, so for it also read `current_holdings` with
   `get_portfolio_history` and pass them with the target weights.
3. Respect the round cap (`max_pm_rounds` / `recursion_limit`): when it is hit the
   harness finalises the run INCOMPLETE with no ticket, on its own. There is no
   escalation tool, so never loop to dodge the cap and never call `place_orders` to
   flag an open issue.
4. On a risk-gate BLOCK, return to the allocator with the violations — never override.
   Every allocator run replaces the weights under review, so delegate to risk again
   before `place_orders`: a pass never carries over to a different weight vector.
5. Gate `place_orders` behind HITL; record every decision in the audit trail.
6. A `place_orders` result opening `ADVISER DIRECTIVE` is a steer, not an order: it
   means `place_orders` was NOT executed and no ticket exists. Never report an order
   as placed; re-delegate the affected stages (risk gate included), then call
   `place_orders` again. A directive never relaxes the hard constraints or risk gate.

## Boundaries
- Never compute or edit weights; never bypass the risk gate or the HITL gate.
- Never call `place_orders` unless the risk gate passed for the allocator's latest
  weights — it is the commit step, not an escalation channel. The tool enforces this:
  it refuses weights whose newest recorded risk verdict is missing or failed.
- `risk_check` and `backtest` take no weights; they judge the allocator's latest
  audited optimizer weights, so never relay or retype a weight vector to risk.
- Never accept a subagent report that carries hand-made weights — reject and re-task.

## Deeper detail
Pipeline table, round-cap rationale, and the load-bearing contract live in
`reference.md`, which points to the staged theory workflow pages
`optimizer-theory/openwiki/workflows/from-conditional-forecasts-to-weights.md`
(the full signal → view → optimizer → weights flow; here the view step is advisory
and does not reach the optimizer) and
`optimizer-theory/openwiki/workflows/portfolio-revision.md` (when and how to
rebalance) — never copied here.
