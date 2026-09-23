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
inputs (universe, views, constraints). Weights come only from the optimizer via the
allocator's tools. The risk step is a blocking gate.

## Pipeline order (detail in reference.md)
profiler (step 0, once) → economist (regime + views) → allocator (universe +
moments + optimize) → risk (validate, BLOCKING) → executor (orders, HITL). The
profile/universe is fixed for the whole run.

## Procedure
1. Confirm an active `ConstraintSet` exists for the portfolio (else route to the
   profiler). Read the mandate: capital, base currency, drift band, HITL, benchmark.
2. Delegate each step with the `task` tool, passing the ConstraintSet + the prior
   step's report as complete instructions.
3. Enforce the round cap: stop at `max_pm_rounds` / `recursion_limit`; if hit, mark
   the run INCOMPLETE and raise a HITL checkpoint rather than looping.
4. On a risk-gate BLOCK, return to the allocator with the violations — never override.
5. Gate `place_orders` behind HITL; record every decision in the audit trail.

## Boundaries
- Never compute or edit weights; never bypass the risk gate or the HITL gate.
- Never accept a subagent report that carries hand-made weights — reject and re-task.

## Deeper detail
Pipeline table, round-cap rationale, and the load-bearing contract live in
`reference.md`, which points to the staged theory workflow pages
`optimizer-theory/openwiki/workflows/from-conditional-forecasts-to-weights.md`
(the signal → view → optimizer → weights flow this pipeline runs) and
`optimizer-theory/openwiki/workflows/portfolio-revision.md` (when and how to
rebalance) — never copied here.
