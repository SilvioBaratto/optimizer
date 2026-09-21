---
name: risk-limits-check
description: |
  Validate a proposed weight vector against the ConstraintSet — drawdown ceilings
  nu1/nu2/nu3, ESG hard gate, robust uncertainty, and a walk-forward backtest — and
  BLOCK on any breach. Use when the risk officer must approve or reject the
  allocator's portfolio. Does NOT propose or edit weights; it only passes or blocks.
---

# Risk-limits check (blocking gate)

The risk officer's step: an independent, BLOCKING validation. It never authors a
portfolio; it checks the allocator's output against the profile and returns pass/block.

## When to use
After the allocator emits weights, before the executor.

## Load-bearing rule
The optimizer / validation computes the risk numbers; the LLM only compares them to
the ConstraintSet limits and decides pass/block. It never edits weights itself.

## Procedure
1. Call `risk_check(weights, constraints)` → passed + violations (drawdown ceilings
   nu1/nu2/nu3, sector caps, ESG exclusions, bounds).
2. Call `backtest(weights, window)` walk-forward (`shuffle=False`) and check drawdown
   / tail metrics against the ceilings. Check `out_of_sample`: when it is `False` the
   panel was too short for a walk-forward fold and the metrics are in-sample — do NOT
   count them as walk-forward validation of robustness.
3. Confirm the ESG hard gate: any excluded GICS name present → immediate BLOCK.
4. For prudent profiles, confirm the robust uncertainty set was applied.
5. Emit pass, or block with the specific violations for the allocator to fix.

## Boundaries
- Never propose new weights or relax a limit to make a portfolio pass.
- An ESG breach and a capacity-ceiling breach are hard blocks — zero auto-exceptions.

## Deeper detail
The ceiling formulas, coherent-risk background, and validation gotchas live in
`reference.md` (theory `04`, `16`, `20`, `15`, `10`, `23`, `24`).
