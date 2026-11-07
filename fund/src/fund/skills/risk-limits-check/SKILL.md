---
name: risk-limits-check
description: |
  Validate a proposed weight vector against the ConstraintSet — drawdown ceilings
  nu1/nu2/nu3, ESG hard gate and a walk-forward backtest — and BLOCK on any breach. Use when the risk officer must approve or reject the
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
1. Call `risk_check()` — it takes no arguments and judges the allocator's latest audited
   proposal → passed + violations (weight bounds and position cap, ESG sector
   exclusions, complex/leveraged-instrument restrictions, and a held name outside the
   portfolio's eligible universe, reported as `not_eligible` with its `reason`; an
   instrument it cannot classify is a violation, not a pass), and the drawdown
   ceilings: it backtests the
   same weights and a loss past nu2 or nu3 is a violation (`drawdown_nu2` /
   `drawdown_nu3`), past nu1 only a `warnings` entry. The ceilings come from the
   client's one-year loss tolerance, so the figure compared is the worst loss inside
   any one-year window of the history (`worst_rolling_loss`), not the multi-year
   peak-to-trough. When the loss cannot be measured (a common history shorter than one
   year, too little common history, or a holding with under half the panel's median
   history) the ceilings were not checked and the verdict carries a blocking
   `drawdown_unverified` violation; report it as a block, never as ceilings met. A
   client who tolerates no loss at all (nu2 = 0) is reported as `zero_loss_tolerance`:
   no invested book can pass, so send it back to the profile (capacity.max_1yr_loss_pct)
   rather than to the allocator.
2. Call `backtest()` (also no arguments) walk-forward (`shuffle=False`) to read the
   numbers behind that verdict. It errors,
   rather than evaluating a different portfolio, when the proposal does not sum to
   the budget or holds a ticker with no prices. It returns
   `drawdown_tiers`: the client's ceilings nu1 (soft warning), nu2 (hard ceiling) and
   nu3 (absolute stop), the worst loss inside any one-year window of the whole history
   (`worst_rolling_loss`, `None` with `verified: false` when the history is shorter
   than a year), the full-history peak-to-trough (`max_drawdown`, information only: it
   can span several years the tolerance does not cover), and the `breached` tiers — a
   breach of nu2 or nu3 is a BLOCK, nu1 is a warning to report. Check
   `out_of_sample`: when it is `False` the metrics are in-sample — the panel was too
   short for a walk-forward fold, or the weights were fitted on the same bars the
   test blocks cover (always the case for the allocator's optimizer weights) — so do
   NOT count them as walk-forward validation of robustness; treat them as a
   consistency check only.
   The walk-forward `max_drawdown`, `full_sample_max_drawdown` and `worst_rolling_loss`
   are all compounded wealth-path declines (`drawdown_basis`), the basis `risk_check`
   compares with nu1/nu2/nu3 (on `worst_rolling_loss`).
   The optimizer does not enforce those ceilings; this gate is where they bind.
3. Confirm the ESG hard gate: any excluded GICS name present → immediate BLOCK.
4. The robust uncertainty level of a prudent profile is recorded but not applied to
   the optimizer; never report it as applied.
5. Emit pass, or block with the specific violations for the allocator to fix.

## Boundaries
- Never propose new weights or relax a limit to make a portfolio pass.
- An ESG breach and a capacity-ceiling breach are hard blocks — zero auto-exceptions.

## Deeper detail
The ceiling formulas, coherent-risk background, and validation gotchas live in
`reference.md` (theory `04`, `16`, `20`, `15`, `10`, `23`, `24`).
