---
name: risk-limits-check
description: |
  Validate every candidate of the allocator's candidate set against the ConstraintSet
  — drawdown ceilings nu1/nu2/nu3, ESG hard gate and a walk-forward backtest — and
  BLOCK each one that breaches. Use when the risk officer must report pass or fail
  for each candidate. Does NOT propose or edit weights and never chooses a
  candidate; it only passes or blocks each one.
---

# Risk-limits check (blocking gate)

The risk officer's step: an independent, BLOCKING validation of each candidate. It
never authors a portfolio and never chooses between candidates; it checks every
candidate the allocator computed against the profile and returns pass/block for each.

## When to use
After the allocator reports its candidate set, before the executor.

## Load-bearing rule
The optimizer / validation computes the risk numbers; the LLM only compares them to
the ConstraintSet limits and reports pass/block per candidate. It never edits weights
itself and never says which candidate the adviser should take.

## Procedure
1. For each candidate id in order (`hrp`, `mean_risk`, `min_risk`, `risk_parity`),
   call `risk_check(candidate_id)` — it takes the id, never weights, and judges that
   candidate's audited weights from the latest set → passed + violations (weight
   bounds and position cap, ESG sector exclusions, complex/leveraged-instrument
   restrictions, and a held name outside the portfolio's eligible universe, reported
   as `not_eligible` with its `reason`; an instrument it cannot classify is a
   violation, not a pass), and the drawdown ceilings: it backtests the same weights
   and a loss past nu2 or nu3 is a violation (`drawdown_nu2` / `drawdown_nu3`), past
   nu1 only a `warnings` entry. The ceilings come from the client's one-year loss
   tolerance, so the figure compared is the worst loss inside any one-year window of
   the history (`worst_rolling_loss`), not the multi-year peak-to-trough. When the
   loss cannot be measured (a common history shorter than one year, too little
   common history, or a holding with under half the panel's median history) the
   ceilings were not checked and the verdict carries a blocking `drawdown_unverified`
   violation; report it as a block, never as ceilings met. A client who tolerates no
   loss at all (nu2 = 0) is reported as `zero_loss_tolerance`: no invested book can
   pass, so send it back to the profile (capacity.max_1yr_loss_pct) rather than to
   the allocator. It records nothing: the verdict a ticket needs is recorded only
   when the adviser chooses a candidate, re-checked under the mandate in force then.
   An infeasible candidate the method could not solve has no weights to judge and
   is refused by id; report it as blocked with its breaches.
2. Call `backtest(candidate_id)` for each candidate, walk-forward
   (`shuffle=False`), to read the numbers behind that verdict. It errors, rather
   than evaluating a different portfolio, when the candidate does not sum to the
   budget or holds a ticker with no prices. It returns `drawdown_tiers`: the
   client's ceilings nu1 (soft warning), nu2 (hard ceiling) and nu3 (absolute stop),
   the worst loss inside any one-year window of the whole history
   (`worst_rolling_loss`, `None` with `verified: false` when the history is shorter
   than a year), the full-history peak-to-trough (`max_drawdown`, information only:
   it can span several years the tolerance does not cover), and the `breached` tiers
   — a breach of nu2 or nu3 is a BLOCK, nu1 is a warning to report. Check
   `out_of_sample`: when it is `False` the metrics are in-sample — the panel was too
   short for a walk-forward fold, or the weights were fitted on the same bars the
   test blocks cover (always the case for the candidates' weights) — so do NOT count
   them as walk-forward validation of robustness; treat them as a consistency check
   only.
   The walk-forward `max_drawdown`, `full_sample_max_drawdown` and `worst_rolling_loss`
   are all compounded wealth-path declines (`drawdown_basis`), the basis `risk_check`
   compares with nu1/nu2/nu3 (on `worst_rolling_loss`).
   The optimizer does not enforce those ceilings; this gate is where they bind.
3. Confirm the ESG hard gate: any excluded GICS name present → immediate BLOCK of
   that candidate.
4. The robust uncertainty level of a prudent profile is recorded but not applied to
   the optimizer; never report it as applied.
5. Emit pass or block for every candidate, in id order, each block with its specific
   violations for the allocator to fix. A pass says a candidate is within the
   limits, nothing more.

## Boundaries
- Never propose new weights or relax a limit to make a candidate pass.
- Never choose, rank or single out a candidate: the adviser chooses at the gate.
- An ESG breach and a capacity-ceiling breach are hard blocks — zero auto-exceptions.

## Deeper detail
The ceiling formulas, coherent-risk background, and validation gotchas live in
`reference.md` (theory `04`, `16`, `20`, `15`, `10`, `23`, `24`).
