---
name: rebalancing-execution
description: |
  Describe, for each feasible candidate, the paper rebalance from the current holdings
  under the mandate's drift band (L1), so the adviser can compare them before choosing
  one at the PM's gate. Use when the executor must lay out the trades each candidate
  would need. Describes trades only; it never re-optimizes, invents weights or
  chooses a candidate, and never places live (real-money) trades.
---

# Rebalancing & execution (paper, HITL)

The executor's step: go from each feasible candidate's weights to a description of
the simulated order ticket it would need, so the adviser sees what choosing it would
trade. The adviser chooses the candidate; the PM's gate places it.

## When to use
After `risk-limits-check` has reported on the candidates, at the end of a run.

## Load-bearing rule
The candidates' weights come from the optimizer, not the LLM. The executor only turns
each current→candidate delta into a description under the drift band. Never re-weight
and never choose a candidate.

## Procedure
1. Take the candidates' weights, the portfolio's `current_holdings` (the PM reads them
   with `get_portfolio_history`) and the drift band (L1) (the mandate's L1 drift
   threshold) from the PM's task instructions. You bind no tool, so if any is missing
   ask the PM — never infer a holding or invent a weight or a band.
2. For each feasible candidate, in id order, compute drift = L1 sum of
   |current - candidate| and state factually whether it is under the drift band you
   were handed (a move inside the band trades little for its cost). With no holdings
   (a first rebalance) there is nothing to drift from: skip the band test and
   describe the full candidate book.
3. Present each candidate's per-name current→candidate deltas side by side, in the
   same terms for each. The ticket books the chosen candidate's full target book,
   not a netted or partial move: `place_orders` takes no holdings or band input and
   books the adviser's chosen weights as they are.
4. Report the descriptions for the PM: per candidate, the per-name deltas, the drift
   and whether it is under the band. Candidates that are identical are reported as
   such. The PM's `place_orders` shows every candidate to the adviser, who chooses
   one — an idempotent paper ticket, gated by HITL (`interrupt_on`), written only
   for the adviser's choice. You never call it.
5. Fill model: next close + estimated slippage + commissions (no look-ahead).

## Boundaries
- Never re-optimize or emit new weights; never place live trades (paper only).
- Never choose, rank or single out a candidate; never bypass the HITL gate on the
  PM's `place_orders`.
- No turnover control is applied: each candidate is a plain long-only solve with no
  previous weights or transaction costs, so never describe one as
  turnover-penalised or as a staged step toward its target.

## Deeper detail
The drift band and turnover theory, and the fill model, live in `reference.md`
(theory `22`, `11`, `07`, `30`).
