---
name: rebalancing-execution
description: |
  Translate approved target weights into a paper rebalance proposal under a no-trade
  band and an L1 turnover penalty, for the PM to commit behind human approval. Use
  when the executor must rebalance from current to target holdings. Emits a proposal
  only; it never re-optimizes or invents weights, and never places live (real-money)
  trades.
---

# Rebalancing & execution (paper, HITL)

The executor's step: go from approved target weights to a rebalance proposal for a
simulated order ticket, trading only when it is worth it, and only after human
approval.

## When to use
After `risk-limits-check` passes, at the end of a run.

## Load-bearing rule
The target weights come from the optimizer, not the LLM. The executor only turns the
current→target delta into orders under the band + turnover penalty. Never re-weight.

## Procedure
1. Take the target weights, the portfolio's `current_holdings` and the no-trade band
   (the mandate's L1 drift threshold) from the PM's task instructions. You bind no
   tool, so if any is missing ask the PM — never infer a holding or invent a weight
   or a band.
2. Compute drift = L1 sum of |current - target|. If below the no-trade band you were
   handed, recommend SKIP — churn is not free.
3. If over the band, size the trade toward target with the L1 turnover penalty (a
   partial move; the optimizer already applied the penalty in the objective).
4. Report the proposal for the PM: the per-name deltas, the drift, and the band
   verdict. The PM commits it with its own `place_orders` — an idempotent paper
   ticket, gated by HITL (`interrupt_on`) that writes only on approval. You never
   call it.
5. Fill model: next close + estimated slippage + commissions (no look-ahead).

## Boundaries
- Never re-optimize or emit new target weights; never place live trades (paper only).
- Never trade inside the no-trade band; never bypass the HITL gate on the PM's
  `place_orders`.

## Deeper detail
The no-trade band, turnover math, and the fill model live in `reference.md`
(theory `22`, `11`, `07`, `30`).
