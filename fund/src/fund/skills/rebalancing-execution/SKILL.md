---
name: rebalancing-execution
description: |
  Translate approved target weights into paper orders under a no-trade band and an
  L1 turnover penalty, then place them behind human approval. Use when the executor
  must rebalance from current to target holdings. Emits orders only; it never
  re-optimizes or invents weights, and never places live (real-money) trades.
---

# Rebalancing & execution (paper, HITL)

The executor's step: go from approved target weights to a simulated order ticket,
trading only when it is worth it, and only after human approval.

## When to use
After `risk-limits-check` passes, at the end of a run.

## Load-bearing rule
The target weights come from the optimizer, not the LLM. The executor only turns the
current→target delta into orders under the band + turnover penalty. Never re-weight.

## Procedure
1. Compute drift = L1 sum of |current - target|. If below the profile's no-trade
   band, SKIP — churn is not free.
2. If over the band, size the trade toward target with the L1 turnover penalty (a
   partial move; the optimizer already applied the penalty in the objective).
3. Call `place_orders(weights, portfolio_id)` — an idempotent paper ticket, gated by
   HITL (`interrupt_on`). On approval it writes the simulated ticket; on reject,
   nothing is written.
4. Fill model: next close + estimated slippage + commissions (no look-ahead).

## Boundaries
- Never re-optimize or emit new target weights; never place live trades (paper only).
- Never trade inside the no-trade band; never bypass the HITL gate on `place_orders`.

## Deeper detail
The no-trade band, turnover math, and the fill model live in `reference.md`
(theory `22`, `11`, `07`, `30`).
