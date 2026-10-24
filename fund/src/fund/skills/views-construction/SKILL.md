---
name: views-construction
description: |
  Turn a regime/macro read into a structured ViewSet (asset- or factor-level
  relative/absolute expected-return views with a confidence), written in the
  Black-Litterman vocabulary. Use when the economist must record a directional
  opinion for the allocator and the audit trail. In this bridge the optimizer does
  not read the views, so they never tilt the weights. Emits ONLY a ViewSet; it never
  emits weights, mu, or covariance.
---

# Views construction (Black-Litterman vocabulary)

The economist's second step: express opinions as a typed `ViewSet`, not as weights.
The ViewSet is recorded in the allocator's decision for the audit trail. No tool
builds a Black-Litterman prior from it, so a view never moves the optimizer's
weights; confidence is the Idzorek confidence a Black-Litterman run would map to the
view-uncertainty matrix Omega.

## When to use
After `macro-regime-read`, when there is a defensible directional opinion.

## Load-bearing rule
The LLM emits a ViewSet only (assets/factors, relative/absolute expected return,
confidence). The optimizer computes the weights — never the LLM — and does not read
the views.

## Procedure
1. Take the regime read + macro drivers as the rationale for each view.
2. Write each view as relative ("A outperforms B by x") or absolute, with a
   confidence in [0,1] (higher confidence → tighter Omega in a Black-Litterman run).
3. For factor views, reference factor NAMES (not asset names): a Black-Litterman
   prior on a factor model is fit on factor returns.
4. Hand the ViewSet to the allocator, which records it with its decision. Never
   tell the client or the adviser that the views tilt the portfolio.

## Boundaries
- Never emit weights, mu, or a covariance matrix.
- Keep views few and defensible: each one is an audit-trail statement of the
  economist's reasoning, and the adviser reads them.

## Deeper detail
ViewSet fields, the confidence→Omega mapping, and the factor-view gotchas live in
`reference.md` (theory `03`, `21`, `05`, `12`).
