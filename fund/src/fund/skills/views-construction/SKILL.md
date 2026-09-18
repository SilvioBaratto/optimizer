---
name: views-construction
description: |
  Turn a regime/macro read into a structured ViewSet (asset- or factor-level
  relative/absolute expected-return views with a confidence) for Black-Litterman.
  Use when the economist must express a directional opinion the optimizer can price
  into the prior. Emits ONLY a ViewSet; it never emits weights, mu, or covariance.
---

# Views construction (Black-Litterman)

The economist's second step: express opinions as a typed `ViewSet`, not as weights.
The optimizer folds views into the Black-Litterman posterior; confidence maps to the
view-uncertainty matrix Omega.

## When to use
After `macro-regime-read`, when there is a defensible directional opinion.

## Load-bearing rule
The LLM emits a ViewSet only (assets/factors, relative/absolute expected return,
confidence). The optimizer computes the posterior and the weights — never the LLM.

## Procedure
1. Take the regime read + macro drivers as the rationale for each view.
2. Write each view as relative ("A outperforms B by x") or absolute, with a
   confidence in [0,1] that becomes Omega (higher confidence → tighter Omega).
3. For factor views, reference factor NAMES (not asset names): the prior is a factor
   model fit on factor returns.
4. Hand the ViewSet to the allocator; the optimizer runs Black-Litterman.

## Boundaries
- Never emit weights, mu, or a covariance matrix.
- Keep views few and defensible; over-viewing just overrides the equilibrium prior.

## Deeper detail
ViewSet fields, the confidence→Omega mapping, and the factor-view gotchas live in
`reference.md` (theory `03`, `21`, `05`, `12`).
