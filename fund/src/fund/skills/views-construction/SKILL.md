---
name: views-construction
description: |
  Turn a regime/macro read into a structured ViewSet (asset- or factor-level
  relative/absolute expected-return views with a confidence) that the allocator
  carries as advisory context. Use when the economist must express a directional
  opinion for the allocator and the audit rationale. Emits ONLY a ViewSet; it never
  emits weights, mu, or covariance.
---

# Views construction (advisory)

The economist's second step: express opinions as a typed `ViewSet`, not as weights.
The schema is Black-Litterman-shaped (confidence maps to the view-uncertainty
matrix Omega), but today the views are advisory.

## Views are advisory today
No bound tool accepts views: `estimate_moments` and `optimize_portfolio` take only a
`universe`, so the optimizer runs on its own prior (empirical mean, Ledoit-Wolf
covariance) and the weights are identical with or without a ViewSet.
`ViewSet.to_black_litterman_config` is the bridge for a future wiring; no run calls
it. Never state or imply that a view moved the weights.

## When to use
After `macro-regime-read`, when there is a defensible directional opinion.

## Load-bearing rule
The LLM emits a ViewSet only (assets/factors, relative/absolute expected return,
confidence). The optimizer computes the weights — never the LLM.

## Procedure
1. Take the regime read + macro drivers as the rationale for each view.
2. Write each view as relative ("A outperforms B by x") or absolute, with a
   confidence in [0,1] (it would become Omega once wired: higher confidence →
   tighter Omega).
3. Prefer asset-level views. A factor view references a factor NAME, not an asset
   name, and would need a factor-model prior that is not built today.
4. Hand the ViewSet to the allocator as context for its rationale and for the
   adviser; the allocator cannot pass it to the optimizer.

## Boundaries
- Never emit weights, mu, or a covariance matrix.
- Never cite a view as the cause of a position or a weight change.
- Keep views few and defensible; they are read by people, not priced by the optimizer.

## Deeper detail
ViewSet fields, the confidence→Omega mapping, and the factor-view gotchas live in
`reference.md` (theory `03`, `21`, `05`, `12`).
