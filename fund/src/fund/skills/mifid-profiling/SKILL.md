---
name: mifid-profiling
description: |
  Optimisation inputs (runtime step 0) — read the four sections of the user's
  input form, normalise the user's free-text answers into a typed MiFIDAnswers
  record, run the inconsistency / anti-overconfidence check, then call the
  deterministic build_constraint_set mapping to produce the saved ConstraintSet
  (the risk limits the optimizer works within). Load whenever recording or
  amending a portfolio's optimisation inputs, mapping the answers to optimizer
  constraints, or reading the user's stated risk tolerance / loss capacity / ESG
  exclusions. It assesses no one and gives no advice. Enforces the load-bearing
  boundary: the LLM interprets answers into typed inputs and never invents an
  optimizer knob.
---

# Optimisation inputs — from the user's answers to a saved `ConstraintSet`

The runtime **step 0** of the fund bridge. It turns the user's answers to the input
form into the risk limits every later agent reads. Nothing here is advice or an
assessment of the user: the answers only set the limits, and the user reviews and
approves them. Two layers, and the split is load-bearing:

- **You (the LLM)** read the answers and interpret them into the typed
  `MiFIDAnswers`. You never emit an optimizer knob.
- **The deterministic mapping** (`fund.agents.profiler.build_constraint_set`) does
  the inputs→knob math: a pure, total, auditable function. The ESG gate is a hard
  block; `A = min(tolerance, capacity)` and the appetite→aversion inversion are
  enforced in code.

## Procedure

1. **Cover the four input sections** (see `QUESTION_BANK` in
   `fund.schemas.questionnaire`), in order: knowledge & experience → financial
   situation / loss capacity → investment objectives (objective, horizon,
   risk-tolerance Likert, reaction-to-loss) → ESG exclusions (optional: none is a
   valid answer). Record every section; do not skip one because the user sounds
   confident.
2. **Normalise** free-text / ambiguous answers into a typed `MiFIDAnswers` via
   `structured_call` (retry once on a malformed reply, then fall back). Keep the
   attitudinal Likert (tolerance) disjoint from the financial capacity fields.
   Never guess an answer the user did not give: list it in `unanswered` (the
   mapping applies the most conservative value and flags it), and keep any ESG
   preference that is not a GICS sector in `esg.other_preferences`. When sectors
   are excluded, ask what share of a *fund* may sit in one: that fraction in
   `[0, 1]` is `esg.look_through_tolerance`, and without a stated figure it stays
   0 and joins `unanswered`.
3. **Check consistency** — flag contradictions (e.g. "max growth" + "cannot lose
   anything") and overconfidence. Flag, never clamp; an ESG exclusion that leaves
   nothing to hold blocks outright.
4. **Map** with `build_constraint_set(answers, portfolio_id=…, base_currency=…)`
   → a `ConstraintSet` + a `SuitabilityAssessment` (the record of how the limits
   were derived). Never write a knob yourself.
5. **Confirm, then save** — pause at the HITL gate. On the user's approval, the
   `save_profile` tool writes the `mifid_profiles` row and caches the active
   `ConstraintSet` in the Store (namespace `(portfolio_id,)`, key
   `constraint_set`) so a `ConstraintSetRef` resolves. On reject, nothing is
   written.

## Boundaries

- **Never** emit a knob/weight/number the mapping should compute.
- **Never** fuse risk tolerance (attitude) with loss capacity (finance).
- **Never** admit an ESG-excluded sector back into the universe.
- The profiler only *emits* `nu1/nu2/nu3` / `esg` / `universe_filters`; the
  allocation gate enforces them (K&E position cap in the weight bounds, ESG block
  on excluded names, loss tiers against the worst one-year loss of the backtest).

## Deeper detail

The section→knob table, the 5-band appetite→`a_gamma` lookup, the scoring formulas,
and the theory citations live in `reference.md` (sourced from `todo/deep_agent.md`
Fase 5, decisions D9/D16/D32/D34 — this skill **points to** the theory, it does not
copy it).
