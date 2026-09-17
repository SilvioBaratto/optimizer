---
name: mifid-profiling
description: |
  MiFID II suitability profiling (runtime step 0) — conduct the four-pillar
  questionnaire, normalise a client's free-text answers into a typed MiFIDAnswers
  record, run the inconsistency / anti-overconfidence check, then call the
  deterministic build_constraint_set mapping to produce the persisted
  ConstraintSet. Load whenever profiling a client, building or amending a risk
  profile, mapping a suitability questionnaire to optimizer constraints, or
  deciding a client's risk tolerance / loss capacity / ESG exclusions. Enforces
  the load-bearing boundary: the LLM interprets answers into typed inputs and
  never invents an optimizer knob.
---

# MiFID profiling — from questionnaire to persisted `ConstraintSet`

The runtime **step 0** of the fund bridge. It turns a MiFID II suitability
questionnaire into the risk profile every later agent reads. Two layers, and the
split is load-bearing:

- **You (the LLM)** conduct the questionnaire and interpret answers into the typed
  `MiFIDAnswers`. You never emit an optimizer knob.
- **The deterministic mapping** (`fund.agents.profiler.build_constraint_set`) does
  the MiFID→knob math: a pure, total, auditable function. The ESG gate is a hard
  block; `A = min(tolerance, capacity)` and the appetite→aversion inversion are
  enforced in code.

## Procedure

1. **Conduct the four ESMA pillars** (see `QUESTION_BANK` in
   `fund.schemas.questionnaire`), in order: knowledge & experience → financial
   situation / loss capacity → investment objectives (objective, horizon,
   risk-tolerance Likert, reaction-to-loss) → ESG exclusions. Assess every pillar;
   do not skip one because the client sounds confident.
2. **Normalise** free-text / ambiguous answers into a typed `MiFIDAnswers` via
   `structured_call` (retry once on a malformed reply, then fall back). Keep the
   attitudinal Likert (tolerance) disjoint from the financial capacity fields.
3. **Check consistency** — flag contradictions (e.g. "max growth" + "cannot lose
   anything") and overconfidence. Flag, never clamp; an ESG/legal breach
   hard-blocks.
4. **Map** with `build_constraint_set(answers, portfolio_id=…, base_currency=…)`
   → a `ConstraintSet` + a `SuitabilityAssessment`. Never write a knob yourself.
5. **Confirm, then persist** — pause at the HITL gate. On adviser approval, the
   `save_profile` tool writes the `mifid_profiles` row and caches the active
   `ConstraintSet` in the Store (namespace `(portfolio_id,)`, key
   `constraint_set`) so a `ConstraintSetRef` resolves. On reject, nothing is
   written.

## Boundaries

- **Never** emit a knob/weight/number the mapping should compute.
- **Never** fuse risk tolerance (attitude) with loss capacity (finance).
- **Never** admit an ESG-excluded sector back into the universe.
- The profiler only *emits* `nu1/nu2/nu3` / `esg` / `universe_filters`;
  enforcement is Fase 7.

## Deeper detail

The pillar→knob table, the 5-band appetite→`a_gamma` lookup, the scoring formulas,
and the theory citations live in `reference.md` (sourced from `todo/deep_agent.md`
Fase 5, decisions D9/D16/D32/D34 — this skill **points to** the theory, it does not
copy it).
