---
name: macro-regime-read
description: |
  Read macro/FRED series from the DB and label the current market regime for the
  economist. Use when the run needs a regime read (calm vs turbulent, expansion vs
  stress) to condition views and de-risking. Interprets the optimizer's regime
  output and the macro series; it NEVER fabricates regime probabilities or weights.
---

# Macro-regime read

The economist's first step: turn raw macro/FRED series plus the optimizer's regime
signal into a short, typed regime label the views step can act on.

## When to use
Start of the economist step, before `views-construction`.

## Load-bearing rule
The optimizer computes the regime statistics (turbulence, absorption ratio, Chow
mixture probability); the LLM only INTERPRETS them into a label and a narrative.
Never invent a probability, an expected return, or a weight.

## Procedure
1. Pull the series with `get_macro_series(names, asof)` (yield curve, credit
   spreads, realised vol, ...). Large frames stay in the tool; read the summary.
2. Read the optimizer regime output (turbulence / absorption ratio / mixture prob).
3. Map to a regime enum (see reference.md): calm / normal / turbulent / stress.
4. Emit a typed regime read (label + drivers + confidence) for `views-construction`.

## Boundaries
- Never output weights, expected returns, or a hand-made probability.
- Never overrule the optimizer's statistic — annotate it, don't replace it.

## Deeper detail
The regime enum, the turbulence/absorption/mixture reading, and the macro-signal
crosswalk live in `reference.md` (theory `28`, `29`, `26`, `21`).
