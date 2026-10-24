# MiFID profiling — pillar→knob reference

On-demand support for the `mifid-profiling` skill. The numbers below are what the
**deterministic mapping** (`fund.agents.profiler`) applies — they are here so you
can explain a profile, **not** so you can type a knob yourself. Theory citations
are `todo/deep_agent.md` line refs (`file:line`), preserved from the roadmap.

## The four pillars → `ConstraintSet` knobs

| MiFID pillar | Sub-datum (typed answer) | → knob | Theory |
|---|---|---|---|
| Risk **tolerance** (attitudinal) | `objectives.likert_items` | risk-**appetite** `A_tol` → `a_gamma` | `01:142`, `30:41` |
| Loss **capacity** (financial) | `capacity.max_1yr_loss_pct`, `capacity.buffer_months` | risk-**appetite** `A_cap` → `a_gamma` + `nu1/nu2/nu3` | `20:75` |
| **Binding rule** | conform to tolerance **AND** capacity | `A = min(tolerance, capacity)` → `a_gamma` | regulatory prudence |
| Reaction to extreme loss | `objectives.loss_reaction` | `risk_measure` + `beta` | `04:18`, `16:22`, `20:29` |
| Horizon | `objectives.horizon` | `horizon` (passthrough) | `11:175`, `30:24` |
| Objective | `objectives.goal` | `objective` (passthrough) | `15:80`, `17:89` |
| Knowledge & experience (low) | `knowledge.level` | `universe_filters` (no_complex, no_leverage, cap) | `06:41`, `06:274` |
| ESG | `esg.exclusions` | `esg` (**hard gate**, exclusions-only, D16): `universe_filter` drops the sectors' names and `risk_check` blocks any weight on them | fund-scope |

## Appetite → aversion (correctness-critical inversion)

`a_gamma` feeds `MeanRisk(risk_aversion=…)` where **higher = more conservative**.
`A = min(tolerance, capacity)` is on **risk-appetite** `A ∈ [0, 1]` (higher = can
take more risk); take the min, then map **monotonically decreasing** to `a_gamma`.
A naive `a_gamma = min(γ_tol, γ_cap)` would be backwards.

**5-band discrete lookup** (`_AVERSION_BANDS`; the band name is recorded in the
`SuitabilityAssessment`):

| appetite | category (`RiskToleranceBand`) | `a_gamma` |
|---|---|---|
| [0.0, 0.2) | Defensive | 12.0 |
| [0.2, 0.4) | Conservative | 8.0 |
| [0.4, 0.6) | Balanced | 5.0 |
| [0.6, 0.8) | Growth | 2.5 |
| [0.8, 1.0] | Aggressive | 1.0 |

**Scoring** (all linear rescales, auditable):
- `a_tol = mean(likert_items)` rescaled `(x - 1) / 6`.
- `a_cap = mean(clip(max_1yr_loss_pct / 0.50, 0, 1), clip(buffer_months / 12, 0, 1))`.
- `appetite = min(a_tol, a_cap)`.

## Reaction → risk_measure + beta (D34)

The more protective the reaction, the deeper the downside the profile controls.

| `loss_reaction` | `risk_measure` | `beta` |
|---|---|---|
| `sell_all` | `max_drawdown` | 0.99 |
| `sell_some` | `cdar` | 0.975 |
| `hold` | `cvar` | 0.95 |
| `buy_more` | `variance` | 0.90 |

## Knowledge → universe_filters (D32)

| `knowledge.level` | `no_complex` | `no_leverage` | `max_position_cap` |
|---|---|---|---|
| `none` | true | true | 0.05 |
| `basic` | true | true | 0.10 |
| `informed` | false | false | — |
| `advanced` | false | false | — |

The cap is not advisory: it tightens `bounds.max_weights` (the tighter of the two
wins), so it binds the optimizer and the `risk_check` ceiling alike. A cap of `c`
needs at least `ceil(1 / c)` priced names; a smaller universe is reported as
infeasible by `optimize_portfolio` rather than as an opaque solver failure.

## Drawdown tiers (`nu1/nu2/nu3`, emitted here, gated by `risk_check`)

Anchored on the stated one-year loss tolerance `L = capacity.max_1yr_loss_pct`:
`nu1 = 0.5·L` (soft warning), `nu2 = L` (hard ceiling), `nu3 = min(1.5·L, 1.0)`
(absolute stop). See `20:32` (max and average drawdown) and `20:136` (drawdown
caps as constraints). `risk_check` compares them with the max drawdown of
a backtest of the proposed weights: past `nu3` or `nu2` blocks, past `nu1` only
adds a warning. When the backtest cannot run (too little history) the tiers are
skipped and the report carries a `drawdown_unverified` warning.

## Hard rules the mapping enforces (never the LLM)

- `A = min(tolerance, capacity)` — the more cautious of attitude and finances binds.
- ESG-excluded GICS sectors are hard-excluded; no other answer can re-admit them.
  Excluding *every* sector raises `SuitabilityBreachError` (no investable universe).
- Tolerance (attitude) and capacity (finance) are computed from **disjoint** answer
  fields — never fused.

## Inconsistency flags (flag, never clamp)

The suitability check records a named flag per contradiction and leaves `a_gamma`
untouched; the adviser sees the flags at the HITL gate. Thresholds are on the same
appetite scale `[0, 1]` as the bands above.

| Flag | Fires when |
|---|---|
| `objective_capacity_mismatch` | `objectives.goal` is `growth` or `max` **and** the weakest of `A_tol`, `A_cap` and the stated loss-capacity sub-score `clip(max_1yr_loss_pct / 0.50)` is below 0.4 |
| `overconfidence` | `A_tol - A_cap >= 0.4` (attitude far outstripping finances) |
| `reaction_tolerance_mismatch` | `A_tol >= 0.6` **and** `loss_reaction` is `sell_all` |
| `unanswered_input:<datum>` | the client left that field unanswered (`MiFIDAnswers.unanswered`); one flag per datum, and the field holds its conservative placeholder |
| `unenforced_esg_preference` | `esg.other_preferences` is non-empty: a preference no GICS sector expresses, shown to the adviser but not enforced |

A growth/max objective ignores `a_gamma` in the optimizer, so the first flag is the
only guard against it. It tests each pillar on its own as well as the blend because
the blended `A_cap` lets a large buffer mask a stated 0% loss capacity, and the
attitude can be the weakest pillar even when the finances are ample.

An unanswered field takes its most conservative value (lowest appetite, narrowest
universe, strongest tail control) so the band can never be lifted by a guess. That
placeholder is not something the client said, so the three contradiction rules only
read scores and answers whose inputs were all stated: an unanswered pillar shows up
as `unanswered_input`, never as a contradiction the client did not make.
