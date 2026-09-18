# Macro-regime read — reference

On-demand support for `macro-regime-read`. Points to the theory; never copies it.

## Regime enum
`calm | normal | turbulent | stress` — chosen from the statistics below, not vibes.

## What the optimizer computes (the LLM only reads it)

| Statistic | Meaning | Read as | Theory |
|---|---|---|---|
| Financial turbulence | Mahalanobis distance of returns | high → turbulent | `28:22`, `28:36` |
| Two-state (Chow) mixture | calm vs turbulent covariance mix + prob | high stress prob → de-risk | `28:52`, `28:62` |
| Absorption ratio | share of variance in the top PCs | rising → fragile / systemic | `28:94` |

## Macro signals (from the DB)

| Signal | Read as | Theory |
|---|---|---|
| Yield-curve slope | inversion → recession risk | `29:26`, `29:42` |
| Credit spreads / excess bond premium | widening → stress | `29:56` |
| Realised volatility | vol-timing signal | `29:74` |
| Dynamic macro factors | compress many series into a few | `29:90` |

## Regime → cross-section / views
Regimes drive momentum/value tilts and the Black-Litterman bridge:
`26:20` (market states & momentum), `26:88` (shared regime engine across allocation
and selection), `21:114` (four interpretable regimes), `21:128` (nowcasting the
macro state), `21:152` (regime → BL view).
