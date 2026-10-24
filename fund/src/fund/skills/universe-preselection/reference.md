# Universe pre-selection — reference

On-demand support for `universe-preselection`. Points to the theory; never copies it.

## What the bound tools take
`universe_filter(universe)` and `estimate_moments(universe)` take only the ticker
list. The run date, the ConstraintSet and every threshold come from the run, not from
the LLM.

## Stages the filter applies (fixed defaults)

Order: validate -> outliers -> drop incomplete history -> impute -> drop zero
variance -> drop correlated. `top_k`, Pareto and expiry selection exist in the
library but are off in the bound tool.

| Stage | Setting | Theory |
|---|---|---|
| Drop-correlated | pairwise correlation cap 0.95 (fixed, not LLM-set) | `05:71`, `08:108` |
| Incomplete history | assets with leading/trailing gaps are dropped | - |

## Mandate screen (applied by the bound tool)
After the stages above, `universe_filter` removes names that breach the client's
ConstraintSet: ESG-excluded GICS sectors, and for a low-knowledge client the K&E
no-complex / no-leverage instrument restrictions. An instrument whose sector cannot
be classified fails closed. Removed names are returned as `mandate_excluded`.

## Not modelled (do not claim these were applied)

| Screen | Status | Theory |
|---|---|---|
| Liquidity floor (min ADV / price) | not available, no volume data is loaded | `13:160` |
| Factor score / style tilts | not available in the filter | `09:100`, `09:116` |
| Signal -> alpha discipline | not available in the filter | `13:142` |
| Sector caps (GICS group bounds) | not available, no sector mapping is passed | `13:108` |

## Factor structure & conditioning
Full covariance is ill-conditioned on ~8898 names → use factor structure + shrinkage:
`05:121` (K-factor covariance decomposition), `08:66` (the shrinkage principle),
`08:108` (linear covariance shrinkage). Factor cyclicality / rotation context:
`27:22`, `27:74`.

## Pipeline gotchas (repo)
- `build_portfolio_pipeline(optimizer, ...)` FLATTENS pre-selection + optimizer into
  one sklearn Pipeline; nested params like `drop_correlated__threshold` are exposed.
- `sector_mapping` is injected as a plain `dict[str, str]`, not queried in the tool.
- Feed linear returns (`prices_to_returns`); `shuffle=False` in any CV.
