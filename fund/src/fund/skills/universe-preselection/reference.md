# Universe pre-selection — reference

On-demand support for `universe-preselection`. Points to the theory; never copies it.

## Filter enums (the LLM chooses; the pipeline applies)

| Filter | Knob | Theory |
|---|---|---|
| Liquidity floor | min ADV / price | screens: `13:160` |
| Factor score / style | size, value, quality tilts | `09:100`, `09:116` |
| Signal → alpha discipline | scale / trim / neutralize | `13:142` |
| Drop-correlated | correlation cap (condition number) | `05:71`, `08:108` |
| Sector caps | GICS group bounds | `13:108` |
| K&E exclusions | no-complex / no-leverage (from the profile) | mifid ConstraintSet |

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
