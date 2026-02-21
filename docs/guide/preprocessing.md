# Preprocessing

Data cleaning transformers for return matrices.

## Transformers

- **DataValidator** -- Replaces `inf` and extreme returns with `NaN`
- **OutlierTreater** -- Three-group z-score methodology (remove / winsorize / keep)
- **SectorImputer** -- Leave-one-out sector-average NaN imputation
- **RegressionImputer** -- OLS regression from top-K correlated assets with sector fallback

All transformers implement the sklearn `fit`/`transform` API and compose in `sklearn.pipeline.Pipeline`.
