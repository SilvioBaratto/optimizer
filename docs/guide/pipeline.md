# Pipeline Overview

The optimizer library follows a linear pipeline:

```
prices → preprocessing → pre_selection → moments → views →
optimization → validation → tuning → rebalancing → pipeline
```

## Entry Points

- `run_full_pipeline()` -- Single entry point for prices to validated weights
- `run_full_pipeline_with_selection()` -- Extends with upstream stock selection

## Key Conventions

- `prices_to_returns()` runs **outside** the pipeline (changes data semantics)
- Pipeline operates on return DataFrames only
- All transformers follow the sklearn `BaseEstimator + TransformerMixin` API
- Configs are frozen `@dataclass` instances; factories create estimator objects
