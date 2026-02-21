# Pre-Selection

Assemble data cleaning and asset filtering into a single sklearn `Pipeline`.

The pre-selection module takes a raw return `DataFrame`, cleans it (validation,
outlier treatment, imputation), and then progressively narrows the asset
universe through a series of skfolio selectors. The result is a tidy,
NaN-free `DataFrame` containing only the assets that pass every filter --
ready to feed into moment estimation and portfolio optimization.

---

## Overview

The module follows the same **frozen dataclass config + factory function**
pattern used throughout the optimizer library:

| Component | Role |
|-----------|------|
| `PreSelectionConfig` | Frozen `@dataclass` holding every pipeline parameter as a plain primitive, enum, or `None`. Serialisable and suitable for hyperparameter sweeps. |
| `build_preselection_pipeline()` | Factory function that reads a `PreSelectionConfig` and returns a fully assembled `sklearn.pipeline.Pipeline`. |

Because the config stores only primitives, it can be serialised to JSON/YAML,
persisted to a database, or passed across process boundaries without issue.
Non-serialisable objects (such as the `sector_mapping` dictionary) are passed
as keyword arguments to the factory, not stored in the config.

```python
from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline

config = PreSelectionConfig(correlation_threshold=0.90, top_k=30)
pipeline = build_preselection_pipeline(config, sector_mapping={"AAPL": "Tech", "JPM": "Financials"})

clean_returns = pipeline.fit_transform(returns_df)
```

---

## Pipeline Steps

`build_preselection_pipeline` assembles the following steps **in this exact
order**. The first six steps are always present; the last three are
conditional on config flags.

```
validate --> outliers --> impute --> SelectComplete --> DropZeroVariance
--> DropCorrelated --> [SelectKExtremes] --> [SelectNonDominated]
--> [SelectNonExpiring]
```

Steps in brackets are **optional** and only added when the corresponding
config parameter is set.

### 1. `validate` -- DataValidator

Replaces `inf`, `-inf`, and returns whose absolute value exceeds
`max_abs_return` with `NaN`. This is a stateless transformer that acts as a
first-pass sanity check, catching data errors (e.g. a return of 50 000%)
before they corrupt downstream statistics.

| Parameter | Config field | Default |
|-----------|-------------|---------|
| `max_abs_return` | `max_abs_return` | `10.0` (i.e. 1 000%) |

!!! info "Why so generous?"
    The default threshold of 10.0 (1 000%) is deliberately high. It catches
    obvious data errors while preserving legitimate large moves such as
    penny-stock spikes or circuit-breaker events. Tighten it to 5.0 or lower
    for conservative universes.

### 2. `outliers` -- OutlierTreater

Three-group z-score methodology applied per-column:

| Group | Condition | Action |
|-------|-----------|--------|
| **Data errors** | `|z| >= remove_threshold` | Replaced with `NaN` |
| **Outliers** | `winsorize_threshold <= |z| < remove_threshold` | Winsorised to `mu +/- winsorize_threshold * sigma` |
| **Normal** | `|z| < winsorize_threshold` | Kept as-is |

The z-scores are computed from the **training data** statistics (`mu_` and
`sigma_` stored during `fit`). Constant-variance columns (sigma = 0) are
assigned a z-score of 0 and left for `DropZeroVariance` to handle.

| Parameter | Config field | Default |
|-----------|-------------|---------|
| `winsorize_threshold` | `winsorize_threshold` | `3.0` |
| `remove_threshold` | `remove_threshold` | `10.0` |

!!! warning "Validation constraint"
    `winsorize_threshold` must be **strictly less than** `remove_threshold`.
    The config raises `ValueError` at construction time if this invariant is
    violated.

### 3. `impute` -- SectorImputer

Fills remaining `NaN` values using leave-one-out sector cross-sectional
averages. For each timestep and each missing cell, the imputer computes the
mean of all *other* assets in the same sector. When the entire sector is
`NaN` for a given row, it falls back to the global cross-sectional mean.

When `sector_mapping` is `None`, all assets are treated as a single sector,
which reduces to plain global cross-sectional mean imputation.

| Parameter | Config field | Default |
|-----------|-------------|---------|
| `fallback_strategy` | `imputation_fallback` | `"global_mean"` |
| `sector_mapping` | Factory kwarg (not in config) | `None` |

!!! note "sector_mapping is a factory argument"
    The sector mapping is a `dict[str, str]` passed directly to
    `build_preselection_pipeline(sector_mapping=...)`, not stored in the
    frozen config. This keeps the config serialisable. Columns absent from
    the mapping are assigned to a catch-all `"__unmapped__"` sector.

### 4. `select_complete` -- SelectComplete

Drops any asset (column) that still contains `NaN` after imputation. In
practice, when `SectorImputer` runs correctly this step is a no-op, but it
acts as a safety net to guarantee a fully complete matrix for downstream
selectors that cannot handle missing data.

This step has **no configurable parameters**.

### 5. `drop_zero_variance` -- DropZeroVariance

Drops any asset with zero variance (constant return series). Constant
columns add no information and cause numerical issues in covariance
estimation.

This step has **no configurable parameters**.

### 6. `drop_correlated` -- DropCorrelated

Drops one asset from each pair whose pairwise correlation exceeds the
threshold. This reduces redundancy in the universe and improves
conditioning of the covariance matrix.

| Parameter | Config field | Default |
|-----------|-------------|---------|
| `threshold` | `correlation_threshold` | `0.95` |
| `absolute` | `correlation_absolute` | `False` |

!!! info "Absolute correlation"
    When `correlation_absolute=True`, the selector uses `|corr|` rather than
    raw correlation, so that strong *negative* correlations are also flagged.
    This is useful when you want to reduce all forms of linear dependence.

### 7. `select_k` -- SelectKExtremes (optional)

Only added when `top_k is not None`. Keeps the *k* assets with the highest
(or lowest) mean return, as measured by `SelectKExtremes`.

| Parameter | Config field | Default |
|-----------|-------------|---------|
| `k` | `top_k` | `None` (step omitted) |
| `highest` | `top_k_highest` | `True` |

### 8. `select_pareto` -- SelectNonDominated (optional)

Only added when `use_pareto=True`. Applies a Pareto non-dominance filter
across risk-return dimensions, retaining only assets that lie on the
efficient frontier of mean return vs. variance.

| Parameter | Config field | Default |
|-----------|-------------|---------|
| `min_n_assets` | `pareto_min_assets` | `None` |

### 9. `select_non_expiring` -- SelectNonExpiring (optional)

Only added when **both** `use_non_expiring=True` **and**
`expiration_lookahead is not None`. Removes assets that expire within the
specified lookahead window, which is relevant for futures and options
universes.

| Parameter | Config field | Default |
|-----------|-------------|---------|
| `expiration_lookahead` | `expiration_lookahead` | `None` (step omitted) |

!!! warning "Both flags required"
    Setting `use_non_expiring=True` without providing `expiration_lookahead`
    silently skips this step. The step is only added when both conditions
    are met.

---

## Configuration Reference

All fields of `PreSelectionConfig` with their types, defaults, and the
pipeline step they control:

| Field | Type | Default | Pipeline step | Description |
|-------|------|---------|---------------|-------------|
| `max_abs_return` | `float` | `10.0` | `validate` | Maximum absolute return before treating as data error |
| `winsorize_threshold` | `float` | `3.0` | `outliers` | Z-score boundary between normal observations and outliers |
| `remove_threshold` | `float` | `10.0` | `outliers` | Z-score boundary between outliers and data errors |
| `outlier_method` | `str` | `"time_series"` | `outliers` | Outlier detection approach (only `"time_series"` supported) |
| `imputation_fallback` | `str` | `"global_mean"` | `impute` | Fallback when sector data unavailable |
| `correlation_threshold` | `float` | `0.95` | `drop_correlated` | Pairwise correlation above which an asset is dropped |
| `correlation_absolute` | `bool` | `False` | `drop_correlated` | Whether to use absolute correlation values |
| `top_k` | `int | None` | `None` | `select_k` | If set, keep only the *k* assets with highest/lowest mean return |
| `top_k_highest` | `bool` | `True` | `select_k` | Select highest (`True`) or lowest (`False`) mean return |
| `use_pareto` | `bool` | `False` | `select_pareto` | Whether to apply Pareto non-dominance filter |
| `pareto_min_assets` | `int | None` | `None` | `select_pareto` | Minimum assets to retain after Pareto filtering |
| `use_non_expiring` | `bool` | `False` | `select_non_expiring` | Whether to remove soon-expiring assets |
| `expiration_lookahead` | `int | None` | `None` | `select_non_expiring` | Calendar days to look ahead for expiring assets |
| `is_log_normal` | `bool` | `True` | *(stored for downstream use)* | Whether returns are assumed log-normal for multi-period scaling |

### Validation rules

The config validates the following constraints at construction time
(`__post_init__`):

- `winsorize_threshold < remove_threshold` -- winsorisation boundary must be
  stricter than the removal boundary.
- `0.0 < correlation_threshold <= 1.0` -- must be a valid correlation value.
- `max_abs_return > 0` -- must be strictly positive.

Violating any of these raises `ValueError` immediately.

---

## Presets

`PreSelectionConfig` provides two class-method presets for common scenarios.

### `for_daily_annual()`

Sensible defaults for daily equity returns over an approximately one-year
horizon. This is equivalent to `PreSelectionConfig()` with all defaults.

```python
cfg = PreSelectionConfig.for_daily_annual()
# max_abs_return=10.0, winsorize_threshold=3.0, remove_threshold=10.0,
# correlation_threshold=0.95, is_log_normal=True
# No optional steps (top_k, pareto, non_expiring all off)
```

### `for_conservative()`

Tighter filters for a more conservative universe. Lowers the data-error
and outlier thresholds, tightens the correlation filter, and activates
`SelectKExtremes` to cap the universe at 50 assets.

```python
cfg = PreSelectionConfig.for_conservative()
# max_abs_return=5.0, winsorize_threshold=2.5, remove_threshold=8.0,
# correlation_threshold=0.85, top_k=50, top_k_highest=True,
# is_log_normal=True
```

---

## Code Examples

### Basic usage with default config

```python
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline

# Load data and convert to returns
prices = load_sp500_dataset()
returns = prices_to_returns(prices)

# Build pipeline with sensible defaults
config = PreSelectionConfig.for_daily_annual()
pipeline = build_preselection_pipeline(config)

# Fit and transform
clean_returns = pipeline.fit_transform(returns)
print(f"Input: {returns.shape[1]} assets -> Output: {clean_returns.shape[1]} assets")
```

### Conservative preset with sector-aware imputation

```python
from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline

# Sector mapping for imputation
sector_mapping = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    # ... more tickers
}

config = PreSelectionConfig.for_conservative()
pipeline = build_preselection_pipeline(config, sector_mapping=sector_mapping)
clean_returns = pipeline.fit_transform(returns)
```

### Custom configuration

```python
from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline

config = PreSelectionConfig(
    max_abs_return=5.0,              # Strict data-error threshold
    winsorize_threshold=2.5,         # Tighter winsorisation
    remove_threshold=8.0,            # Lower removal boundary
    correlation_threshold=0.90,      # Drop assets correlated above 90%
    correlation_absolute=True,       # Use |corr| (catches negative correlation too)
    top_k=30,                        # Keep top 30 by mean return
    top_k_highest=True,              # Highest mean return
    use_pareto=True,                 # Apply Pareto filter after top-k
    pareto_min_assets=15,            # Keep at least 15 assets from Pareto
)

pipeline = build_preselection_pipeline(config)
clean_returns = pipeline.fit_transform(returns)
```

### Futures universe with expiration filtering

```python
from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline

config = PreSelectionConfig(
    use_non_expiring=True,
    expiration_lookahead=90,  # Drop contracts expiring within 90 days
)

pipeline = build_preselection_pipeline(config)
clean_returns = pipeline.fit_transform(futures_returns)
```

### Inspecting and tuning pipeline parameters

```python
pipeline = build_preselection_pipeline()

# List all accessible parameters
params = pipeline.get_params()
for key in sorted(params):
    if "__" in key:
        print(f"  {key} = {params[key]}")

# Modify parameters after construction
pipeline.set_params(
    outliers__winsorize_threshold=2.5,
    drop_correlated__threshold=0.90,
    validate__max_abs_return=5.0,
)
```

### Using pre-selection inside a full optimization pipeline

```python
from skfolio.preprocessing import prices_to_returns
from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline
from optimizer.pipeline import run_full_pipeline

prices = ...  # pd.DataFrame of asset prices

# Pre-selection is handled internally by run_full_pipeline,
# but you can also run it explicitly for inspection:
config = PreSelectionConfig(correlation_threshold=0.90, top_k=50)
preselection_pipe = build_preselection_pipeline(config)

returns = prices_to_returns(prices)
clean_returns = preselection_pipe.fit_transform(returns)
print(f"Selected {clean_returns.shape[1]} assets from {returns.shape[1]}")
```

---

## Gotchas

!!! warning "Pre-selection must run inside CV folds"
    When using cross-validation (walk-forward, CPCV, etc.), the
    pre-selection pipeline **must** be part of the overall sklearn pipeline
    that gets re-fit on each training fold. If you run pre-selection once on
    the full dataset and then cross-validate, you introduce data leakage --
    the `OutlierTreater` z-score statistics and `DropCorrelated` correlation
    matrix will have been computed on data that includes the validation
    period.

    The optimizer library handles this correctly when the pre-selection
    pipeline is composed inside the broader sklearn `Pipeline` that
    `run_full_pipeline` builds.

!!! warning "Parameter names use double-underscore notation"
    All transformer hyper-parameters are accessible via `get_params()` using
    sklearn's `step_name__param_name` notation. For example:

    - `validate__max_abs_return`
    - `outliers__winsorize_threshold`
    - `outliers__remove_threshold`
    - `drop_correlated__threshold`
    - `drop_correlated__absolute`
    - `select_k__k` (only when `top_k` is set)

    This is the notation you must use for `set_params()` and for
    hyperparameter tuning grids.

!!! note "prices_to_returns runs outside the pipeline"
    The pre-selection pipeline operates on a **return** `DataFrame`, not a
    price `DataFrame`. The conversion from prices to returns
    (`skfolio.preprocessing.prices_to_returns`) changes data semantics and
    is therefore performed upstream, before the pipeline runs. This is a
    project-wide convention.

!!! note "SelectNonExpiring requires both flags"
    Setting `use_non_expiring=True` alone does **not** add the step.
    You must also provide `expiration_lookahead` (an integer number of
    calendar days). Without it, the step is silently skipped.

!!! info "The config is frozen"
    `PreSelectionConfig` is a frozen dataclass. You cannot mutate fields
    after construction. To change a parameter, create a new config instance:

    ```python
    # This raises AttributeError:
    config.correlation_threshold = 0.85

    # Do this instead:
    from dataclasses import replace
    new_config = replace(config, correlation_threshold=0.85)
    ```

---

## Quick Reference

```python
from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline

# Presets
cfg = PreSelectionConfig.for_daily_annual()    # sensible defaults
cfg = PreSelectionConfig.for_conservative()    # tighter filters, top_k=50

# Factory
pipe = build_preselection_pipeline(config=cfg, sector_mapping=None)

# Pipeline step names (default)
# validate -> outliers -> impute -> select_complete -> drop_zero_variance -> drop_correlated

# Optional steps (added when config flags are set)
# select_k            (top_k is not None)
# select_pareto       (use_pareto=True)
# select_non_expiring (use_non_expiring=True AND expiration_lookahead is not None)

# Key parameter paths for tuning
# validate__max_abs_return
# outliers__winsorize_threshold
# outliers__remove_threshold
# drop_correlated__threshold
# drop_correlated__absolute
# select_k__k
# select_k__highest
```
