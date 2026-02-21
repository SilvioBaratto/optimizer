# Preprocessing

The preprocessing module provides sklearn-compatible transformers that clean, validate,
and impute asset return data before it enters the optimization pipeline. Every transformer
follows the `BaseEstimator + TransformerMixin` API and composes naturally in
`sklearn.pipeline.Pipeline`.

---

## Overview

### The Problem

Raw return data from market data vendors is rarely clean. Common issues include:

- **Infinite values** from division-by-zero in return calculations
- **Extreme outliers** from stock splits, corporate actions, or data feed errors
- **Missing values** from trading halts, delistings, holidays, or late-starting assets
- **Survivorship bias** when delisted securities silently disappear from datasets

Left untreated, these issues distort moment estimates, break optimizers, and produce
unreliable portfolios.

### Design Philosophy

The preprocessing module addresses each issue with a dedicated transformer:

| Step | Transformer | Purpose |
|------|-------------|---------|
| 1 | `DataValidator` | Replace infinities and physically impossible returns with `NaN` |
| 2 | `OutlierTreater` | Classify observations into normal / outlier / error via z-scores |
| 3 | `SectorImputer` or `RegressionImputer` | Fill remaining `NaN` values |

Plus a standalone utility function:

| Function | Purpose |
|----------|---------|
| `apply_delisting_returns` | Inject terminal delisting returns to prevent survivorship bias |

All transformers accept and return `pd.DataFrame` objects (dates as rows, tickers as
columns). They are stateless or store only lightweight statistics (`mu_`, `sigma_`,
correlation rankings) during `fit()`.

---

## DataValidator

**Module**: `optimizer.preprocessing._validation`

The first line of defense. `DataValidator` replaces infinities and physically impossible
returns with `NaN`, ensuring downstream transformers receive well-formed numeric data.

### Algorithm

1. Replace all `+inf` and `-inf` values with `NaN`.
2. Replace any return where `|r| > max_abs_return` with `NaN`.

The transformer is **stateless**: `fit()` stores only metadata (`n_features_in_`,
`feature_names_in_`) but no learned statistics. This means train/test behavior is
identical.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_abs_return` | `float` | `10.0` | Absolute return threshold. Values with `|r| > max_abs_return` become `NaN`. The default of 10.0 (1,000%) is deliberately generous -- it catches data errors while preserving legitimate large moves. |

### Example

```python
import pandas as pd
import numpy as np
from optimizer.preprocessing import DataValidator

returns = pd.DataFrame(
    {
        "AAPL": [0.01, -0.02, np.inf, 0.005],
        "MSFT": [0.02, 15.0, -0.01, 0.003],  # 15.0 = 1500%, likely an error
        "GOOG": [0.015, -np.inf, 0.008, -0.003],
    },
    index=pd.date_range("2024-01-01", periods=4, freq="B"),
)

validator = DataValidator(max_abs_return=10.0)
clean = validator.fit_transform(returns)

print(clean)
#              AAPL   MSFT   GOOG
# 2024-01-01  0.010  0.020  0.015
# 2024-01-02 -0.020    NaN    NaN   <-- inf and 15.0 replaced
# 2024-01-03    NaN -0.010  0.008
# 2024-01-04  0.005  0.003 -0.003
```

---

## OutlierTreater

**Module**: `optimizer.preprocessing._outliers`

Applies a three-group z-score methodology to classify each observation and treat it
accordingly. This is a **stateful** transformer: `fit()` computes per-column mean and
standard deviation from training data.

### Algorithm

During `fit()`:

- Compute per-column mean (`mu_`) and standard deviation (`sigma_`) from the training
  data.

During `transform()`:

- Compute z-scores for each observation: `z = (x - mu_) / sigma_`.
- Classify into three groups based on `|z|`:

```
                      winsorize_threshold    remove_threshold
|z| ──────────────────────┼───────────────────────┼──────────────►
     Group 3: Keep        │  Group 2: Winsorize   │  Group 1: NaN
     (normal data)        │  (clip to bounds)     │  (data errors)
```

| Group | Condition | Action |
|-------|-----------|--------|
| 1 -- Data errors | `|z| >= remove_threshold` | Replaced with `NaN` |
| 2 -- Outliers | `winsorize_threshold <= |z| < remove_threshold` | Clipped to `mu +/- winsorize_threshold * sigma` |
| 3 -- Normal | `|z| < winsorize_threshold` | Kept as-is |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `winsorize_threshold` | `float` | `3.0` | Z-score boundary between normal observations (Group 3) and outliers (Group 2). |
| `remove_threshold` | `float` | `10.0` | Z-score boundary between outliers (Group 2) and data errors (Group 1). Values at exactly this threshold are treated as errors. |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu_` | `pd.Series` | Per-column mean from training data |
| `sigma_` | `pd.Series` | Per-column standard deviation from training data |

### Example

```python
import pandas as pd
import numpy as np
from optimizer.preprocessing import OutlierTreater

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=200, freq="B")
returns = pd.DataFrame(
    np.random.normal(0.0005, 0.02, size=(200, 3)),
    columns=["AAPL", "MSFT", "GOOG"],
    index=dates,
)

# Inject a data error and an outlier
returns.iloc[50, 0] = 0.50   # ~25 sigma, data error
returns.iloc[100, 1] = 0.10  # ~5 sigma, moderate outlier

treater = OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)
treater.fit(returns)

print(f"AAPL mean: {treater.mu_['AAPL']:.6f}, std: {treater.sigma_['AAPL']:.6f}")

treated = treater.transform(returns)

# Data error at row 50 is now NaN
print(f"Row 50 AAPL (original): {returns.iloc[50, 0]:.4f}")
print(f"Row 50 AAPL (treated):  {treated.iloc[50, 0]}")  # NaN

# Outlier at row 100 is winsorized
print(f"Row 100 MSFT (original): {returns.iloc[100, 1]:.4f}")
print(f"Row 100 MSFT (treated):  {treated.iloc[100, 1]:.4f}")  # clipped
```

---

## SectorImputer

**Module**: `optimizer.preprocessing._imputation`

Fills `NaN` values using leave-one-out sector cross-sectional averages at each timestep.
This approach preserves the cross-sectional return structure better than
forward-filling or global mean imputation.

### Algorithm

For each `NaN` at position `(t, asset_i)`:

1. Identify the sector of `asset_i` using `sector_mapping`.
2. Compute the mean return at timestep `t` across all **other** assets in the same
   sector (leave-one-out to avoid self-influence).
3. If the entire sector is `NaN` at timestep `t`, fall back to the global
   cross-sectional mean (mean of all non-NaN assets at that timestep).

When `sector_mapping` is `None`, all assets are treated as belonging to a single sector,
which reduces the imputer to a global cross-sectional mean.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sector_mapping` | `dict[str, str]` or `None` | `None` | Maps ticker to sector label. Columns absent from the mapping are assigned to `"__unmapped__"`. When `None`, all assets share one group. |
| `fallback_strategy` | `str` | `"global_mean"` | Strategy when the entire sector is `NaN`. Only `"global_mean"` is supported. |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `sector_groups_` | `dict[str, list[str]]` | Mapping of sector label to list of column names in that sector |

### Example

```python
import pandas as pd
import numpy as np
from optimizer.preprocessing import SectorImputer

returns = pd.DataFrame(
    {
        "AAPL": [0.01, np.nan, 0.005, -0.01],
        "MSFT": [0.02, 0.015, np.nan, 0.008],
        "GOOG": [0.015, 0.012, 0.007, np.nan],
        "JPM":  [0.005, np.nan, -0.003, 0.01],
        "BAC":  [0.008, 0.006, -0.005, 0.012],
    },
    index=pd.date_range("2024-01-01", periods=4, freq="B"),
)

sector_mapping = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOG": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
}

imputer = SectorImputer(sector_mapping=sector_mapping)
filled = imputer.fit_transform(returns)

# AAPL NaN at row 1 is filled with mean of MSFT and GOOG at that timestep
# (0.015 + 0.012) / 2 = 0.0135
print(f"AAPL row 1 (imputed): {filled.loc['2024-01-02', 'AAPL']:.4f}")

# JPM NaN at row 1 is filled with BAC's value (only other Financials asset)
print(f"JPM row 1 (imputed):  {filled.loc['2024-01-02', 'JPM']:.4f}")
```

---

## RegressionImputer

**Module**: `optimizer.preprocessing._regression_imputer`

The most sophisticated imputer. Fills `NaN` values using OLS regression from each
asset's most correlated neighbors. This approach preserves the covariance structure
of the imputed values better than mean-based methods.

### Algorithm

**During `fit()`:**

1. Compute pairwise absolute correlations across all assets (using pairwise complete
   observations).
2. For each asset, select the `n_neighbors` most correlated other assets.
3. For each asset, fit an OLS regression on complete rows:
   `r_{i,t} = alpha + sum_j(beta_j * r_{j,t}) + epsilon`
4. Fit an internal `SectorImputer` on the training data for fallback use.

**During `transform()`:**

1. Pre-compute fallback values using the internal `SectorImputer`.
2. For each asset with `NaN` values:
    - If OLS coefficients are available **and** all neighbors have data at that
      timestep: predict using `r_hat = alpha + beta @ r_neighbors`.
    - Otherwise: use the `SectorImputer` fallback value.

**Cold-start handling**: if an asset has fewer than `min_train_periods` complete
observations across itself and its neighbors during `fit()`, no regression is fitted
and all imputation falls back to the `SectorImputer`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | `int` | `5` | Number of most-correlated assets used as regression predictors. |
| `min_train_periods` | `int` | `60` | Minimum complete-row count required to fit OLS. Assets below this threshold fall back to sector mean. |
| `fallback` | `str` | `"sector_mean"` | Fallback imputation strategy. Only `"sector_mean"` is supported. |
| `sector_mapping` | `dict[str, str]` or `None` | `None` | Passed to the internal `SectorImputer` for fallback imputation. |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `neighbors_` | `dict[str, list[str]]` | Top-K neighbor tickers per asset, ranked by absolute correlation |
| `coefs_` | `dict[str, np.ndarray or None]` | OLS coefficients per asset. Shape `(K+1,)` where index 0 is the intercept. `None` if cold-start. |

### Example

```python
import pandas as pd
import numpy as np
from optimizer.preprocessing import RegressionImputer

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=252, freq="B")

# Generate correlated returns
market = np.random.normal(0.0004, 0.01, size=252)
returns = pd.DataFrame(
    {
        "AAPL": market + np.random.normal(0, 0.005, 252),
        "MSFT": market + np.random.normal(0, 0.006, 252),
        "GOOG": market + np.random.normal(0, 0.007, 252),
        "AMZN": market + np.random.normal(0, 0.008, 252),
        "META": market + np.random.normal(0, 0.006, 252),
        "NVDA": market + np.random.normal(0, 0.009, 252),
    },
    index=dates,
)

# Inject some NaN values
returns.iloc[100:103, 0] = np.nan  # AAPL missing for 3 days
returns.iloc[200, 3] = np.nan      # AMZN missing for 1 day

sector_mapping = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer", "META": "Technology", "NVDA": "Technology",
}

imputer = RegressionImputer(
    n_neighbors=3,
    min_train_periods=60,
    sector_mapping=sector_mapping,
)
filled = imputer.fit_transform(returns)

# Check that NaN values are filled
print(f"NaN count before: {returns.isna().sum().sum()}")
print(f"NaN count after:  {filled.isna().sum().sum()}")

# Inspect which neighbors were selected for AAPL
print(f"AAPL neighbors: {imputer.neighbors_['AAPL']}")

# Check if regression was fitted (not cold-start)
print(f"AAPL has OLS coefs: {imputer.coefs_['AAPL'] is not None}")
```

---

## apply_delisting_returns

**Module**: `optimizer.preprocessing._delisting`

A standalone utility function (not a transformer) that injects delisting returns into
the return matrix. This prevents survivorship bias by ensuring that the terminal return
experienced by investors when a stock was delisted is reflected in the data.

### Algorithm

For each ticker in `delisting_returns`:

1. Find the last valid (non-NaN) index in that ticker's column.
2. Replace the return at that position with the provided delisting return value.

If a ticker's column is entirely `NaN`, it is skipped. If a ticker in the mapping
is not found in the DataFrame columns, a `DataError` is raised.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns` | `pd.DataFrame` | Dates-by-tickers return matrix. |
| `delisting_returns` | `dict[str, float]` | Mapping of ticker to its delisting return value. |

### Returns

A copy of the input DataFrame with delisting returns applied.

### Example

```python
import pandas as pd
import numpy as np
from optimizer.preprocessing import apply_delisting_returns

returns = pd.DataFrame(
    {
        "AAPL": [0.01, -0.02, 0.005, 0.003, 0.008],
        "LEHMQ": [0.02, -0.05, -0.15, -0.30, np.nan],  # delisted, last trade day 4
        "MSFT": [0.015, 0.008, -0.003, 0.01, 0.005],
    },
    index=pd.date_range("2024-01-01", periods=5, freq="B"),
)

# Lehman Brothers delisted with ~100% loss
adjusted = apply_delisting_returns(
    returns,
    delisting_returns={"LEHMQ": -1.0},
)

# The last valid return for LEHMQ is replaced with -1.0
print(adjusted["LEHMQ"])
# 2024-01-01    0.02
# 2024-01-02   -0.05
# 2024-01-03   -0.15
# 2024-01-04   -1.00  <-- replaced
# 2024-01-05      NaN
```

---

## Composing Transformers in a Pipeline

All preprocessing transformers are designed to compose in an `sklearn.pipeline.Pipeline`.
The recommended order is: validate, treat outliers, then impute.

```python
from sklearn.pipeline import Pipeline
from optimizer.preprocessing import (
    DataValidator,
    OutlierTreater,
    RegressionImputer,
)

sector_mapping = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
}

preprocessing_pipeline = Pipeline([
    ("validate", DataValidator(max_abs_return=10.0)),
    ("outliers", OutlierTreater(winsorize_threshold=3.0, remove_threshold=10.0)),
    ("impute", RegressionImputer(
        n_neighbors=5,
        min_train_periods=60,
        sector_mapping=sector_mapping,
    )),
])

# Single call handles the entire cleaning workflow
clean_returns = preprocessing_pipeline.fit_transform(returns)
```

For simpler use cases where sector structure is not available, swap
`RegressionImputer` for `SectorImputer` with `sector_mapping=None` (global mean
imputation):

```python
simple_pipeline = Pipeline([
    ("validate", DataValidator()),
    ("outliers", OutlierTreater()),
    ("impute", SectorImputer()),  # global cross-sectional mean
])
```

---

## Gotchas and Tips

!!! warning "Order matters: validate before treating outliers"
    `DataValidator` must run **before** `OutlierTreater`. If infinities reach the
    outlier treater, they will corrupt the `mu_` and `sigma_` statistics computed
    during `fit()`, causing all subsequent z-scores to be meaningless.

!!! warning "OutlierTreater is stateful -- watch for train/test leakage"
    `OutlierTreater` computes `mu_` and `sigma_` from training data and applies them
    at transform time. If you call `fit_transform()` on your full dataset instead of
    fitting on the training fold only, you introduce look-ahead bias. Always
    `fit()` on training data and `transform()` on test data separately, or use the
    transformer inside a pipeline that is wrapped in cross-validation.

!!! info "Zero-variance columns are handled gracefully"
    If a column has zero standard deviation (constant series), `OutlierTreater` treats
    its z-score as 0 (normal) rather than raising an error. These columns will
    typically be removed downstream by `DropZeroVariance` in the pre-selection pipeline.

!!! info "RegressionImputer falls back per-row, not per-asset"
    Even when an asset has a fitted regression, individual rows where any neighbor
    is `NaN` fall back to `SectorImputer`. This means the imputation method can vary
    across timesteps for the same asset. The fallback is computed for the entire
    DataFrame upfront for efficiency.

!!! tip "Cold-start assets get sector mean imputation"
    Assets with fewer than `min_train_periods` (default 60) complete observations
    have no regression fitted. All their `NaN` values are filled by the internal
    `SectorImputer`. This commonly happens with recently listed stocks.

!!! tip "Apply delisting returns before preprocessing"
    Call `apply_delisting_returns()` on your raw return DataFrame **before** passing
    it into the preprocessing pipeline. The delisting return is a real economic event,
    not missing data -- it should flow through the pipeline as a valid observation.

!!! warning "All transformers require pandas DataFrames"
    Passing a NumPy array or other type raises `DataError`. This is by design:
    the transformers rely on column names for sector mapping, correlation lookups,
    and feature name tracking.

!!! info "SectorImputer uses leave-one-out within sectors"
    The imputed value for a missing asset excludes that asset's own value from the
    sector mean. This prevents the imputed value from being influenced by itself
    (which would be circular when the value is `NaN` anyway) and produces more
    conservative fills when sectors have few members.

---

## Quick Reference

| Component | Type | Stateful | Key Parameters | Output |
|-----------|------|----------|----------------|--------|
| `DataValidator` | Transformer | No | `max_abs_return=10.0` | `inf` and extreme values replaced with `NaN` |
| `OutlierTreater` | Transformer | Yes | `winsorize_threshold=3.0`, `remove_threshold=10.0` | Three-group z-score treatment |
| `SectorImputer` | Transformer | Yes | `sector_mapping`, `fallback_strategy="global_mean"` | `NaN` filled with leave-one-out sector mean |
| `RegressionImputer` | Transformer | Yes | `n_neighbors=5`, `min_train_periods=60`, `sector_mapping` | `NaN` filled via OLS regression with sector fallback |
| `apply_delisting_returns` | Function | N/A | `returns`, `delisting_returns` | Terminal returns replaced with delisting values |

**Imports:**

```python
from optimizer.preprocessing import (
    DataValidator,
    OutlierTreater,
    SectorImputer,
    RegressionImputer,
    apply_delisting_returns,
)
```

**Recommended pipeline order:**

```
apply_delisting_returns() → DataValidator → OutlierTreater → RegressionImputer (or SectorImputer)
```
