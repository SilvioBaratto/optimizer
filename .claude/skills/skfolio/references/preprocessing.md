# Preprocessing — `prices_to_returns` & Cross-Sectional Transformers

`skfolio.preprocessing` holds two unrelated tools:

1. `prices_to_returns(prices)` — convert price panel to linear (or log) returns. Always run this first. Linear returns are required by every optimizer (see SKILL.md "the one rule that matters").
2. **Cross-sectional transformers (v0.20.0+)** — feature preprocessors that operate **across assets within each observation** (axis=1), not across time. Built for factor pipelines and `CSLinearRegression` inputs.

## Imports

```python
from skfolio.preprocessing import (
    prices_to_returns,
    BaseCSTransformer,
    CSStandardScaler,
    CSWinsorizer,
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSTanhShrinker,
)
```

## Cross-sectional contract

All five transformers share the same contract:

- Input `X`: `(n_observations, n_assets)` array or DataFrame.
- Output: same shape; **NaN locations preserved**.
- Each row is processed **independently** — no cross-time leakage.
- Optional `cs_weights: (n_observations, n_assets)` — defines the estimation universe per row. Zero weight → asset excluded from that row's stats.
- Optional `cs_groups: (n_observations, n_assets)` integer array — when provided, processing happens within each group then rescaled globally. Groups smaller than `min_group_size` fall back to the global cross-section.
- Stateless `fit` (no learned parameters); use `fit_transform` or `transform` interchangeably.
- Plug into `Pipeline` and `GridSearchCV` like any sklearn transformer.

`cs_weights` / `cs_groups` are passed via `transform(X, cs_weights=..., cs_groups=...)` (or via metadata routing inside a `Pipeline`).

## CSStandardScaler

```python
CSStandardScaler(*, min_group_size=8, atol=1e-12)
```

Z-score across assets per row → weighted mean 0, unit equal-weighted std. NaN preserved. Use as default centering layer before regression or distance computations.

```python
import numpy as np
from skfolio.preprocessing import CSStandardScaler

X = np.array([[1.0, np.nan, 3.0, 4.0],
              [4.0, 3.0, 2.0, 1.0],
              [10.0, 20.0, np.nan, 40.0]])
Z = CSStandardScaler().fit_transform(X)   # (3, 4), NaN kept
```

## CSWinsorizer

```python
CSWinsorizer(*, low=0.01, high=0.99)
```

Hard clip per row to `[low_q, high_q]` percentiles. `0 ≤ low < high ≤ 1`. Use to cap extreme cross-sectional outliers before regression. NaN preserved.

```python
from skfolio.preprocessing import CSWinsorizer
Xw = CSWinsorizer(low=0.05, high=0.95).fit_transform(X)
```

## CSGaussianRankScaler

```python
CSGaussianRankScaler(*, min_group_size=8, scale=True, atol=1e-12)
```

Per-row percentile rank → inverse standard normal CDF → recentered to weighted mean 0; if `scale=True`, rescaled to unit std. Heavy-tailed-robust alternative to `CSStandardScaler`. Output approximately Gaussian per row.

## CSPercentileRankScaler

```python
CSPercentileRankScaler(*, min_group_size=8)
```

Per-row percentile rank in `(0, 1)`. NaN preserved. Use when you need a strictly bounded, monotone, scale-free signal — common for factor zoo features fed into tree models or as inputs to `CSLinearRegression`.

## CSTanhShrinker

```python
CSTanhShrinker(*, knee=3.0, atol=1e-12)
```

Smooth outlier shrinker: `x' = median + h * tanh((x - median) / h)` with `h = knee * robust_scale`. Soft alternative to `CSWinsorizer` — preserves ordering and original units, no hard cliff. Smaller `knee` → more compression.

## Typical pipeline

Stack a CS transformer in front of `CSLinearRegression` for a Fama-MacBeth-style factor extractor:

```python
from sklearn.pipeline import Pipeline
from skfolio.preprocessing import CSWinsorizer, CSStandardScaler
from skfolio.linear_model import CSLinearRegression

pipe = Pipeline([
    ("winsor", CSWinsorizer(low=0.01, high=0.99)),
    ("zscore", CSStandardScaler()),
    ("reg",    CSLinearRegression(fit_intercept=True)),
])
# X: (T, N, K)  ← regression input shape; transformers run on each (T, N) feature slice
```

For 3D feature panels `(T, N, K)`, apply transformers per feature slice (`X[..., k]`) before stacking — they are 2D-only.

## Gotchas

1. **Axis = cross-section, not time.** Standard sklearn `StandardScaler` standardizes per column (per asset over time) — `CSStandardScaler` standardizes per row (across assets per period). Do not confuse them; they solve opposite problems.
2. **NaN are passive.** All five transformers ignore NaN when computing stats and pass them through unchanged. Do **not** impute upstream just to satisfy them.
3. **`min_group_size=8` default** is conservative for narrow universes. Lower it for portfolios under 30 assets or you'll always hit the global fallback.
4. **`scale` flag (Gaussian only).** Set `CSGaussianRankScaler(scale=False)` if downstream model already standardizes — double-scaling distorts magnitudes.
5. **2D input only.** For 3D `(T, N, K)` panels, loop over `K` or wrap with a `ColumnTransformer`-equivalent.
6. **Stateless** — no leakage risk inside `WalkForward` / `CombinatorialPurgedCV`, since `fit` learns nothing.
