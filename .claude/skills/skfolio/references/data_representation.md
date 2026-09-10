# Data Representation & Missing Data (1.0)

## Wide format is the standard

skfolio represents asset data in **wide format** — a `(observations × assets)` matrix with a `DatetimeIndex`, missing values as `NaN`. Deliberate trade-off: more memory (cheap for typical universes) for computational efficiency, simpler code, and clear temporal/asset alignment. Long format (one row per `(date, asset)`) is non-native and must be pivoted first.

`X` for every estimator/pipeline is a pandas `DataFrame` of **linear returns** (tickers as columns). See `preprocessing.md` and the "one rule that matters" in `SKILL.md`.

## AssetPanel (NEW in 1.0)

`from skfolio.containers import AssetPanel` — a dedicated container for aligned **multi-field** cross-sectional data (returns, market cap, fundamentals, categorical fields like industry). Supersedes ad-hoc 3D arrays / xarray / MultiIndex DataFrames. It is the input to `CharacteristicsFactorModel` (see `factor_models.md`).

```python
AssetPanel(fields, observations, asset_names, active_mask=None, estimation_mask=None)
```

- `add_2d_field(name, values)`, `add_3d_field(...)`, `add_categorical_field(name, values, levels=[...])`
- `save(path)`, `AssetPanel.load(path, mmap_mode=None, fields=None)`
- props: `n_observations`, `n_assets`, `n_fields`, `shape`, `ndim`
- other exports: `AssetPanelView`, `Field2D`, `Field3D`, `FieldCategorical`, `InactivePolicy`, `concat`

## Two ways to handle missing data / changing universes

### 1. Pre-selection & imputation (produce finite inputs)

Chain `SelectComplete` (keep full-history assets) / `SelectNonExpiring` in a `Pipeline` before the optimizer so estimators see no NaN. See `distance_clustering.md`.

### 2. Native NaN-aware estimators (1.0)

EW moment estimators accept NaN directly and use two masks:

- **`active_mask`** — boolean per asset per observation: universe membership. Distinguishes *in-universe but missing* (holiday → covariance frozen) from *out-of-universe* (pre-listing / post-delisting).
- **`estimation_mask`** — optional: restrict estimator-specific calculations.

A fitted prior yields a **full-universe** `ReturnDistribution` where non-investable assets are marked `NaN` in `mu` / `variance`. Optimizers then extract the investable subset, solve, and **expand weights back to full-universe shape**. This is the recommended path for online-learning workflows (`online_predict` / `OnlineGridSearch`), where the universe changes each step.

```python
from skfolio.moments import EWCovariance
cov = EWCovariance(half_life=40)
cov.partial_fit(batch)         # NaN-aware; masks handle universe membership
```

## Periodicity

Estimators output moments in the **periodicity of `X`** — daily returns → daily mu/covariance. Nothing is projected to the investment horizon inside optimization. Transaction costs and fees must be converted to match `X`'s periodicity manually. Reporting annualization is display-time via `Portfolio(annualization_factor=252.0)` (renamed from `annualized_factor` in 1.0). Multi-year horizon projection is opt-in via `EmpiricalPrior(is_log_normal=True, investment_horizon=...)`.
