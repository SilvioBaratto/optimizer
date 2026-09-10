# Moment Estimators (Mu, Variance, Covariance)

All follow the scikit-learn estimator API. After `fit(X)`, results live in `mu_`, `variance_`, or `covariance_`. Moments are in the **periodicity of `X`** (daily returns → daily moments) — nothing is annualized inside the estimator.

## ⚠️ 1.0 breaking change — EW estimators use `half_life`, not `alpha`

skfolio 1.0 **removed** the `alpha` constructor arg from every exponentially-weighted estimator. Passing `alpha=` now raises `TypeError` (hard break, not a warning).

```python
# 0.20.x (BROKEN in 1.0)
EWMu(alpha=0.2)
# 1.0
EWMu(half_life=3.11)
```

- `half_life` = number of observations over which the weight decays to 50%.
- Decay factor `lambda = 2 ** (-1 / half_life)`.
- Convert an old alpha: `half_life = -1 / log2(1 - alpha)` (e.g. `alpha=0.2` → `half_life ≈ 3.11`).
- Helper: `from skfolio.utils.tools import half_life_to_decay_factor`.

## Expected Returns

```python
from skfolio.moments import (
    EmpiricalMu, EWMu, ShrunkMu, ShrunkMuMethods, EquilibriumMu,
)
```

| Estimator | Description | Key params |
|---|---|---|
| `EmpiricalMu` | Historical mean | — |
| `EWMu` | Exponentially weighted mean — `partial_fit` + NaN-aware `active_mask` | `half_life=40`, `min_observations`, `window_size` |
| `ShrunkMu` | Shrinkage toward grand mean | `method` (`ShrunkMuMethods`: `JAMES_STEIN`, `BAYES_STEIN`, `BODNAR_OKHRIN`) |
| `EquilibriumMu` | Market equilibrium (CAPM reverse-optimization) | `risk_aversion` |

Signature: `EWMu(half_life=40, min_observations=None, window_size=None)`. Fitted output in `mu_` (1-D, periodicity of `X`, not annualized).

## Variance

Marginal volatility only — use for idiosyncratic risk or orthogonalized series. **Not a drop-in replacement for covariance estimators inside a prior** — priors that need a full matrix require a covariance estimator.

```python
from skfolio.moments import EmpiricalVariance, EWVariance, RegimeAdjustedEWVariance
```

| Estimator | Description | Key params |
|---|---|---|
| `EmpiricalVariance` | Sample variance per asset | — |
| `EWVariance` | EW variance, `partial_fit` + NaN-aware `active_mask` | `half_life` |
| `RegimeAdjustedEWVariance` | STVU regime-adjusted EW | `half_life`, `regime_half_life`, `regime_multiplier_clip` |

Fitted output in `variance_` (1-D), NOT `covariance_` (2-D).

## Covariance

```python
from skfolio.moments import (
    EmpiricalCovariance, EWCovariance, LedoitWolf, OAS,
    ShrunkCovariance, DenoiseCovariance, DetoneCovariance,
    GerberCovariance, GraphicalLassoCV, ImpliedCovariance,
    RegimeAdjustedEWCovariance,
)
```

| Estimator | Description | Key params |
|---|---|---|
| `EmpiricalCovariance` | Sample covariance | — |
| `EWCovariance` | EW covariance — `partial_fit`, NaN-aware `active_mask` | `half_life=40`, `assume_centered`, `nearest`, `higham` |
| `LedoitWolf` | Shrinkage toward structured target | — |
| `OAS` | Oracle Approximating Shrinkage | — |
| `ShrunkCovariance` | Parametric shrinkage | `shrinkage` |
| `DenoiseCovariance` | Random Matrix Theory denoising | — |
| `DetoneCovariance` | Removes the market factor | — |
| `GerberCovariance` | Gerber statistic-based | `threshold` |
| `GraphicalLassoCV` | Sparse precision matrix | `alphas` |
| `ImpliedCovariance` | From options implied vol; needs metadata routing | `implied_vol` (metadata) |
| `RegimeAdjustedEWCovariance` | STVU regime-adjusted EW | see below |

Signature: `EWCovariance(half_life=40, assume_centered=True, min_observations=None, window_size=None, nearest=True, higham=False, higham_max_iteration=100)`.

> `ImpliedCovariance` — the `annualized_factor` param was renamed `annualization_factor` in 1.0 (old name deprecated, `FutureWarning`, removed in 2.0).

## RegimeAdjustedEWCovariance

Rescales EW covariance with a scalar multiplier when realized risk diverges from predicted risk (Short-Term Volatility Update). Supports:
- Separate half-lives for variance vs. correlation
- Newey-West (HAC) correction for autocorrelated returns
- Late-listing bias correction from EWMA initialization
- NaN handling distinguishing holidays (frozen cov) from inactive periods

```python
from skfolio.moments import (
    RegimeAdjustedEWCovariance,
    RegimeAdjustmentTarget,
    RegimeAdjustmentMethod,
)

cov = RegimeAdjustedEWCovariance(
    half_life=23,                                  # variance decay
    corr_half_life=50,                             # slower correlation decay
    regime_half_life=None,                         # auto = half_life / 2
    regime_target=RegimeAdjustmentTarget.PORTFOLIO,
    regime_method=RegimeAdjustmentMethod.FIRST_MOMENT,
    regime_multiplier_clip=(0.7, 1.6),             # widen for fast regimes
    hac_lags=5,
    min_observations=None,
)
cov.fit(X)   # active_mask=... / estimation_mask=... are keyword-only fit() args, NOT constructor args
print(cov.regime_multiplier_)
```

### `RegimeAdjustmentTarget`

- `PORTFOLIO` — variance along one or more weight vectors
- `DIAGONAL` — individual asset vols, ignores correlations
- `MAHALANOBIS` — full covariance structure

### `RegimeAdjustmentMethod`

- `LOG` — outlier-robust logarithmic compression
- `FIRST_MOMENT` — calibrates mean of standardized risk statistic
- `RMS` — chi-squared calibration (sensitive to extremes)

### Online use — `partial_fit`

```python
for batch in batches:
    cov.partial_fit(batch)
    # cov.covariance_ and cov.regime_multiplier_ update in place
```

This is what makes `RegimeAdjustedEWCovariance` drop-in for `online_predict` / `OnlineGridSearch` (see `online_learning.md`).

## NaN-aware / changing universes (1.0)

EW estimators (`EWMu`, `EWCovariance`, `EWVariance`, and the regime-adjusted variants) accept NaNs directly and use `active_mask` (universe membership per asset per observation) and optional `estimation_mask` (restrict estimator-specific calculations) to distinguish *in-universe-but-missing* (e.g. holidays → frozen) from *out-of-universe* (pre-listing / post-delisting). Fitted moments expose non-investable assets as NaN; optimizers then solve over the investable subset and expand weights back to full-universe shape. See `data_representation.md`.
