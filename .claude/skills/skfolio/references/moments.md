# Moment Estimators (Mu, Variance, Covariance)

All follow the scikit-learn estimator API. After `fit(X)`, results live in `mu_`, `variance_`, or `covariance_`.

## Expected Returns

```python
from skfolio.moments import (
    EmpiricalMu, EWMu, ShrunkMu, ShrunkMuMethods, EquilibriumMu,
)
```

| Estimator | Description | Key params |
|---|---|---|
| `EmpiricalMu` | Historical mean | — |
| `EWMu` | Exponentially weighted mean — supports `partial_fit` (0.17.0+), NaN-aware | `half_life` |
| `ShrunkMu` | Shrinkage toward grand mean | `shrinkage_method` |
| `EquilibriumMu` | Market equilibrium (CAPM) | `risk_aversion` |

## Variance (v0.17.0+)

Marginal volatility only — use for idiosyncratic risk or orthogonalized series. **Not a drop-in replacement for covariance estimators inside a prior** — priors that need a full matrix require a covariance estimator.

```python
from skfolio.moments import EmpiricalVariance, EWVariance, RegimeAdjustedEWVariance
```

| Estimator | Description | Key params |
|---|---|---|
| `EmpiricalVariance` | Sample variance per asset | — |
| `EWVariance` | EW variance, `partial_fit` + NaN-aware | `half_life`, `active_mask` |
| `RegimeAdjustedEWVariance` | STVU regime-adjusted EW | `half_life`, `regime_half_life`, `regime_multiplier_clip` |

## Covariance

```python
from skfolio.moments import (
    EmpiricalCovariance, EWCovariance, LedoitWolf, OAS,
    ShrunkCovariance, DenoiseCovariance, DetoneCovariance,
    GerberCovariance, GraphicalLassoCV, ImpliedCovariance,
    RegimeAdjustedEWCovariance,     # v0.17.0+
)
```

| Estimator | Description | Key params |
|---|---|---|
| `EmpiricalCovariance` | Sample covariance | — |
| `EWCovariance` | EW covariance — `partial_fit`, NaN-aware (0.17.0+) | `half_life`, `active_mask` |
| `LedoitWolf` | Shrinkage toward structured target | — |
| `OAS` | Oracle Approximating Shrinkage | — |
| `ShrunkCovariance` | Parametric shrinkage | `shrinkage` |
| `DenoiseCovariance` | Random Matrix Theory denoising | `n_components` |
| `DetoneCovariance` | Removes the market factor | `n_components` |
| `GerberCovariance` | Gerber statistic-based | `threshold` |
| `GraphicalLassoCV` | Sparse precision matrix | `alphas` |
| `ImpliedCovariance` | From options implied vol; needs metadata routing | `implied_vol` (metadata) |
| `RegimeAdjustedEWCovariance` | STVU regime-adjusted EW (0.17.0+) | see below |

## RegimeAdjustedEWCovariance (v0.17.0+)

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
    active_mask=None,
    estimation_mask=None,
)
cov.fit(X)
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
