# Online Learning & Covariance Forecast Evaluation (v0.18.0+)

Online learning keeps a **single stateful estimator** that updates incrementally via `partial_fit`, instead of refitting from scratch at every CV split. This speeds up walk-forward simulations dramatically and more closely matches live-trading semantics.

## Imports

```python
from skfolio.model_selection import (
    online_predict, online_score,
    OnlineGridSearch, OnlineRandomizedSearch,
    covariance_forecast_evaluation,
    online_covariance_forecast_evaluation,
    CovarianceForecastEvaluation,
    CovarianceForecastComparison,
)
```

## Requirements

The estimator **must** implement `partial_fit`. `online_predict` / `online_score` / `OnlineGridSearch` **do not** accept `Pipeline` objects — wrap an incremental estimator directly.

Currently `partial_fit`-capable:
- Moment estimators: `EWMu`, `EWCovariance`, `EWVariance`, `RegimeAdjustedEWCovariance`, `RegimeAdjustedEWVariance`
- Prior: `EmpiricalPrior` (when wrapping incremental moments)
- Optimizer: `MeanRisk` (when wrapping an incremental prior)

## online_predict

Forward-walking prediction: clone estimator → warm up → for each test window, predict then update.

```python
from skfolio.model_selection import online_predict
from skfolio.moments import EWMu, EWCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.optimization import MeanRisk

model = MeanRisk(
    prior_estimator=EmpiricalPrior(
        mu_estimator=EWMu(half_life=40),
        covariance_estimator=EWCovariance(half_life=40),
    ),
)

pred = online_predict(
    model, X,
    warmup_size=252,     # first observations used for initial partial_fit
    test_size=1,         # rebalance step
    freq=None,           # or "W", "M" for frequency-based splits
    purged_size=0,
)
# Returns MultiPeriodPortfolio
```

## online_score

Returns a scalar (or dict) computed on the full concatenated out-of-sample path — more stable than averaging per-fold scores at short rebalance horizons.

```python
from skfolio.metrics import make_scorer
from skfolio import RatioMeasure

score = online_score(
    model, X,
    scoring=make_scorer(RatioMeasure.SORTINO_RATIO),
    warmup_size=252, test_size=1,
)
```

## OnlineGridSearch / OnlineRandomizedSearch

Evaluates each candidate through a complete walk-forward run instead of independent-fold refits.

```python
from skfolio.model_selection import OnlineGridSearch

search = OnlineGridSearch(
    estimator=model,
    param_grid={
        "prior_estimator__mu_estimator__half_life": [20, 40, 80],
        "prior_estimator__covariance_estimator__half_life": [20, 40, 80],
    },
    warmup_size=252,
    test_size=1,
    scoring=make_scorer(RatioMeasure.SORTINO_RATIO),
    n_jobs=-1,
)
search.fit(X)
print(search.best_params_, search.best_score_)
```

`OnlineRandomizedSearch` has the same surface but samples `param_distributions` for `n_iter` candidates.

## Covariance Forecast Evaluation

Diagnose a covariance estimator's out-of-sample quality **independently of any optimizer**. Use this to rank candidate covariance estimators *before* embedding them in a prior.

```python
from skfolio.moments import EWCovariance, RegimeAdjustedEWCovariance

# Walk-forward (refit every split)
ew_eval = covariance_forecast_evaluation(
    EWCovariance(half_life=40), X, warmup_size=252,
)

# Online (partial_fit-based, much faster)
reg_eval = online_covariance_forecast_evaluation(
    RegimeAdjustedEWCovariance(half_life=40), X, warmup_size=252,
)

reg_eval.summary()
reg_eval.plot_calibration()   # Mahalanobis calibration over time
reg_eval.plot_exceedance()    # chi-squared exceedance rate
reg_eval.plot_qlike_loss()    # forecast-vs-realized variance loss
```

### Diagnostics

| Metric | Target | Meaning |
|---|---|---|
| `mahalanobis_calibration_ratio` | 1.0 | Full-structure calibration; >1 ⇒ risk underestimated |
| `diagonal_calibration_ratio` | 1.0 | Per-asset variance calibration |
| `portfolio_standardized_return` | mean 0, var 1 | Portfolio-direction calibration |
| `portfolio_variance_qlike_loss` | lower is better | Portfolio variance forecast quality |

### Side-by-side comparison

```python
comp = CovarianceForecastComparison({"EW": ew_eval, "RegimeEW": reg_eval})
comp.summary()
```
