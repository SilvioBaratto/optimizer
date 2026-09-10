# Online Learning & Covariance Forecast Evaluation

Online learning keeps a **single stateful estimator** that updates incrementally via `partial_fit`, instead of refitting from scratch at every CV split. Speeds up walk-forward simulations dramatically and more closely matches live-trading semantics.

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

The estimator **must** implement `partial_fit`. `online_predict` / `online_score` / `OnlineGridSearch` **do not** accept `Pipeline` objects — wrap an incremental estimator directly. Instances are **not thread-safe** (mutable accumulated state) — one wrapper per thread.

Currently `partial_fit`-capable:
- Moment estimators: `EWMu`, `EWCovariance`, `EWVariance`, `RegimeAdjustedEWCovariance`, `RegimeAdjustedEWVariance`
- Prior: `EmpiricalPrior` (when wrapping incremental moments); `CharacteristicsFactorModel`
- Optimizer: `MeanRisk` (when wrapping an incremental prior)

## online_predict

Forward-walking prediction: clone estimator → warm up → for each test window, predict then update. Restricted to portfolio-optimization estimators; returns a `MultiPeriodPortfolio`.

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
    fallback="previous_weights",   # carry last allocation forward on a failed rebalance
    raise_on_failure=False,        # → FailedPortfolio instead of raising (see portfolio.md)
)

pred = online_predict(
    model, X,
    warmup_size=252,     # initial observations for warmup partial_fit
    test_size=1,         # rebalance step
    purged_size=0,       # optional gap between train/test windows
)
```

## online_score

Returns a scalar (or dict) computed on the full concatenated out-of-sample path — more stable than averaging per-fold scores at short rebalance horizons. Accepts optimizers *and* non-predictor estimators.

```python
from skfolio.metrics import make_scorer
from skfolio import RatioMeasure

score = online_score(
    model, X,
    scoring=RatioMeasure.SORTINO_RATIO,   # measure directly; make_scorer is NOT accepted by online APIs
    warmup_size=252, test_size=1,
)
```

- For **optimizers**, pass a `BaseMeasure`/`RatioMeasure` enum directly — the online evaluators **reject** `make_scorer(...)` (raises `TypeError`).
- For **non-predictor** estimators, use `make_scorer(..., response_method=None)`.
- `portfolio_weights=None` defaults to an inverse-volatility portfolio direction (rather than erroring).

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
    scoring=RatioMeasure.SORTINO_RATIO,   # measure directly; make_scorer not supported for online eval
    refit=None,      # metric name for multi-metric searches
    n_jobs=-1,
)
search.fit(X)
print(search.best_params_, search.best_score_)   # best_estimator_ ready without extra refit
```

`OnlineRandomizedSearch` has the same surface but samples `param_distributions` for `n_iter` candidates.

## Covariance Forecast Evaluation

Diagnose a covariance estimator's out-of-sample quality **independently of any optimizer**. Rank candidates *before* embedding one in a prior.

```python
from skfolio.moments import EWCovariance, RegimeAdjustedEWCovariance

# Walk-forward (refit every split)
ew_eval = covariance_forecast_evaluation(
    EWCovariance(half_life=40), X, train_size=252,
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
comp = CovarianceForecastComparison([ew_eval, reg_eval], names=["EW", "RegimeEW"])
comp.summary()
```
