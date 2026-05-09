# Online Learning

## When to use

Online (incremental) workflows update an estimator one batch at a
time via `partial_fit` instead of re-fitting from scratch on each
walk-forward fold. They match live-trading semantics — every new
observation extends the training set without recomputing — and are
materially faster than full-refit cross-validation when every
candidate exposes `partial_fit`.

Use online wrappers when:

- You are evaluating EW estimators (`EWMu`, `EWCovariance`,
  `EWVariance`, `RegimeAdjustedEW*`) on long histories.
- You want to back-test the exact procedure that will run in
  production (one new bar at a time).
- The full-refit walk-forward is too slow.

The skfolio reference is in
`~/.claude/skills/skfolio/references/online_learning.md`.

## Pipeline is not supported

`online_predict`, `online_score`, `OnlineGridSearch`, and
`OnlineRandomizedSearch` route `partial_fit` through a single
estimator. `sklearn.pipeline.Pipeline` composes `partial_fit`
differently and is **explicitly rejected** at the wrapper boundary —
all four entry points raise `ConfigurationError` if the estimator is
a `Pipeline`. Apply pre-selection upstream (transform `X` before
passing it in).

## Thread safety

Online estimators accumulate mutable state across `partial_fit` calls
and `OnlineGridSearch` mutates the wrapped estimator in place. **Use
one wrapper instance per thread.** Do not share an online optimizer
across daemon threads in scheduled jobs.

## Composition pattern

```python
from skfolio.datasets import load_sp500_dataset
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

from optimizer.online import OnlinePredictConfig, run_online_predict


prices = load_sp500_dataset()
returns = prices_to_returns(prices)

estimator = MeanRisk(
    prior_estimator=EmpiricalPrior(
        mu_estimator=EWMu(half_life=40.0),
        covariance_estimator=EWCovariance(half_life=40.0),
    )
)
portfolio = run_online_predict(
    estimator,
    returns,
    None,
    config=OnlinePredictConfig(warmup_size=252),
)
print(portfolio.annualized_sharpe_ratio)
```

A complete example, including a comparison against full-refit
walk-forward, is in `examples/online_learning.py`.

## Hyperparameter search

`OnlineGridSearchConfig` and `OnlineRandomizedSearchConfig` embed the
existing `GridSearchConfig` / `RandomizedSearchConfig` plus an
`OnlinePredictConfig`. Build them via `build_online_grid_search` /
`build_online_randomized_search`. Same Pipeline-rejection contract as
`run_online_predict`.

## See also

- [Validation: covariance forecast evaluation](../guide/validation.md)
  — the online forecast-evaluation path uses the same `partial_fit`
  contract.
- skfolio reference:
  `~/.claude/skills/skfolio/references/online_learning.md`.
