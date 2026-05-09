# Uncertainty Sets for Robust Mean-Risk

## When to use

Uncertainty sets bound the worst-case mu and covariance over a
confidence ball. Plug them into `RobustMeanRisk` when point estimates
of mu / covariance are noisy and you want the optimizer to hedge
against the estimation error itself, not just against return tails.
For ellipsoidal mu uncertainty, the optimizer solves the worst-case
problem inside a chi-squared ball whose radius scales with
`confidence_level`.

The full skfolio reference is in
`~/.claude/skills/skfolio/references/optimization.md` (sections on
`mu_uncertainty_set_estimator` and
`covariance_uncertainty_set_estimator`).

## Estimator surface

Two parallel Configs cover Mu and Covariance. Each has two variants:

| Type | EMPIRICAL | BOOTSTRAP |
|------|-----------|-----------|
| Mu | `EmpiricalMuUncertaintySet` | `BootstrapMuUncertaintySet` |
| Covariance | `EmpiricalCovarianceUncertaintySet` | `BootstrapCovarianceUncertaintySet` |

The bootstrap variants delegate to `arch.StationaryBootstrap`, with a
Politis-White rule of thumb when `block_size=None`.

## Confidence level vs chi-squared kappa

`MuUncertaintySetConfig.confidence_level` (and the covariance
counterpart) is a probability in `(0, 1)` — for example `0.95`. The
confidence ball is parameterised by chi-squared quantiles internally,
so a `confidence_level` of 0.95 corresponds to a chi-squared kappa
that is automatically derived by skfolio. **You do not pass kappa
directly** — only the confidence level. Higher confidence ⇒ wider
ball ⇒ more conservative weights.

## Composition pattern

The two Configs feed `RobustMeanRiskConfig.mu_uncertainty_set_config`
and `RobustMeanRiskConfig.covariance_uncertainty_set_config`. Leaving
both at `None` recovers plain `MeanRisk` exactly (atol=1e-8 by
contract).

```python
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.optimization import (
    MeanRiskConfig,
    RobustMeanRiskConfig,
    build_robust_mean_risk,
)
from optimizer.uncertainty_set import (
    CovarianceUncertaintySetConfig,
    MuUncertaintySetConfig,
)


prices = load_sp500_dataset()
returns = prices_to_returns(prices)

cfg = RobustMeanRiskConfig(
    mean_risk_config=MeanRiskConfig.for_min_variance(),
    mu_uncertainty_set_config=MuUncertaintySetConfig.for_empirical(
        confidence_level=0.95
    ),
    covariance_uncertainty_set_config=CovarianceUncertaintySetConfig.for_bootstrap(
        n_bootstrap_samples=200, random_state=0
    ),
)
optimizer = build_robust_mean_risk(cfg)
optimizer.fit(returns)
print(optimizer.weights_)
```

## DR-CVaR

For tail-risk hedging via Wasserstein distributional robustness, see
`DRCVaRConfig` and the `examples/dr_cvar.py` script. `epsilon=0`
falls back to plain `MeanRisk(CVaR)` exactly.

## See also

- skfolio reference:
  `~/.claude/skills/skfolio/references/optimization.md`.
- `examples/robust_optimization.py`, `examples/dr_cvar.py`.
