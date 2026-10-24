# Optimizer config skeletons — reference

Read-only support for `optimization-objective-map`. Each block is the optimizer-library
config a profile objective maps to; `ConstraintSet.to_mean_risk_config()` builds the
`MeanRiskConfig` ones from the stored profile in a single call. They show the *shape*
of the intent so you can explain it. `optimize_portfolio(universe)` does not accept
them: the tool builds its own config from the stored ConstraintSet (the fields listed
in the skill), so never present a skeleton as what was run, and never invent a knob
the skeleton shows but the ConstraintSet does not carry.

Names used below come from the ConstraintSet: `a_gamma` (risk aversion), `beta` (tail
confidence), `cap` (`bounds.max_weights`), `cardinality`. Each `MeanRiskConfig` block
sets every knob the mapper sets, so it equals `to_mean_risk_config()` field for field:
`risk_aversion` only acts under MAXIMIZE_UTILITY and each tail confidence only under
its own measure (`cvar_beta` for CVAR, `cdar_beta` for CDAR), so swapping the risk
measure needs no other edit. Feed LINEAR returns (`prices_to_returns`) and keep
`shuffle=False` in any CV.

## protection → MINIMIZE_RISK (QP / LP)

```python
from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
)

config = MeanRiskConfig(
    objective=ObjectiveFunctionType.MINIMIZE_RISK,
    risk_measure=RiskMeasureType.CVAR,  # or VARIANCE / SEMI_VARIANCE / CDAR / MAX_DRAWDOWN
    risk_aversion=a_gamma,
    cvar_beta=beta,
    cdar_beta=beta,
    min_weights=0.0,
    max_weights=cap,
    budget=1.0,
    cardinality=cardinality,
)
model = build_mean_risk(config)
```

## income / growth → MAXIMIZE_UTILITY (QP)

```python
from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
)

config = MeanRiskConfig(
    objective=ObjectiveFunctionType.MAXIMIZE_UTILITY,
    risk_measure=RiskMeasureType.VARIANCE,
    risk_aversion=a_gamma,
    cvar_beta=beta,
    cdar_beta=beta,
    min_weights=0.0,
    max_weights=cap,
    budget=1.0,
    cardinality=cardinality,
)
model = build_mean_risk(config)
```

## max → MAXIMIZE_RATIO (SOCP)

```python
from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
)

config = MeanRiskConfig(
    objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
    risk_measure=RiskMeasureType.VARIANCE,
    risk_aversion=a_gamma,
    cvar_beta=beta,
    cdar_beta=beta,
    risk_free_rate=0.0,
    min_weights=0.0,
    max_weights=cap,
    budget=1.0,
    cardinality=cardinality,
)
model = build_mean_risk(config)
```

## Prudent profiles — robust uncertainty set on mu

```python
from optimizer.optimization import RobustMeanRiskConfig, build_robust_mean_risk

config = RobustMeanRiskConfig.for_moderate(uncertainty_level=0.95)
model = build_robust_mean_risk(config)
```

## Turnover penalty against the current book

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk

config = MeanRiskConfig(transaction_costs=0.001)
model = build_mean_risk(config, previous_weights=previous_weights)
```

## Not reachable from a profile

`RiskBudgeting` and `MaximumDiversification` exist in the optimizer library, but no
`ObjectiveChoice` maps to them.

```python
from optimizer.optimization import (
    MaxDiversificationConfig,
    RiskBudgetingConfig,
    build_max_diversification,
    build_risk_budgeting,
)

erc = build_risk_budgeting(RiskBudgetingConfig())
max_div = build_max_diversification(MaxDiversificationConfig())
```
