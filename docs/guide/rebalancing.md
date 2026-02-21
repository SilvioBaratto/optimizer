# Rebalancing

The rebalancing module implements calendar-based, threshold-based, and hybrid rebalancing strategies for portfolio management. It determines **when** to trade (the rebalancing signal) and provides utility functions for computing drift, turnover, and transaction costs.

## Overview

After optimization produces target weights, the rebalancing module answers the question: "Should we actually trade to reach these weights?" Trading too frequently incurs unnecessary transaction costs, while trading too infrequently allows the portfolio to drift far from optimal allocations. The three strategies offer different trade-offs:

- **Calendar** — rebalance at fixed intervals regardless of drift
- **Threshold** — rebalance only when drift exceeds a limit
- **Hybrid** — check drift only at calendar review dates (best of both worlds)

## Calendar Rebalancing

Triggers portfolio reconstruction at fixed time intervals regardless of how much the portfolio has drifted.

```python
from optimizer.rebalancing import CalendarRebalancingConfig, RebalancingFrequency

config = CalendarRebalancingConfig(
    frequency=RebalancingFrequency.QUARTERLY,
)
```

| Frequency | Trading Days | Approximate Period |
|-----------|-------------|-------------------|
| `MONTHLY` | 21 | 1 month |
| `QUARTERLY` | 63 | 3 months |
| `SEMIANNUAL` | 126 | 6 months |
| `ANNUAL` | 252 | 1 year |

### Presets

```python
CalendarRebalancingConfig.for_monthly()      # 21 trading days
CalendarRebalancingConfig.for_quarterly()     # 63 trading days
CalendarRebalancingConfig.for_semiannual()    # 126 trading days
CalendarRebalancingConfig.for_annual()        # 252 trading days
```

## Threshold Rebalancing

Rebalances only when portfolio drift exceeds specified limits. This avoids unnecessary turnover during stable periods while catching significant deviations.

```python
from optimizer.rebalancing import ThresholdRebalancingConfig, ThresholdType

config = ThresholdRebalancingConfig(
    threshold_type=ThresholdType.ABSOLUTE,
    threshold=0.05,  # 5 percentage points
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `threshold_type` | `ThresholdType` | `ABSOLUTE` | Drift measurement method |
| `threshold` | `float` | 0.05 | Drift limit triggering rebalance |

### Absolute vs Relative Thresholds

- **Absolute** (`ThresholdType.ABSOLUTE`): Triggers when any asset's weight deviates by more than `threshold` percentage points from its target. E.g., `threshold=0.05` means a 25% target weight triggers rebalancing when it drifts below 20% or above 30%.

- **Relative** (`ThresholdType.RELATIVE`): Triggers when any asset's weight deviates by more than `threshold` fraction of its target. E.g., `threshold=0.25` means a 20% target triggers at 15% or 25% (25% of 20% = 5pp).

### Presets

```python
ThresholdRebalancingConfig.for_absolute(threshold=0.05)  # 5pp absolute
ThresholdRebalancingConfig.for_relative(threshold=0.25)   # 25% relative
```

## Hybrid Rebalancing

Combines calendar and threshold strategies: the portfolio is reviewed at regular calendar intervals, but trades are executed only when drift exceeds the threshold at the review date. Between review dates, `should_rebalance_hybrid` always returns `False` regardless of drift.

This is the recommended strategy for most institutional portfolios — it reduces monitoring overhead (only check at review dates) while avoiding unnecessary trades (only trade when drift is significant).

```python
from optimizer.rebalancing import HybridRebalancingConfig

config = HybridRebalancingConfig(
    calendar=CalendarRebalancingConfig.for_monthly(),
    threshold=ThresholdRebalancingConfig.for_absolute(threshold=0.05),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `calendar` | `CalendarRebalancingConfig` | Quarterly | Review schedule |
| `threshold` | `ThresholdRebalancingConfig` | 5pp absolute | Drift threshold at review |

### Presets

```python
HybridRebalancingConfig.for_monthly_with_5pct_threshold()
# Monthly reviews, rebalance only if 5pp drift

HybridRebalancingConfig.for_quarterly_with_10pct_threshold()
# Quarterly reviews, rebalance only if 10pp drift
```

## Decision Functions

### should_rebalance

Checks whether the portfolio should be rebalanced based on threshold drift:

```python
from optimizer.rebalancing import should_rebalance, ThresholdRebalancingConfig
import numpy as np

previous = np.array([0.25, 0.25, 0.25, 0.25])
current = np.array([0.30, 0.20, 0.28, 0.22])

config = ThresholdRebalancingConfig(threshold=0.05)
needs_rebalance = should_rebalance(previous, current, config=config)
print(needs_rebalance)  # True — 5pp drift in first asset
```

### should_rebalance_hybrid

Checks both the calendar gate and the threshold:

```python
from optimizer.rebalancing import should_rebalance_hybrid, HybridRebalancingConfig
import pandas as pd

config = HybridRebalancingConfig.for_monthly_with_5pct_threshold()
needs_rebalance = should_rebalance_hybrid(
    previous, current, config,
    current_date=pd.Timestamp("2024-03-15"),
    last_review_date=pd.Timestamp("2024-02-15"),
)
```

## Utility Functions

### compute_drifted_weights

Compute what weights would be after one period of returns (without rebalancing):

```python
from optimizer.rebalancing import compute_drifted_weights
import numpy as np

weights = np.array([0.50, 0.30, 0.20])
returns = np.array([0.02, -0.01, 0.03])

drifted = compute_drifted_weights(weights, returns)
print(drifted)  # Weights after market movements, normalized to sum to 1
```

### compute_turnover

One-way turnover between two weight vectors:

```python
from optimizer.rebalancing import compute_turnover
import numpy as np

old_weights = np.array([0.25, 0.25, 0.25, 0.25])
new_weights = np.array([0.30, 0.20, 0.30, 0.20])

turnover = compute_turnover(old_weights, new_weights)
print(f"Turnover: {turnover:.2%}")  # 10%
```

### compute_rebalancing_cost

Transaction cost estimation based on turnover:

```python
from optimizer.rebalancing import compute_rebalancing_cost
import numpy as np

cost = compute_rebalancing_cost(
    old_weights=np.array([0.25, 0.25, 0.25, 0.25]),
    new_weights=np.array([0.30, 0.20, 0.30, 0.20]),
    cost_bps=10.0,  # 10 basis points per unit of turnover
)
print(f"Cost: {cost:.4%}")
```

## Code Examples

### Rebalancing in the pipeline

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.rebalancing import ThresholdRebalancingConfig
import numpy as np

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())

result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    previous_weights=np.array([0.25, 0.25, 0.25, 0.25]),
    rebalancing_config=ThresholdRebalancingConfig(threshold=0.05),
)

if result.rebalance_needed:
    print(f"Rebalance! Turnover: {result.turnover:.2%}")
    print(f"New weights: {result.weights}")
else:
    print("No rebalance needed — drift within threshold")
```

### Hybrid rebalancing in the pipeline

```python
from optimizer.rebalancing import HybridRebalancingConfig
import pandas as pd

result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    previous_weights=current_portfolio_weights,
    rebalancing_config=HybridRebalancingConfig.for_monthly_with_5pct_threshold(),
    current_date=pd.Timestamp("2024-06-28"),
    last_review_date=pd.Timestamp("2024-05-31"),
)
```

## Gotchas and Tips

!!! warning "Hybrid always returns False between reviews"
    `should_rebalance_hybrid` returns `False` between calendar review dates regardless of drift. This is by design — it prevents over-trading. If you need continuous monitoring, use `ThresholdRebalancingConfig` alone.

!!! tip "previous_weights alignment"
    When passed to `run_full_pipeline()`, previous weights are automatically aligned to the post-pre-selection universe and re-normalized. Assets dropped by pre-selection have their weights set to zero.

!!! tip "Calendar frequency constants"
    The `TRADING_DAYS` dictionary maps each `RebalancingFrequency` to its trading-day count: `{MONTHLY: 21, QUARTERLY: 63, SEMIANNUAL: 126, ANNUAL: 252}`.

!!! tip "Cost estimation"
    `compute_rebalancing_cost` uses a simple proportional model: `cost = turnover * cost_bps / 10000`. For more realistic costs, consider bid-ask spreads, market impact, and commission schedules.

## Quick Reference

| Task | Code |
|------|------|
| Monthly calendar | `CalendarRebalancingConfig.for_monthly()` |
| 5pp absolute threshold | `ThresholdRebalancingConfig.for_absolute(0.05)` |
| 25% relative threshold | `ThresholdRebalancingConfig.for_relative(0.25)` |
| Monthly + 5pp hybrid | `HybridRebalancingConfig.for_monthly_with_5pct_threshold()` |
| Check rebalance | `should_rebalance(prev, new, config=cfg)` |
| Compute turnover | `compute_turnover(old, new)` |
| Estimate costs | `compute_rebalancing_cost(old, new, cost_bps=10)` |
| Drifted weights | `compute_drifted_weights(weights, returns)` |
