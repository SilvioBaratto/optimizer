# Quickstart

## Basic Optimization

```python
import pandas as pd
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

# Load price data (DatetimeIndex, one column per asset)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

# Configure optimizer and validation
optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
cv_config = WalkForwardConfig.for_quarterly_rolling()

# Run pipeline
result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    cv_config=cv_config,
)

print(result.weights)
print(result.summary)
```

## With Rebalancing

```python
from optimizer.rebalancing import ThresholdRebalancingConfig

result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    previous_weights=current_weights,
    rebalancing_config=ThresholdRebalancingConfig(threshold=0.05),
)

print(f"Should rebalance: {result.should_rebalance}")
```

## More Examples

See the [`examples/`](https://github.com/SilvioBaratto/optimizer/tree/main/examples) directory for complete, runnable scripts.
