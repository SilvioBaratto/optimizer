# Rebalancing

Calendar-based, threshold-based, and hybrid rebalancing strategies.

## Strategies

- **CalendarRebalancingConfig** -- Fixed-interval (21/63/126/252 trading days)
- **ThresholdRebalancingConfig** -- Drift-based (absolute or relative thresholds)
- **HybridRebalancingConfig** -- Calendar-gated threshold: checks drift only at review dates

## Utilities

- `compute_drifted_weights()` -- Weight drift after one period of returns
- `compute_turnover()` -- One-way turnover between portfolios
- `compute_rebalancing_cost()` -- Transaction cost estimation
