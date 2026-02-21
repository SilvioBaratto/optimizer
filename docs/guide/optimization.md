# Optimization

Portfolio optimization models with convex, hierarchical, and ensemble approaches.

## Models

| Model | Factory | Config |
|-------|---------|--------|
| Mean-Risk | `build_mean_risk()` | `MeanRiskConfig` |
| Risk Budgeting | `build_risk_budgeting()` | `RiskBudgetingConfig` |
| HRP | `build_hrp()` | `HRPConfig` |
| HERC | `build_herc()` | `HERCConfig` |
| NCO | `build_nco()` | `NCOConfig` |
| Max Diversification | `build_max_diversification()` | `MaxDiversificationConfig` |
| Benchmark Tracker | `build_benchmark_tracker()` | `BenchmarkTrackerConfig` |
| Equal Weighted | `build_equal_weighted()` | -- |
| Inverse Volatility | `build_inverse_volatility()` | -- |
| Stacking | `build_stacking()` | `StackingConfig` |

## Robust Variants

- **RobustConfig** -- Ellipsoidal uncertainty sets for expected returns
- **DRCVaRConfig** -- Distributionally robust CVaR over Wasserstein ball
- **RegimeRiskConfig** -- HMM-driven regime-conditional risk measures
