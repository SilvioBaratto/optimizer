# Portfolio Optimizer

Quantitative portfolio construction and optimization platform built on [skfolio](https://skfolio.org/) and scikit-learn.

## Features

- **Preprocessing** -- Data validation, outlier treatment, sector imputation, regression imputation
- **Pre-selection** -- Asset filtering (completeness, variance, correlation, dominance)
- **Moments** -- Expected return and covariance estimation with HMM regime blending
- **Views** -- Black-Litterman, Entropy Pooling, Opinion Pooling
- **Optimization** -- Mean-Risk, Risk Budgeting, HRP/HERC/NCO, robust and DR-CVaR variants
- **Validation** -- Walk-Forward, Combinatorial Purged CV, Multiple Randomized CV
- **Rebalancing** -- Calendar-based, threshold-based, and hybrid rebalancing
- **Factors** -- Factor construction, standardization, composite scoring, stock selection
- **Pipeline** -- End-to-end orchestration from prices to validated weights

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
result = run_full_pipeline(
    prices=price_df,
    optimizer=optimizer,
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)
```

See the [Quickstart guide](getting-started/quickstart.md) for a complete example.
