# Quickstart

This guide walks through five progressively complex examples — from a minimal optimization to a full pipeline with stock selection, regime blending, and rebalancing.

## 1. Basic Optimization

The simplest use case: maximize the Sharpe ratio with walk-forward validation.

```python
import pandas as pd
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

# Load price data (DatetimeIndex, one column per asset)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

# Build optimizer from config
optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())

# Run end-to-end pipeline
result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)

print(result.weights)          # pd.Series of asset weights
print(result.summary)          # dict with Sharpe, max drawdown, etc.
print(result.backtest)         # out-of-sample MultiPeriodPortfolio
```

`run_full_pipeline` handles everything internally: price-to-return conversion, pre-selection, optimization, cross-validation, and backtesting. See the [Pipeline Overview](../guide/pipeline.md) for the full data flow.

## 2. Custom Pre-selection and Moments

Control which assets survive filtering and how expected returns and covariance are estimated.

```python
from optimizer.pre_selection import PreSelectionConfig
from optimizer.moments import MomentEstimationConfig
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline

# Drop correlated assets (>85%) and keep top 30 by variance
preselection = PreSelectionConfig(
    correlation_threshold=0.85,
    top_k=30,
)

# Shrunk mu + denoised covariance
moments = MomentEstimationConfig.for_shrunk_denoised()

# Minimum CVaR optimization
optimizer = build_mean_risk(
    MeanRiskConfig.for_min_cvar(beta=0.95),
    moment_config=moments,
)

result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    preselection_config=preselection,
    sector_mapping={"AAPL": "Tech", "JPM": "Financials", ...},
)
```

The `sector_mapping` dict enables sector-aware imputation during preprocessing. See [Preprocessing](../guide/preprocessing.md) and [Pre-selection](../guide/pre-selection.md).

## 3. Black-Litterman Views

Incorporate analyst views into the optimization through the Black-Litterman framework.

```python
from optimizer.views import BlackLittermanConfig
from optimizer.moments import MomentEstimationConfig
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline

# Define views: AAPL returns 12%, MSFT outperforms GOOG by 3%
bl_config = BlackLittermanConfig.for_equilibrium(
    views=("AAPL == 0.12", "MSFT - GOOG == 0.03"),
    tau=0.05,
)

# Build optimizer with BL prior
optimizer = build_mean_risk(
    MeanRiskConfig.for_max_utility(risk_aversion=1.0),
    bl_config=bl_config,
)

result = run_full_pipeline(prices=prices, optimizer=optimizer)
print(result.weights)
```

Views use a string syntax: `"TICKER == value"` for absolute views, `"TICKER1 - TICKER2 == value"` for relative views. See [Views](../guide/views.md) for Entropy Pooling and Opinion Pooling alternatives.

## 4. HMM Regime Blending

Use a Hidden Markov Model to blend moments across market regimes, producing estimates that adapt to the current regime.

```python
from skfolio.preprocessing import prices_to_returns
from optimizer.moments import (
    HMMConfig,
    fit_hmm,
    HMMBlendedMu,
    HMMBlendedCovariance,
    MomentEstimationConfig,
)
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline

returns = prices_to_returns(prices)

# Fit 2-state HMM
hmm_result = fit_hmm(returns.values, config=HMMConfig(n_states=2))
print(f"Current regime: {hmm_result.filtered_probs[-1]}")

# Build optimizer with regime-blended moments
optimizer = build_mean_risk(
    MeanRiskConfig.for_max_sharpe(),
    moment_config=MomentEstimationConfig.for_hmm_blended(),
)

result = run_full_pipeline(prices=prices, optimizer=optimizer)
```

`HMMBlendedCovariance` uses the full law of total variance (including between-regime mean dispersion), while `blend_moments_by_regime()` uses within-regime covariance only. Use the class for optimizer inputs. See [Moments](../guide/moments.md) for details.

## 5. Full Pipeline with Rebalancing

Combine optimization with threshold-based rebalancing to determine whether to trade.

```python
import numpy as np
import pandas as pd
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.rebalancing import HybridRebalancingConfig
from optimizer.pipeline import run_full_pipeline

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())

# Current portfolio weights (from previous period)
previous_weights = np.array([0.25, 0.25, 0.25, 0.25])

result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    previous_weights=previous_weights,
    rebalancing_config=HybridRebalancingConfig.for_monthly_with_5pct_threshold(),
    current_date=pd.Timestamp("2024-06-28"),
    last_review_date=pd.Timestamp("2024-05-31"),
)

if result.rebalance_needed:
    print(f"Rebalance! Turnover: {result.turnover:.2%}")
    print(f"New weights: {result.weights}")
else:
    print("No rebalance needed — drift within threshold")
```

Hybrid rebalancing checks drift only at calendar review dates, preventing over-trading between reviews. See [Rebalancing](../guide/rebalancing.md) for calendar, threshold, and hybrid strategies.

## Next Steps

| Want to... | Read |
|------------|------|
| Understand the full data flow | [Pipeline Overview](../guide/pipeline.md) |
| Clean and impute return data | [Preprocessing](../guide/preprocessing.md) |
| Estimate expected returns and covariance | [Moments](../guide/moments.md) |
| Add analyst views | [Views](../guide/views.md) |
| Choose an optimization model | [Optimization](../guide/optimization.md) |
| Validate out-of-sample | [Validation](../guide/validation.md) |
| Tune hyperparameters | [Tuning](../guide/tuning.md) |
| Run factor-based stock selection | [Factors](../guide/factors.md) |
| Generate synthetic scenarios | [Synthetic Data](../guide/synthetic.md) |
| Screen an investable universe | [Universe Screening](../guide/universe.md) |

See the [`examples/`](https://github.com/SilvioBaratto/optimizer/tree/main/examples) directory for complete, runnable scripts.
