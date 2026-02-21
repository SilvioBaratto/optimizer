"""Full pipeline: prices to validated weights with rebalancing.

Uses the skfolio S&P 500 dataset (2015-2022) and demonstrates
run_full_pipeline with pre-selection, walk-forward validation,
and threshold-based rebalancing analysis.
"""

import numpy as np
from skfolio.datasets import load_sp500_dataset

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.pre_selection import PreSelectionConfig
from optimizer.rebalancing import ThresholdRebalancingConfig
from optimizer.validation import WalkForwardConfig

# --- Load real price data ---
prices = load_sp500_dataset()
prices = prices.loc["2015":]
n_assets = len(prices.columns)

# --- Configure components ---
optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
cv_config = WalkForwardConfig.for_quarterly_rolling()
presel_config = PreSelectionConfig()
rebal_config = ThresholdRebalancingConfig(threshold=0.05)

# Simulate previous weights (equal-weight as starting point)
prev_weights = np.full(n_assets, 1.0 / n_assets)

# --- Run full pipeline ---
result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    pre_selection_config=presel_config,
    cv_config=cv_config,
    previous_weights=prev_weights,
    rebalancing_config=rebal_config,
)

print("Optimal weights:")
print(result.weights.round(4))
print()
print("Portfolio summary:")
print(result.summary)
print()
print(f"Rebalance needed: {result.rebalance_needed}")
if result.turnover is not None:
    print(f"Turnover: {result.turnover:.4f}")
