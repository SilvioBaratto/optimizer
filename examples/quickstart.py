"""Quickstart: MeanRisk optimization with walk-forward backtest.

Uses the skfolio S&P 500 dataset (20 assets) sliced to 2015-2022
and runs a minimum-variance portfolio with quarterly rolling
walk-forward validation.
"""

from skfolio.datasets import load_sp500_dataset

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

# --- Load real price data (recent period for stable walk-forward) ---
prices = load_sp500_dataset()
prices = prices.loc["2015":]

# --- Configure optimizer ---
optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
cv_config = WalkForwardConfig.for_quarterly_rolling()

# --- Run pipeline ---
result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    cv_config=cv_config,
)

print("Optimal weights:")
print(result.weights.round(4))
print()
print("Portfolio summary:")
print(result.summary)
