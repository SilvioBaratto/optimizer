"""Robust optimization: compare different kappa values.

Uses the skfolio S&P 500 dataset (2015-2022) and demonstrates
how increasing kappa (uncertainty aversion) shifts portfolio
allocations toward more conservative positions.
"""

from skfolio.datasets import load_sp500_dataset

from optimizer.optimization import RobustConfig, build_robust_mean_risk
from optimizer.pipeline import run_full_pipeline

# --- Load real price data ---
prices = load_sp500_dataset()
prices = prices.loc["2015":]

# --- Compare robustness levels ---
configs = {
    "Aggressive (kappa=0.5)": RobustConfig.for_aggressive(),
    "Moderate (kappa=1.0)": RobustConfig.for_moderate(),
    "Conservative (kappa=2.0)": RobustConfig.for_conservative(),
}

for label, config in configs.items():
    opt = build_robust_mean_risk(config)
    result = run_full_pipeline(prices=prices, optimizer=opt)
    print(f"\n{label}:")
    print(result.weights.round(4))
