"""Regime blending: HMM-driven moment estimation.

Uses the skfolio S&P 500 dataset, selects 5 assets, fits a
2-state Gaussian HMM, and prints regime-conditional statistics.
"""

from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.moments import HMMConfig, fit_hmm

# --- Load real data and convert to returns ---
prices = load_sp500_dataset()
# Use a subset of assets for clearer regime visualization
assets = ["AAPL", "JPM", "XOM", "JNJ", "GE"]
returns = prices_to_returns(prices[assets])

# --- Fit HMM ---
config = HMMConfig(n_states=2, n_iter=100, random_state=42)
result = fit_hmm(returns, config)

print("Transition matrix:")
print(result.transition_matrix.round(4))

print("\nRegime means (annualized):")
print((result.regime_means * 252).round(4))

print("\nLog-likelihood:")
print(round(result.log_likelihood, 2))

print("\nFiltered regime probabilities (last 5 days):")
print(result.filtered_probs.tail())
