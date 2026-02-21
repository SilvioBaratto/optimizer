# Moment Estimation

Expected return and covariance estimation with optional regime blending.

## Estimator Types

- Empirical, shrinkage, and denoised covariance estimators
- HMM regime-conditional blending via `HMMBlendedMu` / `HMMBlendedCovariance`
- Deep Markov Model (optional, requires `torch` and `pyro-ppl`)
- Lognormal correction and multi-period scaling

## HMM Blending

```python
from optimizer.moments import HMMConfig, fit_hmm

config = HMMConfig(n_states=2, random_state=42)
result = fit_hmm(returns, config)
print(result.transition_matrix)
print(result.filtered_probabilities.tail())
```
