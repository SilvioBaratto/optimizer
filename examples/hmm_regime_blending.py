"""Regime-blended Mean-Risk with HMM-fitted (filtered) regime probabilities.

Same pipeline as ``examples/regime_blending.py``, but the 2-regime
probability matrix comes from a fitted Gaussian HMM
(``optimizer.regime.fit_hmm_regime_probabilities``) instead of a rolling-vol
heuristic. Probabilities are filtered (causal) — safe for walk-forward use.

Run:
    python examples/hmm_regime_blending.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.optimization import (
    RegimeBlendedMeanRiskConfig,
    build_regime_blended_mean_risk,
)
from optimizer.regime import HMMRegimeConfig, fit_hmm_regime_probabilities


def main() -> None:
    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    common_idx = prices.index.intersection(factor_prices.index)
    asset_returns = prices_to_returns(prices.loc[common_idx])
    factor_returns = prices_to_returns(factor_prices.loc[common_idx])

    regime_probs, model = fit_hmm_regime_probabilities(
        asset_returns, HMMRegimeConfig.for_two_regime()
    )
    print(f"HMM converged: {model.monitor_.converged}")
    print(f"Regime probabilities shape: {regime_probs.shape}")
    print(regime_probs.tail().round(4).to_string())

    # Factor/asset returns must be aligned to the (feature-warm-up-shortened)
    # regime index before fitting the blended pipeline.
    aligned_idx = asset_returns.index.intersection(regime_probs.index)
    asset_returns = asset_returns.loc[aligned_idx]
    factor_returns = factor_returns.loc[aligned_idx]

    pipeline, _cv = build_regime_blended_mean_risk(
        RegimeBlendedMeanRiskConfig(),
        factor_returns=factor_returns,
        regime_probabilities=regime_probs,
    )
    pipeline.fit(asset_returns, factor_returns)

    final_optimizer = pipeline.steps[-1][1]
    weights = pd.Series(
        np.asarray(final_optimizer.weights_),
        index=asset_returns.columns,
        name="weight",
    ).sort_values(ascending=False)

    print("\nRegime-blended mean-risk weights (HMM-fitted regimes):")
    print(weights.round(4).to_string())
    print(f"\nSum: {weights.sum():.4f}")


if __name__ == "__main__":
    main()
