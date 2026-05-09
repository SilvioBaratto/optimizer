"""Regime-blended Mean-Risk with externally-supplied regime probabilities.

Builds a 2-regime probability matrix from rolling-vol percentiles on
the bundled SP500 dataset (high-vol regime when 21-day vol is in the
top quartile of the prior year), then runs ``build_regime_blended_mean_risk``
with the bundled factor dataset.

Run:
    python examples/regime_blending.py
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


def _build_regime_probabilities(returns: pd.DataFrame) -> pd.DataFrame:
    """Two-regime soft labels driven by 21-day cross-section volatility."""
    vol = returns.std(axis=1).rolling(21).mean()
    rolling_q75 = vol.rolling(252, min_periods=60).quantile(0.75)
    high = (vol > rolling_q75).astype(float).fillna(0.0)
    return pd.DataFrame({"low": 1.0 - high, "high": high}, index=returns.index)


def main() -> None:
    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    common_idx = prices.index.intersection(factor_prices.index)
    asset_returns = prices_to_returns(prices.loc[common_idx])
    factor_returns = prices_to_returns(factor_prices.loc[common_idx])
    regime_probs = _build_regime_probabilities(asset_returns)

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

    print("Regime-blended mean-risk weights:")
    print(weights.round(4).to_string())
    print(f"\nSum: {weights.sum():.4f}")


if __name__ == "__main__":
    main()
