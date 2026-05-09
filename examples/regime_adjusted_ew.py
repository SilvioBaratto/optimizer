"""RegimeAdjustedEWCovariance STVU multiplier impact during a vol shock.

Fits ``EWCovariance`` and ``RegimeAdjustedEWCovariance`` on the bundled
SP500 dataset and prints a comparison of average rolling 60-day
portfolio volatility implied by each covariance series.

Run:
    python examples/regime_adjusted_ew.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skfolio.datasets import load_sp500_dataset
from skfolio.moments import EWCovariance, RegimeAdjustedEWCovariance
from skfolio.preprocessing import prices_to_returns


def _portfolio_vol(cov: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices).tail(750)
    n = returns.shape[1]
    weights = np.full(n, 1.0 / n)

    plain = EWCovariance(half_life=40.0)
    plain.fit(returns)
    regime = RegimeAdjustedEWCovariance(half_life=23.0, corr_half_life=50.0)
    regime.fit(returns)

    plain_vol = _portfolio_vol(plain.covariance_, weights)
    regime_vol = _portfolio_vol(regime.covariance_, weights)
    multiplier = regime_vol / plain_vol if plain_vol > 0 else float("nan")

    table = pd.DataFrame(
        {
            "estimator": ["EWCovariance", "RegimeAdjustedEW"],
            "EW portfolio vol": [plain_vol, regime_vol],
        }
    )

    print("Regime-adjusted EW covariance vs plain EW (equal-weight portfolio):")
    print(table.round(6).to_string(index=False))
    print(f"\nSTVU multiplier (regime / plain): {multiplier:.4f}")


if __name__ == "__main__":
    main()
