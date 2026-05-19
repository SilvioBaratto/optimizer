"""Distributionally Robust CVaR at three Wasserstein-ball epsilon levels.

Fits ``DistributionallyRobustCVaR`` with epsilon ∈ {0.01, 0.05, 0.10}
and prints a side-by-side weights table for the bundled SP500 dataset.

Run:
    python examples/dr_cvar.py
"""

from __future__ import annotations

import pandas as pd
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.optimization import DRCVaRConfig, build_dr_cvar


def _fit_weights(returns: pd.DataFrame, epsilon: float) -> pd.Series:
    optimizer = build_dr_cvar(DRCVaRConfig(epsilon=epsilon))
    optimizer.fit(returns)
    return pd.Series(optimizer.weights_, index=returns.columns)


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices).tail(500)

    table = pd.DataFrame(
        {f"eps={eps:.2f}": _fit_weights(returns, eps) for eps in (0.01, 0.05, 0.10)}
    )

    print("DR-CVaR weights at three epsilon levels:")
    print(table.round(4).to_string())
    print()
    print("Per-column sums:")
    print(table.sum().round(4).to_string())


if __name__ == "__main__":
    main()
