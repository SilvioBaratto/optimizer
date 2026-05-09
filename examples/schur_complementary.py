"""SchurComplementary gamma sweep: HRP anchor → MVP anchor.

Fits ``SchurComplementary`` at five gamma levels and prints a
comparison weights table on the bundled SP500 dataset.

* ``gamma=0.0`` — Hierarchical Risk Parity anchor.
* ``gamma=1.0`` — Minimum Variance anchor.

Run:
    python examples/schur_complementary.py
"""

from __future__ import annotations

import pandas as pd
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.optimization import (
    SchurComplementaryConfig,
    build_schur_complementary,
)


def _fit_weights(returns: pd.DataFrame, gamma: float) -> pd.Series:
    optimizer = build_schur_complementary(SchurComplementaryConfig(gamma=gamma))
    optimizer.fit(returns)
    return pd.Series(optimizer.weights_, index=returns.columns)


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices)

    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    table = pd.DataFrame(
        {f"gamma={g:.2f}": _fit_weights(returns, g) for g in gammas}
    )

    print("Schur Complementary gamma sweep (HRP=0.0 → MVP=1.0):")
    print(table.round(4).to_string())
    print()
    print("Per-column sums:")
    print(table.sum().round(4).to_string())


if __name__ == "__main__":
    main()
