"""StackingOptimization blending min-variance MeanRisk and ERC RiskBudgeting.

Combines two base optimizers via a min-variance final estimator. Uses
the bundled SP500 dataset.

Run:
    python examples/stacking.py
"""

from __future__ import annotations

import pandas as pd
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.optimization import (
    MeanRiskConfig,
    RiskBudgetingConfig,
    StackingConfig,
    build_mean_risk,
    build_risk_budgeting,
    build_stacking,
)


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices)

    erc_cfg = RiskBudgetingConfig.for_equal_risk_contribution()
    estimators = [
        ("min_var", build_mean_risk(MeanRiskConfig.for_min_variance())),
        ("erc", build_risk_budgeting(erc_cfg)),
    ]
    optimizer = build_stacking(
        StackingConfig.for_min_variance_final(),
        estimators=estimators,
    )
    optimizer.fit(returns)

    weights = pd.Series(
        optimizer.weights_,
        index=returns.columns,
        name="weight",
    ).sort_values(ascending=False)

    print("Stacking (min_var + ERC) → min-variance final weights:")
    print(weights.round(4).to_string())
    print(f"\nSum: {weights.sum():.4f}")


if __name__ == "__main__":
    main()
