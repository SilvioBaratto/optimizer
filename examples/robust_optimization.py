"""Robust mean-risk with ellipsoidal mu uncertainty.

Compares standard ``MeanRisk`` against three robust variants
(``conservative``, ``moderate``, ``aggressive``) that wrap the
optimizer with mu uncertainty-set estimators at three confidence
levels. Uses the bundled SP500 dataset.

Run:
    python examples/robust_optimization.py
"""

from __future__ import annotations

import pandas as pd
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.optimization import (
    MeanRiskConfig,
    RobustMeanRiskConfig,
    build_mean_risk,
    build_robust_mean_risk,
)


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices)

    plain = build_mean_risk(MeanRiskConfig.for_min_variance())
    plain.fit(returns)

    presets = {
        "conservative_99": RobustMeanRiskConfig.for_conservative(),
        "moderate_95": RobustMeanRiskConfig.for_moderate(),
        "aggressive_90": RobustMeanRiskConfig.for_aggressive(),
    }
    columns = {
        "plain_min_var": pd.Series(plain.weights_, index=returns.columns),
    }
    for name, cfg in presets.items():
        # Inject min-variance MeanRisk config into the robust wrapper.
        cfg = RobustMeanRiskConfig(
            mean_risk_config=MeanRiskConfig.for_min_variance(),
            mu_uncertainty_set_config=cfg.mu_uncertainty_set_config,
            covariance_uncertainty_set_config=(
                cfg.covariance_uncertainty_set_config
            ),
        )
        optimizer = build_robust_mean_risk(cfg)
        optimizer.fit(returns)
        columns[name] = pd.Series(optimizer.weights_, index=returns.columns)

    table = pd.DataFrame(columns)
    print("Robust mean-risk weights at three confidence levels:")
    print(table.round(4).to_string())
    print()
    print("Per-column sums:")
    print(table.sum().round(4).to_string())


if __name__ == "__main__":
    main()
