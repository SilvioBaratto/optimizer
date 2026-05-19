"""Covariance forecast evaluation: rank estimators independent of optimizer.

Compares ``EmpiricalCovariance``, ``LedoitWolf``, ``EWCovariance(half_life=40)``
on the bundled SP500 dataset using the offline walk-forward path. Prints
the ranking summary table.

Run:
    python examples/covariance_forecast.py
"""

from __future__ import annotations

from skfolio.datasets import load_sp500_dataset
from skfolio.moments import EmpiricalCovariance, EWCovariance, LedoitWolf
from skfolio.preprocessing import prices_to_returns

from optimizer.validation import (
    CovarianceForecastConfig,
    run_covariance_forecast_evaluation,
)


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices).tail(1500)

    estimators = [
        ("empirical", EmpiricalCovariance()),
        ("ledoit_wolf", LedoitWolf()),
        ("ew_40", EWCovariance(half_life=40)),
    ]
    cfg = CovarianceForecastConfig(train_size=252, test_size=21)

    comparison = run_covariance_forecast_evaluation(estimators, returns, config=cfg)

    print("Names:", comparison.names)
    print()
    print("Bias-statistic summary:")
    print(comparison.bias_statistic_summary())
    print()
    print("Exceedance summary:")
    print(comparison.exceedance_summary())


if __name__ == "__main__":
    main()
