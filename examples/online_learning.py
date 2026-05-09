"""Online learning example: rolling Sharpe via incremental ``partial_fit``.

Compares ``optimizer.online.run_online_predict`` (incremental updates
of ``EWMu`` + ``EWCovariance`` inside ``EmpiricalPrior`` inside
``MeanRisk``) against a full-refit walk-forward baseline. Uses the
skfolio bundled SP500 dataset — no DB or API keys.

Run:
    python examples/online_learning.py
"""

from __future__ import annotations

import numpy as np
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

from optimizer.online import OnlinePredictConfig, run_online_predict

WARMUP = 252


def _build_estimator() -> MeanRisk:
    """Mean-risk with EW prior — partial-fit-capable."""
    return MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40.0),
            covariance_estimator=EWCovariance(half_life=40.0),
        )
    )


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices)

    online_portfolio = run_online_predict(
        _build_estimator(),
        returns,
        None,
        config=OnlinePredictConfig(warmup_size=WARMUP),
    )

    cv = WalkForward(train_size=WARMUP, test_size=1)
    full_refit_portfolio = cross_val_predict(
        _build_estimator(),
        returns,
        cv=cv,
    )

    online_sharpe = float(online_portfolio.annualized_sharpe_ratio)
    full_refit_sharpe = float(full_refit_portfolio.annualized_sharpe_ratio)
    diff = online_sharpe - full_refit_sharpe

    print(f"Warmup size:        {WARMUP}")
    print(f"Out-of-sample obs:  {len(online_portfolio.returns)}")
    print(f"Online Sharpe:      {online_sharpe:.4f}")
    print(f"Full-refit Sharpe:  {full_refit_sharpe:.4f}")
    print(f"Difference:         {diff:+.4f}")
    close = np.isclose(online_sharpe, full_refit_sharpe, atol=1e-6)
    print(f"Allclose 1e-6:      {close}")


if __name__ == "__main__":
    main()
