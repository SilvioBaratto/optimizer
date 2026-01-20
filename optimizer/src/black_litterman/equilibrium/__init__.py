"""
Equilibrium Module - Market equilibrium returns and risk aversion estimation.

Provides:
- EquilibriumCalculator: Calculate market-implied expected returns
- RiskAversionEstimator: Estimate market risk aversion coefficient

Usage:
    from src.black_litterman.equilibrium import EquilibriumCalculator, RiskAversionEstimator

    risk_estimator = RiskAversionEstimator()
    delta = risk_estimator.estimate(market_returns, risk_free_rate)

    eq_calculator = EquilibriumCalculator()
    pi = eq_calculator.calculate(market_caps, cov_matrix, delta, risk_free_rate)
"""

from optimizer.src.black_litterman.equilibrium.calculator import EquilibriumCalculatorImpl
from optimizer.src.black_litterman.equilibrium.risk_aversion import (
    RiskAversionEstimatorImpl,
    RegimeAdjustedRiskAversion,
)

__all__ = [
    "EquilibriumCalculatorImpl",
    "RiskAversionEstimatorImpl",
    "RegimeAdjustedRiskAversion",
]
