"""
Black-Litterman Portfolio Optimization Module
==============================================

Implements Black-Litterman portfolio optimization with:
- AI-driven view generation using BAML
- Robust covariance estimation (Ledoit-Wolf shrinkage)
- Market-implied equilibrium priors
- Regime-adaptive risk aversion

Integration with Portfolio Pipeline:
    Macro Regime → Stock Analyzer → Concentrated Portfolio Builder → Black-Litterman Optimizer

Key Components:
    - BlackLittermanOptimizer: Main optimization engine
    - ViewGenerator: AI-powered view generation from stock signals
    - equilibrium: Market equilibrium calculations
    - risk_models: Covariance estimation (Ledoit-Wolf, sample, exponential)
    - black_litterman: Core BL mathematics
"""

from .portfolio_optimizer import BlackLittermanOptimizer
from .view_generator import ViewGenerator
from .equilibrium import (
    calculate_equilibrium_prior,
    estimate_risk_aversion,
    adjust_risk_aversion_for_regime,
    fetch_market_caps_from_db,
    calculate_implied_returns_from_weights
)

__all__ = [
    'BlackLittermanOptimizer',
    'ViewGenerator',
    'calculate_equilibrium_prior',
    'estimate_risk_aversion',
    'adjust_risk_aversion_for_regime',
    'fetch_market_caps_from_db',
    'calculate_implied_returns_from_weights',
]
