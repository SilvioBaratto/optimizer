"""
Optimization Module - Constrained portfolio weight optimization.

Provides:
- ConstrainedOptimizerImpl: Quadratic programming with constraints
- SectorConstraintBuilder: Build sector weight constraints

Usage:
    from src.black_litterman.optimization import ConstrainedOptimizerImpl

    optimizer = ConstrainedOptimizerImpl()
    weights = optimizer.optimize(
        posterior_returns=mu,
        covariance_matrix=sigma,
        risk_aversion=delta,
        sector_mapping=sectors,
    )
"""

from optimizer.src.black_litterman.optimization.constrained_optimizer import ConstrainedOptimizerImpl
from optimizer.src.black_litterman.optimization.sector_constraints import SectorConstraintBuilder

__all__ = [
    "ConstrainedOptimizerImpl",
    "SectorConstraintBuilder",
]
