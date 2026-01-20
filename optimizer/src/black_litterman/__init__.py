"""
Black-Litterman Portfolio Optimization Module
==============================================

This is the refactored version following SOLID principles:
- Single Responsibility: Separate modules for covariance, equilibrium, views, optimization
- Open/Closed: Pluggable estimators and optimizers
- Dependency Injection: Components injected via configuration

Components:
- optimizer: Main BlackLittermanOptimizer orchestrator
- covariance: Covariance estimation (Ledoit-Wolf, sample, exponential)
- equilibrium: Market equilibrium calculations, risk aversion
- views: AI-powered view generation, matrix construction
- optimization: Constrained portfolio optimization

Usage:
    from src.black_litterman import BlackLittermanOptimizer
    from config import OptimizerConfig

    optimizer = BlackLittermanOptimizer(config=OptimizerConfig())
    result = await optimizer.optimize(positions, signal_date)

Legacy imports maintained for backward compatibility.
"""

# New SOLID-compliant imports
from optimizer.src.black_litterman.optimizer import BlackLittermanOptimizer as NewBlackLittermanOptimizer

from optimizer.src.black_litterman.covariance import (
    LedoitWolfEstimator,
    SampleCovarianceEstimator,
    ExponentialWeightedEstimator,
    get_covariance_estimator,
)

from optimizer.src.black_litterman.equilibrium import (
    EquilibriumCalculatorImpl,
    RiskAversionEstimatorImpl,
    RegimeAdjustedRiskAversion,
)

from optimizer.src.black_litterman.views import (
    ViewGeneratorImpl,
    ViewMatrixBuilder,
)

from optimizer.src.black_litterman.optimization import (
    ConstrainedOptimizerImpl,
    SectorConstraintBuilder,
)

# Legacy imports for backward compatibility
try:
    from .portfolio_optimizer import BlackLittermanOptimizer
    from .view_generator import ViewGenerator
    from .equilibrium import (
        calculate_equilibrium_prior,
        estimate_risk_aversion,
        adjust_risk_aversion_for_regime,
        fetch_market_caps_from_db,
        calculate_implied_returns_from_weights
    )
    _legacy_available = True
except ImportError:
    _legacy_available = False
    # Use new implementations as fallback
    BlackLittermanOptimizer = NewBlackLittermanOptimizer
    ViewGenerator = ViewGeneratorImpl

__all__ = [
    # Main orchestrator
    "BlackLittermanOptimizer",
    "NewBlackLittermanOptimizer",
    # Covariance
    "LedoitWolfEstimator",
    "SampleCovarianceEstimator",
    "ExponentialWeightedEstimator",
    "get_covariance_estimator",
    # Equilibrium
    "EquilibriumCalculatorImpl",
    "RiskAversionEstimatorImpl",
    "RegimeAdjustedRiskAversion",
    # Views
    "ViewGeneratorImpl",
    "ViewMatrixBuilder",
    "ViewGenerator",
    # Optimization
    "ConstrainedOptimizerImpl",
    "SectorConstraintBuilder",
]

# Add legacy exports if available
if _legacy_available:
    __all__.extend([
        "calculate_equilibrium_prior",
        "estimate_risk_aversion",
        "adjust_risk_aversion_for_regime",
        "fetch_market_caps_from_db",
        "calculate_implied_returns_from_weights",
    ])
