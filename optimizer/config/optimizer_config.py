"""
Optimizer Configuration - Externalized settings for Black-Litterman optimization.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OptimizerConfig:
    """
    Configuration for Black-Litterman portfolio optimization.

    All parameters for the optimization process are externalized here.

    Attributes:
        # Black-Litterman parameters
        tau: Scaling factor for uncertainty in equilibrium (typically 0.025-0.1)
        risk_aversion: Market risk aversion coefficient (typically 2.5-5.0)
        risk_free_rate: Annual risk-free rate (decimal)

        # Risk aversion by country (for multi-country portfolios)
        use_country_risk_aversion: Whether to use country-specific risk aversion

        # View parameters
        view_confidence_scaling: Scale factor for BAML view confidences
        min_view_confidence: Minimum view confidence to include
        max_view_weight: Maximum weight a single view can have on posterior

        # Covariance estimation
        covariance_method: Method for covariance estimation
        lookback_days: Historical days for covariance estimation
        shrinkage_target: Target for shrinkage (None for auto)

        # Optimization constraints
        max_iterations: Maximum optimizer iterations
        tolerance: Convergence tolerance
        regularization: L2 regularization for numerical stability

        # Factor priors
        use_factor_priors: Whether to incorporate factor-based priors
        factor_weight: Weight given to factor priors vs market equilibrium
    """

    # Black-Litterman parameters
    tau: float = 0.05
    risk_aversion: float = 2.5
    risk_free_rate: float = 0.045  # 4.5% default

    # Country-specific risk
    use_country_risk_aversion: bool = True

    # View parameters
    view_confidence_scaling: float = 1.0
    min_view_confidence: float = 0.3
    max_view_weight: float = 0.20

    # Covariance estimation
    covariance_method: str = "ledoit_wolf"  # 'sample', 'ledoit_wolf', 'exponential'
    lookback_days: int = 252
    shrinkage_target: Optional[str] = None  # 'constant_variance', 'single_factor', None

    # Optimization constraints
    max_iterations: int = 1000
    tolerance: float = 1e-8
    regularization: float = 1e-6

    # Factor priors
    use_factor_priors: bool = False
    factor_weight: float = 0.3

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tau": self.tau,
            "risk_aversion": self.risk_aversion,
            "risk_free_rate": self.risk_free_rate,
            "use_country_risk_aversion": self.use_country_risk_aversion,
            "view_confidence_scaling": self.view_confidence_scaling,
            "min_view_confidence": self.min_view_confidence,
            "max_view_weight": self.max_view_weight,
            "covariance_method": self.covariance_method,
            "lookback_days": self.lookback_days,
            "shrinkage_target": self.shrinkage_target,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "regularization": self.regularization,
            "use_factor_priors": self.use_factor_priors,
            "factor_weight": self.factor_weight,
        }

    @classmethod
    def from_env(cls) -> "OptimizerConfig":
        """Create configuration from environment variables."""

        def get_float(name: str, default: float) -> float:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def get_int(name: str, default: int) -> int:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def get_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def get_str(name: str, default: str) -> str:
            return os.getenv(name, default)

        return cls(
            tau=get_float("OPTIMIZER_TAU", 0.05),
            risk_aversion=get_float("OPTIMIZER_RISK_AVERSION", 2.5),
            risk_free_rate=get_float("OPTIMIZER_RISK_FREE_RATE", 0.045),
            use_country_risk_aversion=get_bool("OPTIMIZER_COUNTRY_RISK", True),
            view_confidence_scaling=get_float("OPTIMIZER_VIEW_SCALING", 1.0),
            min_view_confidence=get_float("OPTIMIZER_MIN_VIEW_CONFIDENCE", 0.3),
            max_view_weight=get_float("OPTIMIZER_MAX_VIEW_WEIGHT", 0.20),
            covariance_method=get_str("OPTIMIZER_COVARIANCE_METHOD", "ledoit_wolf"),
            lookback_days=get_int("OPTIMIZER_LOOKBACK_DAYS", 252),
            shrinkage_target=os.getenv("OPTIMIZER_SHRINKAGE_TARGET"),
            max_iterations=get_int("OPTIMIZER_MAX_ITERATIONS", 1000),
            tolerance=get_float("OPTIMIZER_TOLERANCE", 1e-8),
            regularization=get_float("OPTIMIZER_REGULARIZATION", 1e-6),
            use_factor_priors=get_bool("OPTIMIZER_USE_FACTORS", False),
            factor_weight=get_float("OPTIMIZER_FACTOR_WEIGHT", 0.3),
        )

    @classmethod
    def for_regime(cls, regime: str) -> "OptimizerConfig":
        """
        Create regime-appropriate optimizer configuration.

        Args:
            regime: Macro regime (early_cycle, mid_cycle, late_cycle, recession)

        Returns:
            Regime-appropriate OptimizerConfig
        """
        if regime == "recession":
            # Higher risk aversion, more shrinkage in recession
            return cls(
                tau=0.03,  # Lower tau = less weight on views
                risk_aversion=4.0,  # Higher risk aversion
                risk_free_rate=0.045,
                view_confidence_scaling=0.8,  # Reduce view impact
                min_view_confidence=0.5,  # Only high confidence views
                covariance_method="ledoit_wolf",
                lookback_days=180,  # Shorter lookback (recent data matters more)
                use_factor_priors=True,  # Use defensive factors
                factor_weight=0.4,
            )
        elif regime == "late_cycle":
            # More cautious
            return cls(
                tau=0.04,
                risk_aversion=3.5,
                risk_free_rate=0.045,
                view_confidence_scaling=0.9,
                min_view_confidence=0.4,
                covariance_method="ledoit_wolf",
                lookback_days=200,
                use_factor_priors=True,
                factor_weight=0.35,
            )
        elif regime == "early_cycle":
            # More aggressive
            return cls(
                tau=0.08,  # Higher tau = more weight on views
                risk_aversion=2.0,  # Lower risk aversion
                risk_free_rate=0.045,
                view_confidence_scaling=1.2,  # Increase view impact
                min_view_confidence=0.25,  # More views included
                covariance_method="ledoit_wolf",
                lookback_days=252,
                use_factor_priors=False,
                factor_weight=0.2,
            )
        else:
            # Default (mid_cycle or uncertain)
            return cls()


@dataclass(frozen=True)
class RiskAversionByCountry:
    """
    Country-specific risk aversion coefficients.

    Different countries have different equity risk premiums and
    market characteristics that warrant different risk aversions.
    """

    # Default coefficients based on market characteristics
    usa: float = 2.5
    germany: float = 3.0
    uk: float = 2.8
    france: float = 3.0
    japan: float = 3.5
    default: float = 3.0

    def get(self, country: str) -> float:
        """Get risk aversion for a country."""
        mapping = {
            "USA": self.usa,
            "Germany": self.germany,
            "UK": self.uk,
            "France": self.france,
            "Japan": self.japan,
        }
        return mapping.get(country, self.default)

    @classmethod
    def for_regime(cls, regime: str) -> "RiskAversionByCountry":
        """Adjust all risk aversions based on regime."""
        if regime == "recession":
            # Higher risk aversion in recession
            multiplier = 1.4
        elif regime == "late_cycle":
            multiplier = 1.2
        elif regime == "early_cycle":
            multiplier = 0.85
        else:
            multiplier = 1.0

        return cls(
            usa=2.5 * multiplier,
            germany=3.0 * multiplier,
            uk=2.8 * multiplier,
            france=3.0 * multiplier,
            japan=3.5 * multiplier,
            default=3.0 * multiplier,
        )
