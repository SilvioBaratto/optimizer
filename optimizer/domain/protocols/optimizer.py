"""
Optimizer Protocols - Portfolio optimization interfaces.

Defines contracts for portfolio optimization components, enabling:
- Pluggable covariance estimators (Ledoit-Wolf, sample, etc.)
- Pluggable optimization strategies (mean-variance, risk parity, etc.)
- Testable optimization pipelines

Design Principles:
- Single Responsibility: Separate covariance, equilibrium, and optimization
- Open/Closed: New estimators/optimizers can be added without modification
- Dependency Inversion: Business logic depends on abstractions, not implementations
"""

from typing import Protocol, List, Dict, Optional, Any, Tuple
from datetime import date


class CovarianceEstimator(Protocol):
    """
    Protocol for covariance matrix estimation.

    Implementations should handle:
    - Missing data (forward fill, drop, or impute)
    - Non-trading days alignment
    - Numerical stability (condition number)

    Example implementations:
    - LedoitWolfEstimator: Shrinkage-based robust estimation
    - SampleCovarianceEstimator: Simple sample covariance
    - ExponentialWeightedEstimator: Time-weighted covariance
    """

    @property
    def name(self) -> str:
        """Human-readable estimator name."""
        ...

    def estimate(
        self,
        prices: Any,  # pd.DataFrame of prices (dates x tickers)
        frequency: int = 252,  # Annualization factor
    ) -> Any:  # Returns pd.DataFrame (covariance matrix)
        """
        Estimate the covariance matrix from price data.

        Args:
            prices: DataFrame with dates as index, tickers as columns
            frequency: Annualization factor (252 for daily data)

        Returns:
            Annualized covariance matrix as DataFrame
        """
        ...

    def get_shrinkage_factor(self) -> Optional[float]:
        """
        Get the shrinkage factor used (if applicable).

        Returns:
            Shrinkage factor between 0 and 1, or None for non-shrinkage methods
        """
        ...


class RiskAversionEstimator(Protocol):
    """
    Protocol for market risk aversion estimation.

    Risk aversion (delta) is a key parameter in Black-Litterman that controls
    the balance between market equilibrium and investor views.
    """

    def estimate(
        self,
        market_returns: Any,  # pd.Series of market returns
        risk_free_rate: float,
    ) -> float:
        """
        Estimate the implied market risk aversion.

        Args:
            market_returns: Historical market returns (e.g., S&P 500)
            risk_free_rate: Risk-free rate (annualized, decimal)

        Returns:
            Risk aversion coefficient (typically 1.5 to 5.0)
        """
        ...

    def adjust_for_regime(
        self,
        base_delta: float,
        regime: str,
        recession_risk: Optional[float] = None,
    ) -> float:
        """
        Adjust risk aversion based on macro regime.

        Args:
            base_delta: Base risk aversion estimate
            regime: Macro regime (early_cycle, mid_cycle, late_cycle, recession)
            recession_risk: Optional recession probability (0-1)

        Returns:
            Adjusted risk aversion coefficient
        """
        ...


class EquilibriumCalculator(Protocol):
    """
    Protocol for calculating market equilibrium returns.

    The equilibrium return (pi) represents the market-implied expected return
    for each asset, derived from market capitalizations.
    """

    def calculate(
        self,
        market_caps: Any,  # pd.Series of market caps
        covariance_matrix: Any,  # pd.DataFrame
        risk_aversion: float,
        risk_free_rate: float,
    ) -> Any:  # Returns pd.Series of equilibrium returns
        """
        Calculate market-implied equilibrium returns.

        Formula: pi = delta * Sigma * w_mkt

        Args:
            market_caps: Market capitalizations by ticker
            covariance_matrix: Covariance matrix (annualized)
            risk_aversion: Risk aversion coefficient (delta)
            risk_free_rate: Risk-free rate (annualized, decimal)

        Returns:
            Series of equilibrium expected returns by ticker
        """
        ...


class ViewGenerator(Protocol):
    """
    Protocol for generating Black-Litterman views.

    Views are investor expectations about asset returns, combined with
    the equilibrium to produce posterior expected returns.
    """

    def generate(
        self,
        signals: List[Any],  # List[StockSignalDTO]
        macro_regime: Optional[str] = None,
    ) -> List[Any]:  # Returns List[BlackLittermanViewDTO]
        """
        Generate views from stock signals.

        Args:
            signals: Stock signals to generate views for
            macro_regime: Optional macro regime for context

        Returns:
            List of BlackLittermanViewDTO objects
        """
        ...

    def construct_matrices(
        self,
        views: List[Any],  # List[BlackLittermanViewDTO]
        universe_tickers: List[str],
    ) -> Tuple[Any, Any, Any]:  # Returns (P, Q, Omega) numpy arrays
        """
        Construct P, Q, Omega matrices from views.

        Args:
            views: Generated views
            universe_tickers: Complete list of tickers in universe

        Returns:
            Tuple of (P, Q, Omega):
            - P: KxN picking matrix
            - Q: Kx1 expected returns vector
            - Omega: KxK view uncertainty matrix
        """
        ...


class PortfolioOptimizer(Protocol):
    """
    Protocol for portfolio weight optimization.

    This is the main orchestration interface that combines all components
    to produce optimized portfolio weights.
    """

    def optimize(
        self,
        positions: List[Any],  # List[PositionDTO]
        signal_date: Optional[date] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:  # Returns (optimized_positions, metrics)
        """
        Optimize portfolio weights using Black-Litterman.

        Args:
            positions: Initial positions to optimize
            signal_date: Signal date (defaults to most recent)

        Returns:
            Tuple of (optimized_positions, metrics):
            - optimized_positions: List of PositionDTO with optimized weights
            - metrics: Dictionary of optimization metrics and diagnostics
        """
        ...

    def get_posterior_returns(self) -> Optional[Any]:  # Returns Optional[pd.Series]
        """
        Get the posterior expected returns from last optimization.

        Returns:
            Series of posterior returns, or None if not yet optimized
        """
        ...

    def get_equilibrium_returns(self) -> Optional[Any]:  # Returns Optional[pd.Series]
        """
        Get the equilibrium returns from last optimization.

        Returns:
            Series of equilibrium returns, or None if not yet optimized
        """
        ...


class ConstrainedOptimizer(Protocol):
    """
    Protocol for constrained weight optimization.

    Handles the quadratic programming optimization with constraints:
    - Sector limits
    - Position limits
    - Full investment constraint
    """

    def optimize(
        self,
        posterior_returns: Any,  # pd.Series
        covariance_matrix: Any,  # pd.DataFrame
        risk_aversion: float,
        sector_mapping: Dict[str, List[int]],  # sector -> position indices
        max_sector_weight: float = 0.15,
        max_position_weight: float = 0.10,
        min_position_weight: float = 0.0,
    ) -> Any:  # Returns pd.Series of optimized weights
        """
        Run constrained optimization.

        Objective: minimize (1/2) w^T (delta*Sigma) w - mu^T w
        Subject to:
        - Sum(w) = 1 (fully invested)
        - w_i >= min_weight for all i
        - w_i <= max_weight for all i
        - Sum(w_i) <= max_sector for each sector

        Args:
            posterior_returns: Expected returns (mu)
            covariance_matrix: Covariance matrix (Sigma)
            risk_aversion: Risk aversion (delta)
            sector_mapping: Mapping of sectors to position indices
            max_sector_weight: Maximum weight per sector
            max_position_weight: Maximum weight per position
            min_position_weight: Minimum weight per position

        Returns:
            Series of optimized weights by ticker
        """
        ...

    def verify_constraints(
        self,
        weights: Any,  # pd.Series
        sector_mapping: Dict[str, List[int]],
        max_sector_weight: float,
        max_position_weight: float,
        min_position_weight: float,
    ) -> Dict[str, Any]:
        """
        Verify that weights satisfy all constraints.

        Args:
            weights: Optimized weights to verify
            sector_mapping: Mapping of sectors to position indices
            max_sector_weight: Maximum weight per sector
            max_position_weight: Maximum weight per position
            min_position_weight: Minimum weight per position

        Returns:
            Dictionary with verification results:
            - all_satisfied: bool
            - total_weight: float
            - sector_weights: Dict[str, float]
            - violations: List[str]
        """
        ...
