"""
Portfolio Domain Model - Pure Python representation of portfolios and positions.

Portfolios are collections of positions with weights, created by the
optimization pipeline.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from uuid import UUID


@dataclass
class PositionDTO:
    """
    Data Transfer Object for portfolio positions.

    Represents a single holding within a portfolio with weight and metrics.

    Note: Not frozen because weights are updated during optimization.
    """

    # Required fields
    ticker: str
    weight: float  # Portfolio weight (0.0 to 1.0)

    # Optional identifiers
    instrument_id: Optional[str] = None
    signal_id: Optional[str] = None
    yfinance_ticker: Optional[str] = None

    # Company info
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: str = "USA"
    exchange: Optional[str] = None

    # Price data
    price: Optional[float] = None

    # Risk metrics (from signal)
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    volatility: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    max_drawdown: Optional[float] = None
    annualized_return: Optional[float] = None

    # Signal info
    signal_type: Optional[str] = None
    confidence_level: str = "medium"
    conviction_tier: int = 2  # 1=highest, 2=medium, 3=lowest
    data_quality_score: Optional[float] = None

    # Optimization metadata
    original_weight: Optional[float] = None  # Weight before BL optimization
    equilibrium_return: Optional[float] = None  # π from Black-Litterman
    posterior_return: Optional[float] = None  # E[R] from Black-Litterman
    selection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "ticker": self.ticker,
            "weight": self.weight,
            "instrument_id": self.instrument_id,
            "signal_id": self.signal_id,
            "yfinance_ticker": self.yfinance_ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "country": self.country,
            "exchange": self.exchange,
            "price": self.price,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "volatility": self.volatility,
            "alpha": self.alpha,
            "beta": self.beta,
            "max_drawdown": self.max_drawdown,
            "annualized_return": self.annualized_return,
            "signal_type": self.signal_type,
            "confidence_level": self.confidence_level,
            "conviction_tier": self.conviction_tier,
            "data_quality_score": self.data_quality_score,
            "original_weight": self.original_weight,
            "equilibrium_return": self.equilibrium_return,
            "posterior_return": self.posterior_return,
            "selection_reason": self.selection_reason,
        }

    @property
    def weight_change(self) -> Optional[float]:
        """Calculate weight change from optimization."""
        if self.original_weight is not None:
            return self.weight - self.original_weight
        return None

    @property
    def weight_pct(self) -> str:
        """Format weight as percentage string."""
        return f"{self.weight * 100:.2f}%"

    def with_weight(self, new_weight: float) -> "PositionDTO":
        """Create a new position with updated weight."""
        import copy
        new_pos = copy.copy(self)
        new_pos.weight = new_weight
        return new_pos


@dataclass
class PortfolioDTO:
    """
    Data Transfer Object for portfolios.

    Represents a complete portfolio with positions, metadata, and metrics.

    Note: Not frozen because positions can be updated.
    """

    # Required fields
    id: UUID
    portfolio_date: date
    positions: List[PositionDTO]

    # Portfolio metadata
    name: Optional[str] = None
    optimization_method: str = "black_litterman"
    used_baml_views: bool = True
    used_factor_priors: bool = False

    # Portfolio metrics
    total_positions: int = 0
    total_weight: float = 1.0
    risk_aversion: Optional[float] = None
    tau: Optional[float] = None
    regime: Optional[str] = None

    # Risk-free rate info
    risk_free_rate: Optional[float] = None
    risk_free_rate_by_country: Optional[Dict[str, float]] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Additional metrics (stored as JSON in DB)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and compute derived fields."""
        if self.total_positions == 0:
            self.total_positions = len(self.positions)
        if self.total_weight != sum(p.weight for p in self.positions):
            self.total_weight = sum(p.weight for p in self.positions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "portfolio_date": self.portfolio_date.isoformat(),
            "name": self.name,
            "optimization_method": self.optimization_method,
            "used_baml_views": self.used_baml_views,
            "used_factor_priors": self.used_factor_priors,
            "total_positions": self.total_positions,
            "total_weight": self.total_weight,
            "risk_aversion": self.risk_aversion,
            "tau": self.tau,
            "regime": self.regime,
            "risk_free_rate": self.risk_free_rate,
            "risk_free_rate_by_country": self.risk_free_rate_by_country,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metrics": self.metrics,
            "positions": [p.to_dict() for p in self.positions],
        }

    @property
    def sector_weights(self) -> Dict[str, float]:
        """Calculate weights by sector."""
        weights: Dict[str, float] = {}
        for pos in self.positions:
            sector = pos.sector or "Unknown"
            weights[sector] = weights.get(sector, 0.0) + pos.weight
        return weights

    @property
    def country_weights(self) -> Dict[str, float]:
        """Calculate weights by country."""
        weights: Dict[str, float] = {}
        for pos in self.positions:
            country = pos.country or "Unknown"
            weights[country] = weights.get(country, 0.0) + pos.weight
        return weights

    @property
    def tickers(self) -> List[str]:
        """Get list of all tickers in portfolio."""
        return [p.ticker for p in self.positions]

    def get_position(self, ticker: str) -> Optional[PositionDTO]:
        """Get position by ticker."""
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None

    def validate_weights(self) -> Dict[str, Any]:
        """
        Validate portfolio weights.

        Returns:
            Dictionary with validation results:
            - is_valid: bool
            - total_weight: float
            - issues: List[str]
        """
        issues = []
        total = sum(p.weight for p in self.positions)

        # Check total weight
        if abs(total - 1.0) > 0.01:
            issues.append(f"Total weight {total:.4f} != 1.0")

        # Check for negative weights
        negative = [p.ticker for p in self.positions if p.weight < 0]
        if negative:
            issues.append(f"Negative weights: {negative}")

        # Check for excessive weights
        excessive = [p.ticker for p in self.positions if p.weight > 0.15]
        if excessive:
            issues.append(f"Excessive weights (>15%): {excessive}")

        return {
            "is_valid": len(issues) == 0,
            "total_weight": total,
            "issues": issues,
        }
