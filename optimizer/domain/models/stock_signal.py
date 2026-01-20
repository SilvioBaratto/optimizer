"""
Stock Signal Domain Model - Pure Python representation of stock signals.

This DTO represents a daily stock signal with all quantitative metrics,
independent of database implementation.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID


class SignalType(str, Enum):
    """Signal type classification."""
    LARGE_DECLINE = "large_decline"
    SMALL_DECLINE = "small_decline"
    NEUTRAL = "neutral"
    SMALL_GAIN = "small_gain"
    LARGE_GAIN = "large_gain"


class ConfidenceLevel(str, Enum):
    """Signal confidence level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    """Risk level assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StockSignalDTO:
    """
    Data Transfer Object for stock signals.

    Immutable representation of a daily stock signal with all metrics.
    Used for passing signal data between layers without database dependencies.

    Attributes:
        id: Unique signal identifier
        instrument_id: Reference to instrument (stock)
        ticker: Trading212 ticker symbol (e.g., 'AAPL_US_EQ')
        yfinance_ticker: Yahoo Finance ticker (e.g., 'AAPL')
        signal_date: Date when signal was generated
        signal_type: Signal classification (large_gain, etc.)

        # Company info
        sector: GICS sector (e.g., 'Technology')
        industry: GICS industry (e.g., 'Software')
        exchange_name: Exchange name (e.g., 'NASDAQ')

        # Price data
        close_price: Closing price on signal date
        open_price: Opening price on signal date
        daily_return: Daily return percentage
        volume: Trading volume

        # Risk-adjusted metrics
        annualized_return: Annualized return over lookback period
        volatility: Annualized volatility
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio (downside risk adjusted)
        max_drawdown: Maximum drawdown (negative number)
        calmar_ratio: Return / max drawdown

        # Market metrics
        beta: Market beta
        alpha: Jensen's alpha
        r_squared: R-squared vs benchmark
        information_ratio: Information ratio

        # Signal scores
        valuation_score: -1 (overvalued) to +1 (undervalued)
        momentum_score: -1 (downtrend) to +1 (uptrend)
        quality_score: 0 (poor) to 1 (excellent)
        growth_score: -1 (contracting) to +1 (high growth)
        technical_score: -1 (bearish) to +1 (bullish)

        # Risk factors
        volatility_level: Risk level assessment
        beta_risk: Market sensitivity risk
        debt_risk: Debt level risk
        liquidity_risk: Trading liquidity risk

        # Metadata
        confidence_level: Signal confidence
        data_quality_score: Data quality (0-1)
        upside_potential_pct: Estimated upside percentage
        downside_risk_pct: Estimated downside percentage
    """

    # Required fields
    id: UUID
    instrument_id: UUID
    ticker: str
    signal_date: date
    signal_type: SignalType

    # Optional identifiers
    yfinance_ticker: Optional[str] = None
    exchange_name: Optional[str] = None

    # Company info
    sector: Optional[str] = None
    industry: Optional[str] = None

    # Price data
    close_price: Optional[float] = None
    open_price: Optional[float] = None
    daily_return: Optional[float] = None
    volume: Optional[float] = None

    # Risk-adjusted metrics
    annualized_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Market metrics
    beta: Optional[float] = None
    alpha: Optional[float] = None
    r_squared: Optional[float] = None
    information_ratio: Optional[float] = None
    benchmark_return: Optional[float] = None
    rsi: Optional[float] = None

    # Signal scores
    valuation_score: Optional[float] = None
    momentum_score: Optional[float] = None
    quality_score: Optional[float] = None
    growth_score: Optional[float] = None
    technical_score: Optional[float] = None

    # Risk factors
    volatility_level: Optional[RiskLevel] = None
    beta_risk: Optional[RiskLevel] = None
    debt_risk: Optional[RiskLevel] = None
    liquidity_risk: Optional[RiskLevel] = None

    # Metadata
    confidence_level: Optional[ConfidenceLevel] = None
    data_quality_score: Optional[float] = None
    upside_potential_pct: Optional[float] = None
    downside_risk_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "instrument_id": str(self.instrument_id),
            "ticker": self.ticker,
            "yfinance_ticker": self.yfinance_ticker,
            "exchange_name": self.exchange_name,
            "sector": self.sector,
            "industry": self.industry,
            "signal_date": self.signal_date.isoformat(),
            "signal_type": self.signal_type.value,
            "close_price": self.close_price,
            "open_price": self.open_price,
            "daily_return": self.daily_return,
            "volume": self.volume,
            "quantitative_metrics": {
                "annualized_return": self.annualized_return,
                "volatility": self.volatility,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "calmar_ratio": self.calmar_ratio,
                "beta": self.beta,
                "alpha": self.alpha,
                "r_squared": self.r_squared,
                "information_ratio": self.information_ratio,
                "benchmark_return": self.benchmark_return,
                "rsi": self.rsi,
            },
            "signal_drivers": {
                "valuation_score": self.valuation_score,
                "momentum_score": self.momentum_score,
                "quality_score": self.quality_score,
                "growth_score": self.growth_score,
                "technical_score": self.technical_score,
            },
            "risk_factors": {
                "volatility_level": self.volatility_level.value if self.volatility_level else None,
                "beta_risk": self.beta_risk.value if self.beta_risk else None,
                "debt_risk": self.debt_risk.value if self.debt_risk else None,
                "liquidity_risk": self.liquidity_risk.value if self.liquidity_risk else None,
            },
            "confidence_level": self.confidence_level.value if self.confidence_level else None,
            "data_quality_score": self.data_quality_score,
            "upside_potential_pct": self.upside_potential_pct,
            "downside_risk_pct": self.downside_risk_pct,
        }

    @property
    def is_large_gain(self) -> bool:
        """Check if this is a LARGE_GAIN signal."""
        return self.signal_type == SignalType.LARGE_GAIN

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish (LARGE_GAIN or SMALL_GAIN)."""
        return self.signal_type in (SignalType.LARGE_GAIN, SignalType.SMALL_GAIN)

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish (LARGE_DECLINE or SMALL_DECLINE)."""
        return self.signal_type in (SignalType.LARGE_DECLINE, SignalType.SMALL_DECLINE)

    @property
    def has_valid_metrics(self) -> bool:
        """Check if signal has valid core metrics for filtering."""
        return (
            self.sharpe_ratio is not None
            and self.volatility is not None
            and self.close_price is not None
        )

    @property
    def daily_dollar_volume(self) -> Optional[float]:
        """Calculate daily dollar volume."""
        if self.volume is not None and self.close_price is not None:
            return self.volume * self.close_price
        return None

    def with_updated_weight(self, weight: float) -> "StockSignalDTO":
        """
        Create a new signal with updated values (for immutability).

        Note: This is a workaround since we can't modify frozen dataclass.
        """
        # This method is a placeholder - actual implementation would use
        # dataclasses.replace() if we had a weight field
        return self
