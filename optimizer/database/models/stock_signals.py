"""
Stock Signal Models - MATHEMATICAL SIGNAL VERSION (Quantitative Metrics Only)
=============================================================================
SQLAlchemy models for storing daily stock signals with complete quantitative metrics.

OPTIMIZATION CHANGES:
- Converted all risk level strings to RiskLevelEnum
- Removed all summary/notes Text fields
- Removed analyst_score, data_gaps, primary_risks (not used in mathematical signals)
- Added comprehensive quantitative metrics: Sharpe, Sortino, Alpha, Beta, etc.
- Result: Pure mathematical/quantitative signal storage

Signal Types:
- large_decline: Significant downward movement
- small_decline: Minor downward movement
- neutral: No significant movement
- small_gain: Minor upward movement
- large_gain: Significant upward movement
"""

import uuid
from datetime import date
from typing import Optional, TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String,
    Date,
    Float,
    ForeignKey,
    Index,
    UniqueConstraint,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
import enum

from optimizer.database.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from optimizer.database.models.universe import Instrument


class SignalEnum(str, enum.Enum):
    """Enum for daily stock signal types"""

    LARGE_DECLINE = "large_decline"
    SMALL_DECLINE = "small_decline"
    NEUTRAL = "neutral"
    SMALL_GAIN = "small_gain"
    LARGE_GAIN = "large_gain"


class ConfidenceLevelEnum(str, enum.Enum):
    """Enum for signal confidence levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevelEnum(str, enum.Enum):
    """Enum for risk level assessments (replaces string fields).

    Names are lowercase to match PostgreSQL enum values.
    """

    low = "low"
    medium = "medium"
    high = "high"
    unknown = "unknown"  # For cases where risk cannot be assessed


class StockSignal(Base, TimestampMixin):
    """
    Daily stock signal for an instrument - MATHEMATICAL VERSION.

    Contains:
    - All risk levels stored as enums
    - Complete quantitative metrics (Sharpe, Sortino, Alpha, Beta, IR, etc.)
    - Signal driver scores
    - No text summaries or LLM-generated content
    - Pure mathematical signal data for Chapter 2 framework
    """

    __tablename__ = "stock_signals"

    # Primary Key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="UUID primary key",
    )

    # Foreign Key to Instrument
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the instrument (stock)",
    )

    # Denormalized Instrument Data (from universe table - avoids joins)
    ticker: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Trading212 ticker symbol (denormalized from instruments.ticker)",
    )

    yfinance_ticker: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        comment="Yahoo Finance ticker (denormalized from instruments.yfinance_ticker)",
    )

    exchange_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Exchange name (denormalized from exchanges.exchange_name)",
    )

    # Company Information (from yfinance API)
    sector: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Company sector from yfinance (e.g., 'Technology', 'Healthcare')",
    )

    industry: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
        index=True,
        comment="Company industry from yfinance (e.g., 'Software', 'Pharmaceuticals')",
    )

    # Signal Data
    signal_date: Mapped[date] = mapped_column(
        Date, nullable=False, index=True, comment="Date when the signal was generated"
    )

    signal_type: Mapped[SignalEnum] = mapped_column(
        SQLEnum(SignalEnum, name="signal_type_enum", create_type=False),
        nullable=False,
        index=True,
        comment="Type of signal: large_decline, small_decline, neutral, small_gain, large_gain",
    )

    # Price Information
    close_price: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Closing price on signal date"
    )

    open_price: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Opening price on signal date"
    )

    daily_return: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Daily return percentage (close-to-close)"
    )

    volume: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Trading volume on signal date"
    )

    # Quantitative Metrics - Chapter 2 Framework
    annualized_return: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Annualized return over lookback period"
    )

    volatility: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Annualized volatility (standard deviation of returns)",
    )

    sharpe_ratio: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Sharpe ratio (risk-adjusted return)"
    )

    sortino_ratio: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Sortino ratio (downside risk-adjusted return)"
    )

    max_drawdown: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Maximum drawdown over lookback period"
    )

    calmar_ratio: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Calmar ratio (return / max drawdown)"
    )

    beta: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Market beta (sensitivity to benchmark)"
    )

    alpha: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Jensen's alpha (excess return vs. benchmark)"
    )

    r_squared: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="R² (coefficient of determination vs. benchmark)"
    )

    information_ratio: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Information ratio (alpha / tracking error)"
    )

    benchmark_return: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Benchmark annualized return over same period"
    )

    rsi: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Relative Strength Index (RSI) indicator"
    )

    # Metadata
    data_quality_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Data quality score (0-1) for signal reliability"
    )

    confidence_level: Mapped[Optional[ConfidenceLevelEnum]] = mapped_column(
        SQLEnum(ConfidenceLevelEnum, name="confidence_level_enum", create_type=False),
        nullable=True,
        index=True,
        comment="Confidence level in signal: low, medium, high",
    )

    # Signal Drivers - NUMERICAL SCORES ONLY (no summaries)
    valuation_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Valuation score: -1 (overvalued) to +1 (undervalued)",
    )

    momentum_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Momentum score: -1 (downtrend) to +1 (uptrend)"
    )

    quality_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Quality score: 0 (poor) to 1 (excellent)"
    )

    growth_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Growth score: -1 (contracting) to +1 (high growth)",
    )

    technical_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Technical score: -1 (bearish) to +1 (bullish)"
    )

    # Risk Factors - ALL ENUMS (no text fields)
    volatility_level: Mapped[Optional[RiskLevelEnum]] = mapped_column(
        SQLEnum(RiskLevelEnum, name="risk_level_enum", create_type=False),
        nullable=True,
        comment="Volatility assessment: low/medium/high",
    )

    beta_risk: Mapped[Optional[RiskLevelEnum]] = mapped_column(
        SQLEnum(RiskLevelEnum, name="risk_level_enum", create_type=False),
        nullable=True,
        comment="Market sensitivity assessment: low/medium/high",
    )

    debt_risk: Mapped[Optional[RiskLevelEnum]] = mapped_column(
        SQLEnum(RiskLevelEnum, name="risk_level_enum", create_type=False),
        nullable=True,
        comment="Debt level assessment: low/medium/high",
    )

    liquidity_risk: Mapped[Optional[RiskLevelEnum]] = mapped_column(
        SQLEnum(RiskLevelEnum, name="risk_level_enum", create_type=False),
        nullable=True,
        comment="Trading liquidity assessment: low/medium/high",
    )

    # Price Targets
    upside_potential_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Estimated upside % to fair value or target"
    )

    downside_risk_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Estimated downside % risk to support levels"
    )

    # Relationships
    instrument: Mapped["Instrument"] = relationship(
        "Instrument", back_populates="signals", lazy="selectin"
    )

    # Indexes and Constraints
    __table_args__ = (
        # Unique constraint: one signal per instrument per day
        UniqueConstraint(
            "instrument_id", "signal_date", name="uq_stock_signals_instrument_date"
        ),
        # Composite indexes for common queries
        Index("idx_stock_signals_instrument_date", "instrument_id", "signal_date"),
        Index("idx_stock_signals_date", "signal_date"),
        Index("idx_stock_signals_type", "signal_type"),
        Index("idx_stock_signals_instrument_type", "instrument_id", "signal_type"),
        # Performance index for time-series queries
        Index("idx_stock_signals_date_type", "signal_date", "signal_type"),
        # Indexes for denormalized fields (avoids joins)
        Index("idx_stock_signals_ticker", "ticker"),
        Index("idx_stock_signals_yfinance_ticker", "yfinance_ticker"),
        Index("idx_stock_signals_exchange", "exchange_name"),
        Index("idx_stock_signals_sector", "sector"),
        Index("idx_stock_signals_industry", "industry"),
        # Composite indexes for sector/industry analysis
        Index("idx_stock_signals_date_sector", "signal_date", "sector"),
        Index(
            "idx_stock_signals_date_sector_type", "signal_date", "sector", "signal_type"
        ),
        Index("idx_stock_signals_exchange_date", "exchange_name", "signal_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<StockSignal(id={self.id}, "
            f"instrument={self.instrument.ticker if self.instrument else None}, "
            f"date={self.signal_date}, "
            f"signal={self.signal_type.value})>"
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the signal (mathematical version)."""
        return {
            "id": str(self.id),
            "instrument_id": str(self.instrument_id),
            # Denormalized instrument data (fast access, no joins)
            "ticker": self.ticker,
            "yfinance_ticker": self.yfinance_ticker,
            "exchange_name": self.exchange_name,
            "sector": self.sector,
            "industry": self.industry,
            # Signal data
            "signal_date": self.signal_date.isoformat(),
            "signal_type": self.signal_type.value,
            # Price data
            "close_price": self.close_price,
            "open_price": self.open_price,
            "daily_return": self.daily_return,
            "volume": self.volume,
            # Quantitative metrics
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
            # Metadata
            "data_quality_score": self.data_quality_score,
            "confidence_level": (
                self.confidence_level.value if self.confidence_level else None
            ),
            # Signal drivers (numerical only)
            "signal_drivers": {
                "valuation_score": self.valuation_score,
                "momentum_score": self.momentum_score,
                "quality_score": self.quality_score,
                "growth_score": self.growth_score,
                "technical_score": self.technical_score,
            },
            # Risk factors (enums only)
            "risk_factors": {
                "volatility_level": (
                    self.volatility_level.value if self.volatility_level else None
                ),
                "beta_risk": self.beta_risk.value if self.beta_risk else None,
                "debt_risk": self.debt_risk.value if self.debt_risk else None,
                "liquidity_risk": (
                    self.liquidity_risk.value if self.liquidity_risk else None
                ),
            },
            # Price targets
            "upside_potential_pct": self.upside_potential_pct,
            "downside_risk_pct": self.downside_risk_pct,
            "created_at": self.created_at.isoformat(),
        }
