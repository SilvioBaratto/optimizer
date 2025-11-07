"""
Macro Regime Models - Simplified Business Cycle Analysis Schema
=================================================================
SQLAlchemy models for storing macro regime analysis data.

DESIGN PRINCIPLES:
- Align with BusinessCycleClassification BAML output
- Follow macroeconomic framework from portfolio_guideline
- Eliminate redundancy and over-normalization
- Keep only actionable portfolio positioning data

Tables:
- MacroAnalysisRun: Top-level analysis run metadata
- MarketIndicators: Global market indicators (FRED data)
- EconomicIndicators: Country-specific indicators (Il Sole 24 Ore)
- CountryRegimeAssessment: Regime classification and portfolio positioning
- NewsArticle: News articles used in analysis (optional)
- RegimeTransition: Historical regime changes
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum
from typing import Optional, TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, Float, DateTime, Text,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import Enum as SQLEnum

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.news import NewsArticle
# ============================================================================
# ENUMS - Aligned with BAML types
# ============================================================================

class RegimeEnum(str, Enum):
    """Business cycle regime (from BAML BusinessCycleRegime)"""
    EARLY_CYCLE = "EARLY_CYCLE"
    MID_CYCLE = "MID_CYCLE"
    LATE_CYCLE = "LATE_CYCLE"
    RECESSION = "RECESSION"
    UNCERTAIN = "UNCERTAIN"


class ISMSignalEnum(str, Enum):
    """ISM PMI signal classification (from BAML)"""
    STRONG_EXPANSION = "STRONG_EXPANSION"  # >52
    MILD_EXPANSION = "MILD_EXPANSION"      # 50-52
    MILD_CONTRACTION = "MILD_CONTRACTION"  # 45-50
    DEEP_CONTRACTION = "DEEP_CONTRACTION"  # <45


class YieldCurveSignalEnum(str, Enum):
    """Yield curve signal classification (from BAML)"""
    STEEP = "STEEP"        # >100bps
    NORMAL = "NORMAL"      # 50-100bps
    FLAT = "FLAT"          # 0-50bps
    INVERTED = "INVERTED"  # <0bps


class CreditSpreadSignalEnum(str, Enum):
    """Credit spread signal classification (from BAML)"""
    TIGHT = "TIGHT"        # <350bps
    NEUTRAL = "NEUTRAL"    # 350-500bps
    WIDENING = "WIDENING"  # 500-700bps
    STRESS = "STRESS"      # >700bps


class FactorExposureEnum(str, Enum):
    """Factor exposure recommendation (from BAML)"""
    GROWTH_MOMENTUM = "GROWTH_MOMENTUM"
    QUALITY_DEFENSIVE = "QUALITY_DEFENSIVE"
    BALANCED = "BALANCED"


class VIXSignalEnum(str, Enum):
    """VIX volatility signal"""
    LOW = "LOW"      # <15
    MEDIUM = "MEDIUM"  # 15-25
    HIGH = "HIGH"      # 25-35
    EXTREME = "EXTREME"  # >35


# ============================================================================
# MODELS
# ============================================================================

class MacroAnalysisRun(BaseModel):
    """
    Top-level container for a macro regime analysis execution.
    Represents a single run across one or more countries.
    """

    __tablename__ = "macro_analysis_runs"

    run_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When the analysis was executed"
    )

    num_countries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of countries analyzed"
    )

    countries: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of country codes (e.g., ['USA', 'Germany'])"
    )

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Optional notes"
    )

    # Relationships
    market_indicators: Mapped["MarketIndicators"] = relationship(
        "MarketIndicators",
        back_populates="analysis_run",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="joined"
    )

    country_assessments: Mapped[List["CountryRegimeAssessment"]] = relationship(
        "CountryRegimeAssessment",
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        lazy="noload"
    )

    __table_args__ = (
        Index('idx_run_timestamp', 'run_timestamp'),
        Index('idx_run_countries_gin', 'countries', postgresql_using='gin'),
        CheckConstraint('num_countries > 0', name='ck_num_countries_positive'),
        {
            'comment': 'Top-level container for macro regime analysis run',
        }
    )

    def __repr__(self) -> str:
        return f"<MacroAnalysisRun(id={self.id}, timestamp='{self.run_timestamp}')>"


class MarketIndicators(BaseModel):
    """
    Global market indicators from FRED (guide.md pages 83-87).

    Essential indicators:
    - ISM Manufacturing PMI (real-time economic indicator)
    - Yield Curve 2s10s (recession forecasting)
    - HY Credit Spreads (risk appetite)
    - VIX (market volatility)
    """

    __tablename__ = "market_indicators"

    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("macro_analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to parent analysis run"
    )

    data_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When data was fetched"
    )

    # Core FRED indicators
    ism_pmi: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="ISM Manufacturing PMI"
    )

    yield_curve_2s10s: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="2s10s yield spread (bps)"
    )

    hy_spread: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="High-yield credit spread (bps)"
    )

    vix: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="VIX volatility index"
    )

    # Relationships
    analysis_run: Mapped["MacroAnalysisRun"] = relationship(
        "MacroAnalysisRun",
        back_populates="market_indicators"
    )

    __table_args__ = (
        Index('idx_market_timestamp', 'data_timestamp'),
        Index('idx_market_run_id', 'analysis_run_id'),
        UniqueConstraint('analysis_run_id', name='uq_market_per_run'),
        CheckConstraint('ism_pmi IS NULL OR ism_pmi >= 0', name='ck_ism_positive'),
        CheckConstraint('vix IS NULL OR vix >= 0', name='ck_vix_positive'),
        {
            'comment': 'Global market indicators from FRED',
        }
    )

    def __repr__(self) -> str:
        return f"<MarketIndicators(ISM={self.ism_pmi}, VIX={self.vix})>"


class EconomicIndicators(BaseModel):
    """
    Country-specific economic indicators from Il Sole 24 Ore.

    Includes current readings and forecasts for:
    - GDP growth (quarterly and annual)
    - Unemployment
    - Inflation
    - Industrial production
    - Earnings forecasts
    """

    __tablename__ = "economic_indicators"

    country: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Country code"
    )

    data_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When data was fetched"
    )

    # Current indicators
    gdp_growth_qq: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="GDP growth QQ (%)"
    )

    gdp_growth_yy: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="GDP growth YY (%)"
    )

    unemployment: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Unemployment rate (%)"
    )

    inflation: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Inflation rate (%)"
    )

    industrial_production: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Industrial production growth (%)"
    )

    # Forecasts
    gdp_forecast_6m: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="GDP forecast 6M (%)"
    )

    inflation_forecast_6m: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Inflation forecast 6M (%)"
    )

    earnings_forecast_12m: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Earnings forecast 12M (%)"
    )

    # Relationships
    assessments: Mapped[List["CountryRegimeAssessment"]] = relationship(
        "CountryRegimeAssessment",
        back_populates="economic_indicators"
    )

    __table_args__ = (
        Index('idx_econ_country', 'country'),
        Index('idx_econ_timestamp', 'data_timestamp'),
        Index('idx_econ_country_timestamp', 'country', 'data_timestamp'),
        UniqueConstraint('country', 'data_timestamp', name='uq_country_timestamp'),
        CheckConstraint('unemployment IS NULL OR unemployment >= 0', name='ck_unemployment_positive'),
        {
            'comment': 'Country-specific economic indicators from Il Sole 24 Ore',
        }
    )

    def __repr__(self) -> str:
        return f"<EconomicIndicators(country='{self.country}', GDP_qq={self.gdp_growth_qq})>"


class CountryRegimeAssessment(BaseModel):
    """
    Business cycle regime assessment for a country.

    Stores the complete BAML BusinessCycleClassification output:
    - Regime classification with confidence
    - Market indicator signals (ISM, yield curve, credit spreads)
    - Recession risk (6M and 12M)
    - Portfolio positioning (sector tilts, factor exposure)
    - Risk monitoring (primary risks, conflicting signals)
    """

    __tablename__ = "country_regime_assessments"

    # Foreign Keys
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("macro_analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to analysis run"
    )

    economic_indicators_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("economic_indicators.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Reference to economic indicators"
    )

    # Country and timestamp
    country: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Country code"
    )

    assessment_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When assessment was made"
    )

    # ========================================================================
    # BAML BusinessCycleClassification fields
    # ========================================================================

    # Core classification
    regime: Mapped[str] = mapped_column(
        SQLEnum(RegimeEnum, native_enum=False, length=30),
        nullable=False,
        index=True,
        comment="Business cycle regime"
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Classification confidence (0.0-1.0)"
    )

    rationale: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Rationale for classification (2-3 sentences)"
    )

    # Key indicator signals
    ism_signal: Mapped[str] = mapped_column(
        SQLEnum(ISMSignalEnum, native_enum=False, length=30),
        nullable=False,
        comment="ISM PMI signal"
    )

    yield_curve_signal: Mapped[str] = mapped_column(
        SQLEnum(YieldCurveSignalEnum, native_enum=False, length=30),
        nullable=False,
        comment="Yield curve signal"
    )

    credit_spread_signal: Mapped[str] = mapped_column(
        SQLEnum(CreditSpreadSignalEnum, native_enum=False, length=30),
        nullable=False,
        comment="Credit spread signal"
    )

    # Recession risk
    recession_risk_6m: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="6-month recession risk (0.0-1.0)"
    )

    recession_risk_12m: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="12-month recession risk (0.0-1.0)"
    )

    recession_drivers: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        comment="2-4 key recession risk factors"
    )

    # Portfolio positioning
    sector_tilts: Mapped[Dict[str, float]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Sector weight adjustments (-0.10 to +0.10)"
    )

    recommended_overweights: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        comment="3-5 sectors to overweight with rationale"
    )

    recommended_underweights: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        comment="3-5 sectors to underweight with rationale"
    )

    factor_exposure: Mapped[str] = mapped_column(
        SQLEnum(FactorExposureEnum, native_enum=False, length=30),
        nullable=False,
        comment="Factor exposure recommendation"
    )

    # Risk monitoring
    primary_risks: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        comment="2-3 main risks to monitor"
    )

    conflicting_signals: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Major contradictions (empty if none)"
    )

    # Relationships
    analysis_run: Mapped["MacroAnalysisRun"] = relationship(
        "MacroAnalysisRun",
        back_populates="country_assessments"
    )

    economic_indicators: Mapped["EconomicIndicators"] = relationship(
        "EconomicIndicators",
        back_populates="assessments",
        lazy="joined"
    )

    news_articles: Mapped[List["NewsArticle"]] = relationship(
        "NewsArticle",
        back_populates="assessment",
        cascade="all, delete-orphan",
        lazy="noload"
    )

    regime_transitions: Mapped[List["RegimeTransition"]] = relationship(
        "RegimeTransition",
        back_populates="assessment",
        cascade="all, delete-orphan",
        lazy="noload"
    )

    __table_args__ = (
        Index('idx_assessment_country', 'country'),
        Index('idx_assessment_regime', 'regime'),
        Index('idx_assessment_timestamp', 'assessment_timestamp'),
        Index('idx_assessment_country_regime', 'country', 'regime'),
        Index('idx_assessment_run_country', 'analysis_run_id', 'country'),
        Index('idx_assessment_sector_tilts_gin', 'sector_tilts', postgresql_using='gin'),
        Index('idx_assessment_risks_gin', 'primary_risks', postgresql_using='gin'),
        UniqueConstraint('country', 'assessment_timestamp', name='uq_country_assessment_ts'),
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='ck_confidence_range'),
        CheckConstraint('recession_risk_6m >= 0.0 AND recession_risk_6m <= 1.0', name='ck_recession_6m_range'),
        CheckConstraint('recession_risk_12m >= 0.0 AND recession_risk_12m <= 1.0', name='ck_recession_12m_range'),
        {
            'comment': 'Business cycle regime assessment with portfolio positioning recommendations',
        }
    )

    def __repr__(self) -> str:
        return f"<CountryRegimeAssessment(country='{self.country}', regime='{self.regime}', confidence={self.confidence:.2f})>"


class RegimeTransition(BaseModel):
    """
    Historical regime transitions for tracking changes over time.
    Used for monitoring regime shifts and generating alerts.
    """

    __tablename__ = "regime_transitions"

    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("country_regime_assessments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to assessment where transition occurred"
    )

    country: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Country code"
    )

    transition_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When transition was detected"
    )

    from_regime: Mapped[str] = mapped_column(
        SQLEnum(RegimeEnum, native_enum=False, length=30),
        nullable=False,
        comment="Previous regime"
    )

    to_regime: Mapped[str] = mapped_column(
        SQLEnum(RegimeEnum, native_enum=False, length=30),
        nullable=False,
        comment="New regime"
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Transition confidence (0.0-1.0)"
    )

    days_since_last_transition: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Days since previous transition"
    )

    # Relationships
    assessment: Mapped["CountryRegimeAssessment"] = relationship(
        "CountryRegimeAssessment",
        back_populates="regime_transitions"
    )

    __table_args__ = (
        Index('idx_transition_country', 'country'),
        Index('idx_transition_date', 'transition_date'),
        Index('idx_transition_country_date', 'country', 'transition_date'),
        Index('idx_transition_from_to', 'from_regime', 'to_regime'),
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='ck_transition_confidence'),
        CheckConstraint('days_since_last_transition IS NULL OR days_since_last_transition >= 0', name='ck_days_positive'),
        {
            'comment': 'Historical regime transitions for tracking changes',
        }
    )

    def __repr__(self) -> str:
        return f"<RegimeTransition(country='{self.country}', {self.from_regime} → {self.to_regime})>"