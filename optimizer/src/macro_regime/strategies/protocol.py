from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
from enum import Enum


class MacroRegime(str, Enum):
    """Business cycle regime enumeration."""

    EARLY_CYCLE = "early_cycle"
    MID_CYCLE = "mid_cycle"
    LATE_CYCLE = "late_cycle"
    RECESSION = "recession"
    UNCERTAIN = "uncertain"

    @classmethod
    def from_string(cls, value: str) -> "MacroRegime":
        """Parse regime from string with normalization."""
        if not value:
            return cls.UNCERTAIN

        normalized = value.lower().replace(" ", "_").strip()

        mapping = {
            "mid_cycle": cls.MID_CYCLE,
            "midcycle": cls.MID_CYCLE,
            "mid": cls.MID_CYCLE,
            "early_cycle": cls.EARLY_CYCLE,
            "earlycycle": cls.EARLY_CYCLE,
            "early": cls.EARLY_CYCLE,
            "late_cycle": cls.LATE_CYCLE,
            "latecycle": cls.LATE_CYCLE,
            "late": cls.LATE_CYCLE,
            "recession": cls.RECESSION,
            "contraction": cls.RECESSION,
            "uncertain": cls.UNCERTAIN,
            "unknown": cls.UNCERTAIN,
            "mixed": cls.UNCERTAIN,
        }

        return mapping.get(normalized, cls.UNCERTAIN)


@dataclass(frozen=True)
class EconomicIndicators:
    """Container for economic indicators used in classification."""

    # GDP
    gdp_growth_qq: Optional[float] = None
    gdp_growth_yy: Optional[float] = None
    gdp_forecast_6m: Optional[float] = None

    # Labor Market
    unemployment: Optional[float] = None

    # Production
    industrial_production: Optional[float] = None
    capacity_utilization: Optional[float] = None

    # Inflation
    inflation: Optional[float] = None
    inflation_mom: Optional[float] = None
    core_inflation: Optional[float] = None
    inflation_forecast: Optional[float] = None

    # Consumption
    retail_sales_mom: Optional[float] = None

    # Corporate
    earnings_growth_forecast: Optional[float] = None

    # Confidence
    business_confidence: Optional[float] = None
    consumer_confidence: Optional[float] = None

    # External
    trade_balance: Optional[float] = None
    current_account: Optional[float] = None
    current_account_gdp: Optional[float] = None

    # Fiscal
    government_debt_gdp: Optional[float] = None
    budget_balance_gdp: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary."""
        return {
            "gdp_growth_qq": self.gdp_growth_qq,
            "gdp_growth_yy": self.gdp_growth_yy,
            "gdp_forecast_6m": self.gdp_forecast_6m,
            "unemployment": self.unemployment,
            "industrial_production": self.industrial_production,
            "capacity_utilization": self.capacity_utilization,
            "inflation": self.inflation,
            "inflation_mom": self.inflation_mom,
            "core_inflation": self.core_inflation,
            "inflation_forecast": self.inflation_forecast,
            "retail_sales_mom": self.retail_sales_mom,
            "earnings_growth_forecast": self.earnings_growth_forecast,
            "business_confidence": self.business_confidence,
            "consumer_confidence": self.consumer_confidence,
            "trade_balance": self.trade_balance,
            "current_account": self.current_account,
            "current_account_gdp": self.current_account_gdp,
            "government_debt_gdp": self.government_debt_gdp,
            "budget_balance_gdp": self.budget_balance_gdp,
        }


@dataclass(frozen=True)
class MarketIndicators:
    """Container for market indicators used in classification."""

    # PMI
    ism_pmi: Optional[float] = None
    manufacturing_pmi: Optional[float] = None
    services_pmi: Optional[float] = None
    composite_pmi: Optional[float] = None

    # Interest Rates
    interest_rate: Optional[float] = None

    # Yield Curve
    yield_curve_2s10s: Optional[float] = None
    bond_yield_2y: Optional[float] = None
    bond_yield_5y: Optional[float] = None
    bond_yield_10y: Optional[float] = None
    bond_yield_30y: Optional[float] = None

    # Credit
    hy_spread: Optional[float] = None

    # Volatility
    vix: Optional[float] = None
    vix_signal: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ism_pmi": self.ism_pmi,
            "manufacturing_pmi": self.manufacturing_pmi,
            "services_pmi": self.services_pmi,
            "composite_pmi": self.composite_pmi,
            "interest_rate": self.interest_rate,
            "yield_curve_2s10s": self.yield_curve_2s10s,
            "bond_yield_2y": self.bond_yield_2y,
            "bond_yield_5y": self.bond_yield_5y,
            "bond_yield_10y": self.bond_yield_10y,
            "bond_yield_30y": self.bond_yield_30y,
            "hy_spread": self.hy_spread,
            "vix": self.vix,
            "vix_signal": self.vix_signal,
        }


@dataclass
class ClassificationResult:
    """Result of regime classification."""

    regime: MacroRegime
    confidence: float
    rationale: str
    signals: Dict[str, Any] = field(default_factory=dict)

    # Recession risk
    recession_risk_6m: float = 0.05
    recession_risk_12m: float = 0.10

    # Recommendations
    sector_tilts: Dict[str, float] = field(default_factory=dict)
    factor_timing: Dict[str, str] = field(default_factory=dict)

    # Transition probability
    transition_probability: float = 0.15

    # Additional metadata
    primary_risks: List[str] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "signals": self.signals,
            "recession_risk_6m": self.recession_risk_6m,
            "recession_risk_12m": self.recession_risk_12m,
            "sector_tilts": self.sector_tilts,
            "factor_timing": self.factor_timing,
            "transition_probability": self.transition_probability,
            "primary_risks": self.primary_risks,
            "conflicting_signals": self.conflicting_signals,
        }


@runtime_checkable
class ClassificationStrategy(Protocol):
    """Protocol for regime classification strategies."""

    def classify(
        self,
        economic_indicators: EconomicIndicators,
        market_indicators: MarketIndicators,
        country: str = "USA",
    ) -> ClassificationResult:
        """
        Classify the current business cycle regime.

        Args:
            economic_indicators: Economic data for classification
            market_indicators: Market data for classification
            country: Country code for analysis

        Returns:
            ClassificationResult with regime, confidence, and recommendations
        """
        ...


class BaseClassificationStrategy(ABC):
    """Base class for classification strategies with common utilities."""

    # Regime definitions from institutional methodology
    REGIME_DEFINITIONS = {
        MacroRegime.EARLY_CYCLE: {
            "duration": "~1 year",
            "returns": "20%+ annualized",
            "description": "Recovery from recession",
        },
        MacroRegime.MID_CYCLE: {
            "duration": "~4 years",
            "returns": "14% annualized",
            "description": "Sustained expansion",
        },
        MacroRegime.LATE_CYCLE: {
            "duration": "~1.5 years",
            "returns": "5% annualized",
            "description": "Peak and slowdown",
        },
        MacroRegime.RECESSION: {
            "duration": "<1 year",
            "returns": "Negative",
            "description": "Economic contraction",
        },
    }

    def __init__(self):
        pass

    @abstractmethod
    def classify(
        self,
        economic_indicators: EconomicIndicators,
        market_indicators: MarketIndicators,
        country: str = "USA",
    ) -> ClassificationResult:
        """Classify the current regime."""
        pass

    @staticmethod
    def calculate_momentum(
        forecast_value: Optional[float],
        current_value: Optional[float],
    ) -> Optional[float]:
        """Calculate momentum as forecast - current."""
        if forecast_value is None or current_value is None:
            return None
        return forecast_value - current_value
