"""
Macro Regime Analysis Module
=============================
Comprehensive business cycle detection and portfolio positioning system.

This is the refactored version following SOLID principles:
- Strategy Pattern for classification (rule-based vs LLM)
- Single Responsibility: Separate modules for classification, indicators
- Dependency Injection: Pluggable classification strategies

Components:
- classifier: Main orchestration class
- strategies: Classification strategies (rule-based, LLM)
- indicators: Economic indicator analysis (recession risk, sector tilts, factor timing)

Usage:
    from src.macro_regime import BusinessCycleClassifier, MacroRegime

    classifier = BusinessCycleClassifier(strategy="llm")
    result = await classifier.classify_async(
        ilsole_data=data,
        fred_data=market,
        news_data=news,
        country="USA",
    )
    # Access result.regime, result.recession_risk_6m, etc.
"""

from optimizer.src.macro_regime.classifier import BusinessCycleClassifier
from optimizer.src.macro_regime.strategies.protocol import (
    MacroRegime,
    ClassificationResult,
    EconomicIndicators,
    MarketIndicators,
    ClassificationStrategy,
)
from optimizer.src.macro_regime.strategies.rule_based import RuleBasedClassifier
from optimizer.src.macro_regime.strategies.llm_based import LLMBasedClassifier
from optimizer.src.macro_regime.indicators.recession_risk import RecessionRiskCalculator
from optimizer.src.macro_regime.indicators.sector_tilts import SectorTiltCalculator
from optimizer.src.macro_regime.indicators.factor_timing import FactorTimingCalculator

__version__ = "2.0.0"

__all__ = [
    # Main classifier
    "BusinessCycleClassifier",
    # Protocols and types
    "MacroRegime",
    "ClassificationResult",
    "EconomicIndicators",
    "MarketIndicators",
    "ClassificationStrategy",
    # Strategies
    "RuleBasedClassifier",
    "LLMBasedClassifier",
    # Indicators
    "RecessionRiskCalculator",
    "SectorTiltCalculator",
    "FactorTimingCalculator",
]
