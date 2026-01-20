"""
Macro Regime Strategies Module - Classification strategy implementations.

Provides:
- ClassificationStrategy: Protocol for regime classification strategies
- RuleBasedClassifier: Rule-based regime classification
- LLMBasedClassifier: LLM-based regime classification

Usage:
    from src.macro_regime.strategies import RuleBasedClassifier, LLMBasedClassifier

    # Rule-based classification
    classifier = RuleBasedClassifier()
    result = classifier.classify(indicators)

    # LLM-based classification
    llm_classifier = LLMBasedClassifier()
    result = await llm_classifier.classify_async(indicators, news)
"""

from optimizer.src.macro_regime.strategies.protocol import (
    ClassificationStrategy,
    ClassificationResult,
    EconomicIndicators,
    MarketIndicators,
)
from optimizer.src.macro_regime.strategies.rule_based import RuleBasedClassifier
from optimizer.src.macro_regime.strategies.llm_based import LLMBasedClassifier

__all__ = [
    "ClassificationStrategy",
    "ClassificationResult",
    "EconomicIndicators",
    "MarketIndicators",
    "RuleBasedClassifier",
    "LLMBasedClassifier",
]
