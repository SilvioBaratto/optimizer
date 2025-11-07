"""
Stock Analyzer Module
=====================
Daily stock signal generation and analysis for portfolio optimization.

This module provides tools to:
- Analyze price movements for all stocks in the universe
- Generate daily signals (large_decline, small_decline, neutral, small_gain, large_gain)
- Calculate technical indicators (RSI, volatility)
- Save signals to database for historical tracking

Refactored Structure:
- core/: Main signal calculator orchestrator
- data/: Data fetching (yfinance, database)
- technical/: Technical indicators and metrics
- factors/: Factor calculations (value, momentum, quality, growth)
- adjustments/: Macro and risk adjustments
- classification/: Signal classification and scoring

Main Components:
- mathematical_signal_calculator.py: Backward compatibility layer
- core/signal_calculator.py: Refactored signal calculation logic
- run_signal_analysis.py: Main script to analyze all stocks

Usage:
    cd src/stock_analyzer
    python run_signal_analysis.py --help
"""

# Note: LLMSignalCalculator has been removed/deprecated
# The mathematical signal calculator is the primary implementation
from .mathematical_signal_calculator import MathematicalSignalCalculator

# For backward compatibility, if LLMSignalCalculator was used elsewhere
SignalCalculator = MathematicalSignalCalculator

__all__ = ['SignalCalculator', 'MathematicalSignalCalculator']
