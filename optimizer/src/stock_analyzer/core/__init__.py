"""
Core Signal Calculator
======================

Main orchestrator for institutional-grade signal generation.

Coordinates:
- Data fetching (yfinance, database)
- Technical metrics calculation
- Factor calculations (value, momentum, quality, growth)
- Macro adjustments
- Signal classification
- Risk assessment
"""

from .signal_calculator import MathematicalSignalCalculator

__all__ = ['MathematicalSignalCalculator']
