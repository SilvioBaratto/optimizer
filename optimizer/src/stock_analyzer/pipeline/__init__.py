"""
Signal Analysis Pipeline
========================

Modular pipeline for daily stock signal generation.

Components:
- utils: Helper functions (enum mapping, safe conversions)
- database: Database operations (fetch, check, save)
- statistics: Cross-sectional statistics calculation
- processing: Batch processing and signal generation
- analyzer: Main SignalAnalyzer orchestrator
- cli: Command-line interface
"""

from .analyzer import SignalAnalyzer
from .cli import run_pipeline

__all__ = ['SignalAnalyzer', 'run_pipeline']
