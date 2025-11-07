#!/usr/bin/env python3
"""
Stock Signal Analyzer - Daily Signal Generation with Mathematical Formulas
===========================================================================
This script analyzes all stocks in the universe and generates daily signals
using pure mathematical formulas (NO LLM = NO COST).

Features:
- Loops through all active instruments in the database
- Generates signals using Chapter 2 macroeconomic framework
- Combines: technical metrics, macro regime, risk-adjusted performance, alpha
- Saves signals to stock_signals table with detailed analysis notes
- Handles duplicate prevention (unique constraint on instrument_id + signal_date)
- Provides progress tracking and summary statistics
- Async batch processing for efficiency
- ZERO LLM COSTS - only yfinance API usage

REFACTORED:
-----------
This file has been refactored into a modular structure. The implementation is now in:
- pipeline/analyzer.py     - Main SignalAnalyzer orchestrator
- pipeline/cli.py          - Command-line interface
- pipeline/processing.py   - Batch processing logic
- pipeline/database.py     - Database operations
- pipeline/statistics.py   - Cross-sectional statistics
- pipeline/utils.py        - Helper functions

This file now serves as a backward compatibility layer, re-exporting the
refactored components to maintain existing imports and entry points.

Usage:
    Run from VS Code debugger using "Signal Analyzer - Run All" configuration.

    Or from command line:
    $ python run_signal_analysis.py                 # Normal run
    $ python run_signal_analysis.py --skip-pass1    # Debug mode (use cached data)

Configuration (via environment variables):
    SIGNAL_BATCH_SIZE: Batch size for database commits (default: 50)
"""

import sys

# Re-export SignalAnalyzer from refactored module
from pipeline.analyzer import SignalAnalyzer

# Re-export run_pipeline from refactored CLI module (for programmatic use)
from pipeline.cli import run_pipeline  # noqa: F401

# For backward compatibility, create a main() function that calls run_pipeline
import asyncio
import argparse
from datetime import date as date_type


async def main(skip_pass1: bool = False):
    """
    Main async function - analyzes all stocks in universe and generates
    mathematical-based signals (NO LLM = NO COST).

    Smart Resume Logic:
    - Automatically detects incomplete runs from today or yesterday
    - Resumes incomplete runs instead of starting fresh
    - Saves each signal immediately (no data loss on crash)

    Configuration can be customized via environment variables:
    - SIGNAL_BATCH_SIZE: Batch size for commits (default: 10)

    Args:
        skip_pass1: If True, skip Pass 1 and load cached data (for debugging)

    Note:
        This function is a backward compatibility wrapper around the refactored
        pipeline. The actual implementation is in pipeline/cli.py
    """
    # For backward compatibility, we need to parse args and call run_pipeline
    # However, run_pipeline expects different parameters
    # So we'll call the SignalAnalyzer directly as the old code did

    import logging
    import os
    from app.database import init_db, database_manager
    from src.stock_analyzer.core import MathematicalSignalCalculator

    logger = logging.getLogger(__name__)

    DEFAULT_CONCURRENT_BATCHES = int(os.getenv('SIGNAL_CONCURRENT_BATCHES', '10'))
    DEFAULT_STOCKS_PER_BATCH = int(os.getenv('SIGNAL_STOCKS_PER_BATCH', '10'))

    print("=" * 80)
    print("STOCK SIGNAL ANALYZER - MATHEMATICAL FORMULAS (NO LLM COST)")
    print("=" * 80)

    if skip_pass1:
        print("🔍 DEBUG MODE: Will skip Pass 1 and load cached data if available")
        print("=" * 80)

    try:
        # Initialize database
        logger.info("Initializing database connection...")
        init_db()

        if not database_manager.is_initialized:
            logger.error("Database initialization failed")
            sys.exit(1)

        logger.info("Database connection established")

        # Create Mathematical calculator (NO LLM = NO COST)
        calculator = MathematicalSignalCalculator()

        # Create analyzer with parallel batch processing
        analyzer = SignalAnalyzer(
            calculator=calculator,
            concurrent_batches=DEFAULT_CONCURRENT_BATCHES,
            stocks_per_batch=DEFAULT_STOCKS_PER_BATCH
        )

        # Run analysis with skip_pass1 parameter
        signal_date = date_type.today()

        print(f"\nRun Mode:           FRESH or RESUME (auto-detected)")
        print(f"Signal date:        {signal_date}")
        print(f"\nAnalysis method:    Mathematical (Chapter 2 Framework)")
        print(f"Cost per run:       $0.00 (No LLM calls)")
        print(f"Update existing:    Yes")
        print(f"\nParallel Processing:")
        print(f"  - Concurrent batches:  {DEFAULT_CONCURRENT_BATCHES}")
        print(f"  - Stocks per batch:    {DEFAULT_STOCKS_PER_BATCH}")
        print(f"  - Total parallel:      {DEFAULT_CONCURRENT_BATCHES * DEFAULT_STOCKS_PER_BATCH} stocks at once")
        print(f"\nSmart Resume:       Enabled")
        print(f"  - Automatically detects incomplete runs from today or yesterday")
        print(f"  - Skips already processed stocks")
        print(f"  - Saves each signal immediately (no data loss on crash)")
        print("=" * 80)

        # Run analysis with full 7-pass institutional cross-sectional standardization
        # This implements the complete approach from the original run_signal_analysis.py:
        # - Pass 1: Fetch raw fundamentals (with caching)
        # - Pass 1.5: Calculate robust cross-sectional statistics (iterative outlier removal)
        # - Pass 1B: Recalculate z-scores with robust stats
        # - Pass 2: Cross-sectional standardization (winsorization + scaling)
        # - Pass 2.5: Robust factor statistics
        # - Pass 2.6: Factor correlation analysis
        # - Pass 3: Classify and save signals (with momentum filters)
        await analyzer.analyze_instruments(
            signal_date=signal_date,
            max_instruments=None,
            update_existing=True,
            enable_cross_sectional=True,  # Enable full 7-pass institutional approach
            skip_pass1=skip_pass1
        )

        # Print summary
        analyzer.print_summary()

        logger.info("Mathematical signal analysis completed successfully (Cost: $0.00)")

    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Stock Signal Analyzer - Mathematical formulas (NO LLM cost)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal run (fetches all data from yfinance)
  python run_signal_analysis.py

  # Debug mode (skip Pass 1 and load cached data)
  python run_signal_analysis.py --skip-pass1

Debug Mode:
  The --skip-pass1 flag loads cached raw_data from a previous Pass 1 run,
  skipping the expensive yfinance fetching. Useful for debugging Pass 1.5,
  Pass 1B, Pass 2, or Pass 3 without re-fetching all stock data.

  Cache location: src/stock_analyzer/.cache/pass1_raw_data_YYYY-MM-DD.pkl
        """
    )
    parser.add_argument(
        '--skip-pass1',
        action='store_true',
        help='Skip Pass 1 and load cached raw_data (for debugging Pass 1.5+)'
    )

    args = parser.parse_args()
    asyncio.run(main(skip_pass1=args.skip_pass1))
