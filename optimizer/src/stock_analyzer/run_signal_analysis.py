#!/usr/bin/env python3
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
    mathematical-based signals
    """
    import os
    from database.database import init_db, database_manager
    from src.stock_analyzer.core import MathematicalSignalCalculator

    DEFAULT_CONCURRENT_BATCHES = int(os.getenv("SIGNAL_CONCURRENT_BATCHES", "10"))
    DEFAULT_STOCKS_PER_BATCH = int(os.getenv("SIGNAL_STOCKS_PER_BATCH", "10"))

    try:
        # Initialize database
        init_db()

        if not database_manager.is_initialized:
            sys.exit(1)

        # Create Mathematical calculator (NO LLM = NO COST)
        calculator = MathematicalSignalCalculator()

        # Create analyzer with parallel batch processing
        analyzer = SignalAnalyzer(
            calculator=calculator,
            concurrent_batches=DEFAULT_CONCURRENT_BATCHES,
            stocks_per_batch=DEFAULT_STOCKS_PER_BATCH,
        )

        # Run analysis with skip_pass1 parameter
        signal_date = date_type.today()

        # Run analysis with full 7-pass institutional cross-sectional standardization
        await analyzer.analyze_instruments(
            signal_date=signal_date,
            max_instruments=None,
            update_existing=True,
            enable_cross_sectional=True,
            skip_pass1=skip_pass1,
        )

    except KeyboardInterrupt:
        sys.exit(1)

    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Signal Analyzer - Mathematical formulas (NO LLM cost)",
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
        """,
    )
    parser.add_argument(
        "--skip-pass1",
        action="store_true",
        help="Skip Pass 1 and load cached raw_data (for debugging Pass 1.5+)",
    )

    args = parser.parse_args()
    asyncio.run(main(skip_pass1=args.skip_pass1))
