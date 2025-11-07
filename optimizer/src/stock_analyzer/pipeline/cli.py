"""
CLI Entry Point
===============

Command-line interface for the signal analysis pipeline.
"""

import asyncio
import argparse
import logging
from datetime import date as date_type
from typing import Optional

from app.database import init_db
from .analyzer import SignalAnalyzer

logger = logging.getLogger(__name__)


async def run_pipeline(
    signal_date: Optional[date_type] = None,
    max_instruments: Optional[int] = None,
    update_existing: bool = True,
    enable_cross_sectional: bool = False,
    skip_pass1: bool = False,
    concurrent_batches: int = 10,
    stocks_per_batch: int = 10,
) -> None:
    """
    Run the signal analysis pipeline.

    Args:
        signal_date: Date to generate signals for (defaults to today)
        max_instruments: Maximum number of instruments (for testing)
        update_existing: Update existing signals
        enable_cross_sectional: Use cross-sectional standardization
        skip_pass1: Skip Pass 1 (for resuming Pass 2)
        concurrent_batches: Number of parallel batches
        stocks_per_batch: Stocks per batch
    """
    # Initialize database
    init_db()

    # Create analyzer
    analyzer = SignalAnalyzer(
        concurrent_batches=concurrent_batches,
        stocks_per_batch=stocks_per_batch
    )

    # Run analysis
    await analyzer.analyze_instruments(
        signal_date=signal_date,
        max_instruments=max_instruments,
        update_existing=update_existing,
        enable_cross_sectional=enable_cross_sectional,
        skip_pass1=skip_pass1,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Signal Analyzer - Generate daily signals for all instruments"
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Signal date (YYYY-MM-DD). Defaults to today.'
    )

    parser.add_argument(
        '--max-instruments',
        type=int,
        help='Maximum number of instruments to process (for testing)'
    )

    parser.add_argument(
        '--no-update',
        action='store_true',
        help='Do not update existing signals (skip duplicates)'
    )

    parser.add_argument(
        '--cross-sectional',
        action='store_true',
        help='Enable two-pass cross-sectional standardization'
    )

    parser.add_argument(
        '--skip-pass1',
        action='store_true',
        help='Skip Pass 1 (use when resuming Pass 2)'
    )

    parser.add_argument(
        '--concurrent-batches',
        type=int,
        default=10,
        help='Number of concurrent batches (default: 10)'
    )

    parser.add_argument(
        '--stocks-per-batch',
        type=int,
        default=10,
        help='Number of stocks per batch (default: 10)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse date
    signal_date = None
    if args.date:
        from datetime import datetime
        signal_date = datetime.strptime(args.date, '%Y-%m-%d').date()

    # Run pipeline
    asyncio.run(run_pipeline(
        signal_date=signal_date,
        max_instruments=args.max_instruments,
        update_existing=not args.no_update,
        enable_cross_sectional=args.cross_sectional,
        skip_pass1=args.skip_pass1,
        concurrent_batches=args.concurrent_batches,
        stocks_per_batch=args.stocks_per_batch,
    ))


if __name__ == '__main__':
    main()
