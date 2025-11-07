"""
Signal Processing
=================

Handles batch processing and signal generation for instruments.
"""

import logging
import asyncio
from datetime import date as date_type
from typing import List, Optional
from collections import defaultdict

from app.models.universe import Instrument
from app.database import database_manager
from baml_client.types import StockSignalOutput

from .database import save_signal

logger = logging.getLogger(__name__)


async def generate_signal_for_instrument(
    calculator,
    instrument: Instrument,
    signal_date: date_type
) -> Optional[StockSignalOutput]:
    """
    Generate signal for a single instrument.

    Args:
        calculator: MathematicalSignalCalculator instance
        instrument: Instrument object
        signal_date: Date to generate signal for

    Returns:
        StockSignalOutput or None if failed
    """
    try:
        if not instrument.yfinance_ticker:
            return None

        signal_output = await calculator.generate_signal(
            yf_ticker=instrument.yfinance_ticker,
            target_date=signal_date
        )

        return signal_output

    except Exception as e:
        logger.error(f"Error generating signal for {instrument.ticker}: {e}")
        return None


async def process_batch(
    calculator,
    batch: List[Instrument],
    signal_date: date_type,
    update_existing: bool,
    batch_num: int,
    stats: dict
) -> int:
    """
    Process a single batch of instruments concurrently.

    Args:
        calculator: MathematicalSignalCalculator instance
        batch: List of instruments to process
        signal_date: Date to generate signals for
        update_existing: If True, update existing signals
        batch_num: Batch number for logging
        stats: Statistics dictionary to update

    Returns:
        Number of successfully processed instruments in this batch
    """
    logger.info(f"Processing batch {batch_num} ({len(batch)} instruments)...")

    # Generate signals concurrently
    tasks = [
        generate_signal_for_instrument(calculator, instrument, signal_date)
        for instrument in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Save signals to database
    processed_count = 0
    with database_manager.get_session() as session:
        for instrument, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Exception for {instrument.ticker}: {result}")
                stats['errors'] += 1
                continue

            if result is None or not isinstance(result, StockSignalOutput):
                stats['skipped_no_ticker'] += 1
                continue

            # Save signal
            technical_metrics = calculator.last_technical_metrics
            success = save_signal(
                session,
                instrument.id,
                result,
                technical_metrics,
                update_if_exists=update_existing,
                instrument=instrument
            )

            if success:
                processed_count += 1
                stats['signals_generated'] += 1
                stats['by_signal_type'][result.signal_type.value] += 1
            else:
                stats['skipped_duplicate'] += 1

        session.commit()

    logger.info(f"Batch {batch_num} complete: {processed_count}/{len(batch)} processed")
    return processed_count


def create_batches(instruments: List[Instrument], batch_size: int) -> List[List[Instrument]]:
    """
    Split instruments into batches for parallel processing.

    Args:
        instruments: List of instruments
        batch_size: Number of instruments per batch

    Returns:
        List of batches
    """
    return [
        instruments[i:i + batch_size]
        for i in range(0, len(instruments), batch_size)
    ]


def initialize_stats() -> dict:
    """
    Initialize statistics tracking dictionary.

    Returns:
        Statistics dictionary
    """
    return {
        'total_instruments': 0,
        'signals_generated': 0,
        'signals_updated': 0,
        'errors': 0,
        'skipped_no_ticker': 0,
        'skipped_duplicate': 0,
        'skipped_already_processed': 0,
        'resumed_from_previous': False,
        'by_signal_type': defaultdict(int)
    }


def print_summary(stats: dict) -> None:
    """
    Print summary of signal generation run.

    Args:
        stats: Statistics dictionary
    """
    logger.info("\n" + "="*80)
    logger.info("SIGNAL GENERATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total instruments: {stats['total_instruments']}")
    logger.info(f"Signals generated: {stats['signals_generated']}")
    logger.info(f"Signals updated: {stats['signals_updated']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Skipped (no ticker): {stats['skipped_no_ticker']}")
    logger.info(f"Skipped (duplicate): {stats['skipped_duplicate']}")
    logger.info(f"Skipped (already processed): {stats['skipped_already_processed']}")

    if stats['by_signal_type']:
        logger.info("\nSignals by type:")
        for signal_type, count in sorted(stats['by_signal_type'].items()):
            logger.info(f"  {signal_type}: {count}")

    logger.info("="*80 + "\n")
