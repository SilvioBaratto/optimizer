import asyncio
from datetime import date as date_type
from typing import List, Optional
from collections import defaultdict

from optimizer.database.models.universe import Instrument
from optimizer.database.database import database_manager
from baml_client.types import StockSignalOutput

from .database import save_signal


async def generate_signal_for_instrument(
    calculator, instrument: Instrument, signal_date: date_type
) -> Optional[StockSignalOutput]:
    """Generate signal for a single instrument."""
    try:
        if not instrument.yfinance_ticker:
            return None

        signal_output = await calculator.generate_signal(
            yf_ticker=instrument.yfinance_ticker, target_date=signal_date
        )

        return signal_output

    except Exception:
        return None


async def process_batch(
    calculator,
    batch: List[Instrument],
    signal_date: date_type,
    update_existing: bool,
    batch_num: int,
    stats: dict,
) -> int:
    """Process a single batch of instruments concurrently."""
    # Generate signals concurrently
    tasks = [
        generate_signal_for_instrument(calculator, instrument, signal_date) for instrument in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Save signals to database
    processed_count = 0
    with database_manager.get_session() as session:
        for instrument, result in zip(batch, results):
            if isinstance(result, Exception):
                stats["errors"] += 1
                continue

            if result is None or not isinstance(result, StockSignalOutput):
                stats["skipped_no_ticker"] += 1
                continue

            # Save signal
            technical_metrics = calculator.last_technical_metrics
            success = save_signal(
                session,
                instrument.id,
                result,
                technical_metrics,
                update_if_exists=update_existing,
                instrument=instrument,
            )

            if success:
                processed_count += 1
                stats["signals_generated"] += 1
                stats["by_signal_type"][result.signal_type.value] += 1
            else:
                stats["skipped_duplicate"] += 1

        session.commit()

    return processed_count


def create_batches(instruments: List[Instrument], batch_size: int) -> List[List[Instrument]]:
    """Split instruments into batches for parallel processing."""
    return [instruments[i : i + batch_size] for i in range(0, len(instruments), batch_size)]


def initialize_stats() -> dict:
    """Initialize statistics tracking dictionary."""
    return {
        "total_instruments": 0,
        "signals_generated": 0,
        "signals_updated": 0,
        "errors": 0,
        "skipped_no_ticker": 0,
        "skipped_duplicate": 0,
        "skipped_already_processed": 0,
        "resumed_from_previous": False,
        "by_signal_type": defaultdict(int),
    }


def format_summary(stats: dict) -> str:
    """
    Format summary of signal generation run as a string.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted summary string
    """
    lines = [
        "",
        "=" * 80,
        "SIGNAL GENERATION SUMMARY",
        "=" * 80,
        f"Total instruments: {stats['total_instruments']}",
        f"Signals generated: {stats['signals_generated']}",
        f"Signals updated: {stats['signals_updated']}",
        f"Errors: {stats['errors']}",
        f"Skipped (no ticker): {stats['skipped_no_ticker']}",
        f"Skipped (duplicate): {stats['skipped_duplicate']}",
        f"Skipped (already processed): {stats['skipped_already_processed']}",
    ]

    if stats["by_signal_type"]:
        lines.append("")
        lines.append("Signals by type:")
        for signal_type, count in sorted(stats["by_signal_type"].items()):
            lines.append(f"  {signal_type}: {count}")

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)
