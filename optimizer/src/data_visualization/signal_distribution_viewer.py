#!/usr/bin/env python3
"""
Signal Distribution Viewer - Terminal-Based Histograms
======================================================
Displays visual distributions of stock signals in the terminal:
- Signal type distribution (LARGE_GAIN, SMALL_GAIN, etc.)
- Sector distribution
- Exchange distribution
- Industry distribution (top 20)
- Confidence level distribution

Usage:
    python src/data_visualization/signal_distribution_viewer.py

Output:
    Visual histograms directly in the terminal
"""

import sys
import logging
from datetime import date as date_type
from typing import Dict, List
from collections import Counter

from sqlalchemy import select, func
from sqlalchemy.orm import joinedload

from dotenv import load_dotenv
load_dotenv()

# Import database and models
from app.database import database_manager, init_db
from app.models.stock_signals import StockSignal, SignalEnum
from app.models.universe import Instrument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalDistributionViewer:
    """
    Displays signal distributions as terminal histograms.
    """

    def __init__(self, signal_date: date_type | None = None):
        """
        Initialize viewer.

        Args:
            signal_date: Date to analyze (defaults to today)
        """
        self.signal_date = signal_date or date_type.today()
        self.signal_type_counts = {}
        self.sector_counts = {}
        self.industry_counts = {}
        self.exchange_counts = {}
        self.confidence_counts = {}

    def fetch_signal_statistics(self) -> None:
        """
        Fetch signal type distribution from database.
        """
        logger.info(f"Fetching signal type distribution for {self.signal_date}")

        with database_manager.get_session() as session:
            # Count signals by type
            query = select(
                StockSignal.signal_type,
                func.count(StockSignal.id).label('count')
            ).where(
                StockSignal.signal_date == self.signal_date
            ).group_by(
                StockSignal.signal_type
            )

            results = session.execute(query).all()
            self.signal_type_counts = {signal_type.value: count for signal_type, count in results}

            logger.info(f"Found {sum(self.signal_type_counts.values())} total signals")

    def fetch_all_signals_with_details(self) -> List[tuple]:
        """
        Fetch all signals with instrument details for detailed distributions.

        Returns:
            List of (signal, instrument) tuples
        """
        logger.info(f"Fetching all signals with details for {self.signal_date}")

        with database_manager.get_session() as session:
            query = select(StockSignal).where(
                StockSignal.signal_date == self.signal_date
            ).options(
                joinedload(StockSignal.instrument).joinedload(Instrument.exchange)
            )

            signals = session.execute(query).scalars().all()

            if not signals:
                logger.warning(f"No signals found for {self.signal_date}")
                return []

            results = [(signal, signal.instrument) for signal in signals]
            logger.info(f"Fetched {len(results)} signals with details")

            return results

    def analyze_distributions(self) -> None:
        """
        Analyze all distributions from signals.
        """
        # Get signal type counts
        self.fetch_signal_statistics()

        if not self.signal_type_counts:
            logger.warning("No signals to analyze")
            return

        # Get detailed signal data
        signals_and_instruments = self.fetch_all_signals_with_details()

        if not signals_and_instruments:
            return

        # Track distributions
        sectors = []
        industries = []
        exchanges = []
        confidences = []

        # Track unique instruments to avoid double-counting
        seen_instruments = set()

        for signal, instrument in signals_and_instruments:
            if not instrument:
                continue

            instrument_id = str(instrument.id)

            # Skip if already counted (same instrument multiple times)
            if instrument_id in seen_instruments:
                continue

            seen_instruments.add(instrument_id)

            # Collect distributions (use denormalized fields from signal when available)
            if signal.sector:
                sectors.append(signal.sector)

            if signal.industry:
                industries.append(signal.industry)

            if signal.exchange_name:
                exchanges.append(signal.exchange_name)
            elif instrument.exchange:
                exchanges.append(instrument.exchange.exchange_name)

            if signal.confidence_level:
                confidences.append(signal.confidence_level.value)

        # Calculate distributions
        self.sector_counts = dict(Counter(sectors))
        self.industry_counts = dict(Counter(industries))
        self.exchange_counts = dict(Counter(exchanges))
        self.confidence_counts = dict(Counter(confidences))

        logger.info(
            f"Analyzed {len(seen_instruments)} unique instruments: "
            f"{len(self.sector_counts)} sectors, "
            f"{len(self.industry_counts)} industries, "
            f"{len(self.exchange_counts)} exchanges"
        )

    def print_histogram(self, title: str, data: Dict[str, int], max_bars: int = 50,
                       top_n: int | None = None, show_percentages: bool = True) -> None:
        """
        Print a histogram in the terminal.

        Args:
            title: Title for the histogram
            data: Dictionary of {label: count}
            max_bars: Maximum width of bars in characters
            top_n: Show only top N items (None = show all)
            show_percentages: Whether to show percentages
        """
        if not data:
            print(f"\n{title}")
            print("=" * 100)
            print("No data available")
            return

        print(f"\n{title}")
        print("=" * 100)

        # Sort by count (descending)
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

        # Limit to top N if specified
        if top_n:
            sorted_data = sorted_data[:top_n]

        total = sum(data.values())
        max_count = max(item[1] for item in sorted_data)

        # Print each bar
        for label, count in sorted_data:
            # Calculate bar length
            if max_count > 0:
                bar_length = int((count / max_count) * max_bars)
            else:
                bar_length = 0

            bar = "█" * bar_length

            # Format percentage
            if show_percentages and total > 0:
                percentage = (count / total) * 100
                print(f"  {label[:40]:40} {count:5d} ({percentage:5.1f}%) {bar}")
            else:
                print(f"  {label[:40]:40} {count:5d} {bar}")

        print(f"\nTotal: {total}")

    def display_all_distributions(self) -> None:
        """
        Display all distributions as terminal histograms.
        """
        print("\n" + "=" * 100)
        print(f"SIGNAL DISTRIBUTION ANALYSIS - {self.signal_date}")
        print("=" * 100)

        # 1. Signal Type Distribution
        signal_order = ['large_gain', 'small_gain', 'neutral', 'small_decline', 'large_decline']
        ordered_signal_counts = {sig: self.signal_type_counts.get(sig, 0) for sig in signal_order}
        self.print_histogram("1. SIGNAL TYPE DISTRIBUTION", ordered_signal_counts)

        # 2. Sector Distribution
        self.print_histogram("2. SECTOR DISTRIBUTION (All Unique Stocks)", self.sector_counts)

        # 3. Exchange Distribution
        self.print_histogram("3. EXCHANGE DISTRIBUTION (All Unique Stocks)", self.exchange_counts)

        # 4. Industry Distribution (Top 20)
        self.print_histogram("4. INDUSTRY DISTRIBUTION (Top 20, All Unique Stocks)",
                           self.industry_counts, top_n=20)

        # 5. Confidence Level Distribution
        confidence_order = ['high', 'medium', 'low']
        ordered_confidence = {conf: self.confidence_counts.get(conf, 0) for conf in confidence_order}
        self.print_histogram("5. CONFIDENCE LEVEL DISTRIBUTION", ordered_confidence)

        # Summary
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"  Total Signals:        {sum(self.signal_type_counts.values())}")
        print(f"  Unique Sectors:       {len(self.sector_counts)}")
        print(f"  Unique Industries:    {len(self.industry_counts)}")
        print(f"  Unique Exchanges:     {len(self.exchange_counts)}")
        print("=" * 100)

    def run(self) -> None:
        """
        Run the distribution analysis and display results.
        """
        logger.info(f"Starting distribution analysis for {self.signal_date}")

        # Analyze all distributions
        self.analyze_distributions()

        if not self.signal_type_counts:
            print(f"\nNo signals found for {self.signal_date}")
            return

        # Display results
        self.display_all_distributions()

        logger.info("Analysis complete")


def get_most_recent_signal_date() -> date_type | None:
    """
    Fetch the most recent signal date from the database.

    Returns:
        Most recent signal date or None if no signals found
    """
    from sqlalchemy import func

    with database_manager.get_session() as session:
        query = select(func.max(StockSignal.signal_date))
        result = session.execute(query).scalar_one_or_none()

        if result:
            logger.info(f"Most recent signal date: {result}")
            return result
        else:
            logger.warning("No signals found in database")
            return None


def main():
    """
    Main function - display signal distributions.
    """
    print("=" * 100)
    print("SIGNAL DISTRIBUTION VIEWER")
    print("=" * 100)

    try:
        # Initialize database
        logger.info("Initializing database connection...")
        init_db()

        if not database_manager.is_initialized:
            logger.error("Database initialization failed")
            sys.exit(1)

        logger.info("Database connection established")

        # Get the most recent signal date from database
        signal_date = get_most_recent_signal_date()

        if signal_date is None:
            logger.error("No signals found in database. Run signal analysis first.")
            sys.exit(1)

        logger.info(f"Using most recent signal date: {signal_date}")
        print(f"\nAnalyzing signals for: {signal_date}")

        # Create viewer
        viewer = SignalDistributionViewer(signal_date=signal_date)

        # Run analysis and display
        viewer.run()

    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
