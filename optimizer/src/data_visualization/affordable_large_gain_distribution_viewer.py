#!/usr/bin/env python3
"""
Affordable Large Gain Distribution Viewer - Capital-Constrained Analysis
========================================================================
Displays visual distributions of LARGE_GAIN stocks filtered by affordability:
- Takes capital parameter (e.g., €1000)
- Calculates max affordable price: capital / 20
- Shows only stocks <= max affordable price
- Sector distribution
- Exchange distribution
- Industry distribution (top 20)
- Confidence level distribution
- Country distribution
- Price range distribution

This allows analysis of which LARGE_GAIN stocks fit a diversified 20-position portfolio.

Usage:
    python src/data_visualization/affordable_large_gain_distribution_viewer.py --capital 1000
    python src/data_visualization/affordable_large_gain_distribution_viewer.py --capital 5000

Output:
    Visual histograms in terminal for affordable LARGE_GAIN signals only
"""

import sys
import argparse
import logging
from datetime import date as date_type
from typing import Dict, List
from collections import Counter

from sqlalchemy import select, func
from sqlalchemy.orm import joinedload

from dotenv import load_dotenv

load_dotenv()

# Import database and models
from optimizer.database.database import database_manager, init_db
from optimizer.database.models.stock_signals import StockSignal, SignalEnum
from optimizer.database.models.universe import Instrument

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AffordableLargeGainDistributionViewer:
    """
    Displays LARGE_GAIN signal distributions filtered by affordability.
    Only includes stocks where price <= (capital / 20).
    """

    def __init__(self, signal_date: date_type | None = None, capital: float = 1000.0):
        """
        Initialize viewer.

        Args:
            signal_date: Date to analyze (defaults to today)
            capital: Total capital available (EUR)
        """
        self.signal_date = signal_date or date_type.today()
        self.capital = capital
        self.max_price = capital / 20.0  # For 20-position portfolio
        self.sector_counts = {}
        self.industry_counts = {}
        self.exchange_counts = {}
        self.confidence_counts = {}
        self.country_counts = {}
        self.price_range_counts = {}
        self.total_large_gain = 0
        self.total_excluded = 0

    def fetch_affordable_large_gain_signals(self) -> List[tuple]:
        """
        Fetch ONLY LARGE_GAIN signals where price <= max_price.

        Returns:
            List of (signal, instrument) tuples
        """
        logger.info(f"Fetching LARGE_GAIN signals for {self.signal_date}")
        logger.info(
            f"Capital: €{self.capital:,.2f} → Max price per stock: €{self.max_price:.2f}"
        )

        with database_manager.get_session() as session:
            # First, get total LARGE_GAIN count (before filtering)
            total_query = select(func.count(StockSignal.id)).where(
                StockSignal.signal_date == self.signal_date,
                StockSignal.signal_type == SignalEnum.LARGE_GAIN,
            )
            total_count = session.execute(total_query).scalar_one()

            # Now get affordable signals only
            query = (
                select(StockSignal)
                .where(
                    StockSignal.signal_date == self.signal_date,
                    StockSignal.signal_type == SignalEnum.LARGE_GAIN,
                    StockSignal.close_price <= self.max_price,
                )
                .options(
                    joinedload(StockSignal.instrument).joinedload(Instrument.exchange)
                )
                .order_by(StockSignal.close_price.desc())
            )

            signals = session.execute(query).scalars().all()

            affordable_count = len(signals)
            self.total_excluded = total_count - affordable_count

            if not signals:
                logger.warning(
                    f"No affordable LARGE_GAIN signals found (max price: €{self.max_price:.2f}). "
                    f"Total LARGE_GAIN signals: {total_count}"
                )
                return []

            results = [(signal, signal.instrument) for signal in signals]
            logger.info(
                f"Found {affordable_count} affordable LARGE_GAIN signals "
                f"(excluded {self.total_excluded} too expensive)"
            )

            return results

    def categorize_price(self, price: float | None) -> str:
        """
        Categorize stock price into ranges.

        Args:
            price: Stock close price

        Returns:
            Price range category
        """
        if price is None:
            return "Unknown"

        if price < 10:
            return "< €10"
        elif price < 50:
            return "€10-€50"
        elif price < 100:
            return "€50-€100"
        elif price < 500:
            return "€100-€500"
        else:
            return "> €500"

    def get_country_from_exchange(self, exchange_name: str | None) -> str:
        """
        Map exchange name to country.

        Args:
            exchange_name: Exchange name

        Returns:
            Country name
        """
        if not exchange_name:
            return "Unknown"

        exchange_to_country = {
            "NYSE": "USA",
            "NASDAQ": "USA",
            "London Stock Exchange": "UK",
            "Deutsche Börse Xetra": "Germany",
            "Gettex": "Germany",
            "Euronext Paris": "France",
            "Euronext Amsterdam": "Netherlands",
            "SIX Swiss Exchange": "Switzerland",
            "Tokyo Stock Exchange": "Japan",
        }

        return exchange_to_country.get(exchange_name, exchange_name)

    def analyze_distributions(self) -> None:
        """
        Analyze all distributions from affordable LARGE_GAIN signals.
        """
        # Get affordable LARGE_GAIN signals
        signals_and_instruments = self.fetch_affordable_large_gain_signals()

        if not signals_and_instruments:
            logger.warning("No affordable LARGE_GAIN signals to analyze")
            return

        # Track distributions
        sectors = []
        industries = []
        exchanges = []
        confidences = []
        countries = []
        price_ranges = []

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

            # Exchange
            exchange_name = None
            if signal.exchange_name:
                exchange_name = signal.exchange_name
            elif instrument.exchange:
                exchange_name = instrument.exchange.exchange_name

            if exchange_name:
                exchanges.append(exchange_name)
                # Derive country from exchange
                country = self.get_country_from_exchange(exchange_name)
                countries.append(country)

            # Confidence level
            if signal.confidence_level:
                confidences.append(signal.confidence_level.value)

            # Price range
            price_range = self.categorize_price(signal.close_price)
            price_ranges.append(price_range)

        # Calculate distributions
        self.sector_counts = dict(Counter(sectors))
        self.industry_counts = dict(Counter(industries))
        self.exchange_counts = dict(Counter(exchanges))
        self.confidence_counts = dict(Counter(confidences))
        self.country_counts = dict(Counter(countries))
        self.price_range_counts = dict(Counter(price_ranges))
        self.total_large_gain = len(seen_instruments)

        logger.info(
            f"Analyzed {self.total_large_gain} affordable LARGE_GAIN instruments: "
            f"{len(self.sector_counts)} sectors, "
            f"{len(self.industry_counts)} industries, "
            f"{len(self.exchange_counts)} exchanges, "
            f"{len(self.country_counts)} countries"
        )

    def print_histogram(
        self,
        title: str,
        data: Dict[str, int],
        max_bars: int = 50,
        top_n: int | None = None,
        show_percentages: bool = True,
    ) -> None:
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
        print(f"AFFORDABLE LARGE_GAIN DISTRIBUTION ANALYSIS - {self.signal_date}")
        print("=" * 100)
        print(f"\nCapital: €{self.capital:,.2f}")
        print(f"Max Price per Stock: €{self.max_price:.2f} (capital / 20)")
        print(f"Affordable LARGE_GAIN stocks: {self.total_large_gain}")
        print(f"Excluded (too expensive): {self.total_excluded}")
        print("=" * 100)

        # 1. Sector Distribution
        self.print_histogram(
            f"1. SECTOR DISTRIBUTION (Affordable LARGE_GAIN: ≤€{self.max_price:.2f})",
            self.sector_counts,
        )

        # 2. Country Distribution
        self.print_histogram(
            f"2. COUNTRY DISTRIBUTION (Affordable LARGE_GAIN: ≤€{self.max_price:.2f})",
            self.country_counts,
        )

        # 3. Exchange Distribution
        self.print_histogram(
            f"3. EXCHANGE DISTRIBUTION (Affordable LARGE_GAIN: ≤€{self.max_price:.2f})",
            self.exchange_counts,
        )

        # 4. Industry Distribution (Top 20)
        self.print_histogram(
            f"4. INDUSTRY DISTRIBUTION (Top 20, Affordable LARGE_GAIN: ≤€{self.max_price:.2f})",
            self.industry_counts,
            top_n=20,
        )

        # 5. Price Range Distribution
        price_range_order = [
            "< €10",
            "€10-€50",
            "€50-€100",
            "€100-€500",
            "> €500",
            "Unknown",
        ]
        ordered_price_ranges = {
            pr: self.price_range_counts.get(pr, 0)
            for pr in price_range_order
            if pr in self.price_range_counts
        }
        self.print_histogram(
            f"5. PRICE RANGE DISTRIBUTION (Affordable LARGE_GAIN: ≤€{self.max_price:.2f})",
            ordered_price_ranges,
        )

        # 6. Confidence Level Distribution
        confidence_order = ["high", "medium", "low"]
        ordered_confidence = {
            conf: self.confidence_counts.get(conf, 0) for conf in confidence_order
        }
        self.print_histogram(
            f"6. CONFIDENCE LEVEL DISTRIBUTION (Affordable LARGE_GAIN: ≤€{self.max_price:.2f})",
            ordered_confidence,
        )

        # Summary
        print("\n" + "=" * 100)
        print("SUMMARY - AFFORDABLE LARGE_GAIN STOCKS ONLY")
        print("=" * 100)
        print(f"  Capital Available:         €{self.capital:,.2f}")
        print(f"  Max Price per Stock:       €{self.max_price:.2f}")
        print(f"  Affordable LARGE_GAIN:     {self.total_large_gain}")
        print(f"  Excluded (too expensive):  {self.total_excluded}")
        print(f"  Unique Sectors:            {len(self.sector_counts)}")
        print(f"  Unique Industries:         {len(self.industry_counts)}")
        print(f"  Unique Exchanges:          {len(self.exchange_counts)}")
        print(f"  Unique Countries:          {len(self.country_counts)}")
        print(f"  Price Ranges Covered:      {len(self.price_range_counts)}")
        print("=" * 100)

        # Top performers by sector
        if self.sector_counts:
            print("\nTOP 3 SECTORS (Most affordable LARGE_GAIN stocks):")
            top_sectors = sorted(
                self.sector_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            for i, (sector, count) in enumerate(top_sectors, 1):
                percentage = (
                    (count / self.total_large_gain) * 100
                    if self.total_large_gain > 0
                    else 0
                )
                print(f"  {i}. {sector:30} {count:3d} stocks ({percentage:5.1f}%)")

        # Top performers by country
        if self.country_counts:
            print("\nTOP 3 COUNTRIES (Most affordable LARGE_GAIN stocks):")
            top_countries = sorted(
                self.country_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            for i, (country, count) in enumerate(top_countries, 1):
                percentage = (
                    (count / self.total_large_gain) * 100
                    if self.total_large_gain > 0
                    else 0
                )
                print(f"  {i}. {country:30} {count:3d} stocks ({percentage:5.1f}%)")

        print("=" * 100)

    def run(self) -> None:
        """
        Run the distribution analysis and display results.
        """
        logger.info(
            f"Starting affordable LARGE_GAIN distribution analysis for {self.signal_date} "
            f"(capital: €{self.capital:,.2f}, max price: €{self.max_price:.2f})"
        )

        # Analyze all distributions
        self.analyze_distributions()

        if self.total_large_gain == 0:
            print(
                f"\nNo affordable LARGE_GAIN signals found for {self.signal_date} "
                f"(max price: €{self.max_price:.2f})"
            )
            print(f"Excluded {self.total_excluded} signals that were too expensive.")
            print(f"\nTip: Try increasing --capital parameter to see more stocks.")
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
    Main function - display affordable LARGE_GAIN signal distributions.
    """
    parser = argparse.ArgumentParser(
        description="Analyze LARGE_GAIN signals filtered by affordability"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Total capital available in EUR (default: 1000). Max price = capital/20",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("AFFORDABLE LARGE_GAIN DISTRIBUTION VIEWER")
    print("=" * 100)
    print(f"Capital: €{args.capital:,.2f}")
    print(f"Max Price per Stock: €{args.capital / 20:.2f} (for 20-position portfolio)")
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
        print(f"\nAnalyzing affordable LARGE_GAIN signals for: {signal_date}")

        # Create viewer
        viewer = AffordableLargeGainDistributionViewer(
            signal_date=signal_date, capital=args.capital
        )

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
