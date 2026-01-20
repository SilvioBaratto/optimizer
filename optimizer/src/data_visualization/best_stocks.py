#!/usr/bin/env python3
"""
Best Stocks Visualization - LARGE_GAIN Signal Analysis
=======================================================
Visualizes stocks with LARGE_GAIN signals:
1. Close prices comparison chart (bar chart)
2. Sector distribution pie chart
3. 5-year price growth chart (line chart with normalized prices)

Usage:
    Run from VS Code debugger or directly:
    python src/data_visualization/best_stocks.py
"""

import sys
import logging
from datetime import date as date_type
from typing import List, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns  # type: ignore
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

from optimizer.src.yfinance import YFinanceClient

# Import database and models
from optimizer.database.database import database_manager, init_db
from optimizer.database.models.stock_signals import StockSignal, SignalEnum
from optimizer.database.models.universe import Instrument

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)


class LargeGainVisualizer:
    """
    Visualizer for LARGE_GAIN signal analysis.

    Creates:
    - Close price comparison chart (bar chart) for LARGE_GAIN stocks
    - Sector distribution pie chart
    - 5-year price growth chart (line chart with normalized prices)
    """

    def __init__(self, signal_date: date_type | None = None):
        """
        Initialize visualizer.

        Args:
            signal_date: Date to analyze (defaults to today)
        """
        self.signal_date = signal_date or date_type.today()
        self.large_gain_stocks = []
        self.sector_data = {}
        self.exchange_data = {}
        # Track unique instruments to avoid duplicates
        self.distinct_stocks_by_instrument = {}  # instrument_id -> stock_data

    def fetch_large_gain_signals(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Fetch all LARGE_GAIN signals for the target date.

        Returns:
            List of tuples (StockSignal, Instrument)
        """
        logger.info(f"Fetching LARGE_GAIN signals for {self.signal_date}")

        with database_manager.get_session() as session:
            query = (
                select(StockSignal)
                .where(
                    StockSignal.signal_date == self.signal_date,
                    StockSignal.signal_type == SignalEnum.LARGE_GAIN,
                )
                .options(
                    joinedload(StockSignal.instrument).joinedload(Instrument.exchange)
                )
                .order_by(StockSignal.close_price.desc())
            )

            signals = session.execute(query).scalars().all()

            if not signals:
                logger.warning(f"No LARGE_GAIN signals found for {self.signal_date}")
                return []

            results = [(signal, signal.instrument) for signal in signals]
            logger.info(f"Found {len(results)} LARGE_GAIN signals")

            return results

    def fetch_sector_info(self, yf_ticker: str) -> str:
        """
        Fetch sector information from yfinance.

        Args:
            yf_ticker: Yahoo Finance ticker symbol

        Returns:
            Sector name or 'Unknown' if not available
        """
        try:
            client = YFinanceClient.get_instance()
            info = client.fetch_info(yf_ticker)
            if info is None:
                return "Unknown"
            sector = info.get("sector", "Unknown")
            return sector
        except Exception as e:
            logger.warning(f"Could not fetch sector for {yf_ticker}: {e}")
            return "Unknown"

    def prepare_data(self) -> None:
        """
        Fetch and prepare data for visualization.
        Tracks DISTINCT stocks by instrument_id to avoid duplicates.
        """
        signals_and_instruments = self.fetch_large_gain_signals()

        if not signals_and_instruments:
            logger.warning("No data to visualize")
            return

        # Collect stock data with deduplication by instrument_id
        sectors = []
        exchanges = []

        for signal, instrument in signals_and_instruments:
            if not instrument.yfinance_ticker:
                logger.debug(f"Skipping {instrument.ticker} - no yfinance ticker")
                continue

            instrument_id = str(instrument.id)

            # Check if we've already processed this instrument
            if instrument_id in self.distinct_stocks_by_instrument:
                # Skip duplicate - same instrument already processed
                logger.debug(
                    f"Skipping duplicate instrument: {instrument.yfinance_ticker}"
                )
                continue

            # Get sector info (only once per unique instrument)
            sector = self.fetch_sector_info(instrument.yfinance_ticker)

            # Get exchange info
            exchange_name = (
                instrument.exchange.exchange_name if instrument.exchange else "Unknown"
            )

            # Store stock data
            stock_data = {
                "instrument_id": instrument_id,
                "ticker": instrument.short_name or instrument.yfinance_ticker,
                "yf_ticker": instrument.yfinance_ticker,
                "close_price": signal.close_price,
                "sector": sector,
                "exchange": exchange_name,
                "confidence": (
                    signal.confidence_level.value if signal.confidence_level else "N/A"
                ),
                "upside_pct": signal.upside_potential_pct,
            }

            # Add to unique stocks
            self.distinct_stocks_by_instrument[instrument_id] = stock_data

            # Track for distribution counts
            sectors.append(sector)
            exchanges.append(exchange_name)

        # Store distinct stocks as list
        self.large_gain_stocks = list(self.distinct_stocks_by_instrument.values())

        # Count sectors and exchanges
        self.sector_data = dict(Counter(sectors))
        self.exchange_data = dict(Counter(exchanges))

        logger.info(
            f"Prepared data for {len(self.large_gain_stocks)} DISTINCT stocks across "
            f"{len(self.sector_data)} sectors and {len(self.exchange_data)} exchanges"
        )

    def plot_close_prices(self) -> None:
        """
        Create bar chart of close prices for LARGE_GAIN stocks.
        """
        if not self.large_gain_stocks:
            logger.warning("No stock data to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        tickers = [stock["ticker"] for stock in self.large_gain_stocks]
        prices = [stock["close_price"] for stock in self.large_gain_stocks]
        sectors = [stock["sector"] for stock in self.large_gain_stocks]

        # Create color map based on sectors
        unique_sectors = list(set(sectors))
        colors = sns.color_palette("husl", len(unique_sectors))
        sector_color_map = {
            sector: colors[i] for i, sector in enumerate(unique_sectors)
        }
        bar_colors = [sector_color_map[sector] for sector in sectors]

        # Create bar chart
        bars = ax.bar(
            range(len(tickers)), prices, color=bar_colors, alpha=0.8, edgecolor="black"
        )

        # Customize plot
        ax.set_xlabel("Stock Ticker", fontsize=12, fontweight="bold")
        ax.set_ylabel("Close Price ($)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"LARGE_GAIN Stocks - Close Prices ({self.signal_date})\n"
            f"Total: {len(self.large_gain_stocks)} stocks",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha="right")

        # Add price labels on bars
        for i, (bar, price) in enumerate(zip(bars, prices)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"${price:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Add legend for sectors
        legend_elements = [
            mpatches.Rectangle(
                (0, 0), 1, 1, fc=sector_color_map[sector], edgecolor="black", alpha=0.8
            )
            for sector in unique_sectors
        ]
        ax.legend(
            legend_elements,
            unique_sectors,
            loc="upper right",
            title="Sectors",
            fontsize=9,
        )

        plt.tight_layout()
        plt.grid(axis="y", alpha=0.3)

        # Save figure
        filename = f"large_gain_close_prices_{self.signal_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved close price chart: {filename}")

        plt.show()

    def plot_sector_distribution(self) -> None:
        """
        Create pie chart of sector distribution for LARGE_GAIN stocks.
        """
        if not self.sector_data:
            logger.warning("No sector data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        sectors = list(self.sector_data.keys())
        counts = list(self.sector_data.values())

        # Create color palette
        colors = sns.color_palette("husl", len(sectors))

        # Create pie chart
        pie_result = ax.pie(
            counts,
            labels=sectors,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=[0.05] * len(sectors),  # Slightly separate all slices
            shadow=True,
            textprops={"fontsize": 11, "fontweight": "bold"},
        )

        # Enhance text (pie_result contains wedges, texts, and optionally autotexts)
        if len(pie_result) == 3:
            autotexts = pie_result[2]
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontsize(12)
                autotext.set_fontweight("bold")

        ax.set_title(
            f"LARGE_GAIN Stocks - Sector Distribution ({self.signal_date})\n"
            f"Total: {sum(counts)} stocks across {len(sectors)} sectors",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add legend with counts
        legend_labels = [
            f"{sector}: {count} ({count/sum(counts)*100:.1f}%)"
            for sector, count in zip(sectors, counts)
        ]
        ax.legend(
            legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10
        )

        plt.tight_layout()

        # Save figure
        filename = f"large_gain_sector_distribution_{self.signal_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved sector distribution chart: {filename}")

        plt.show()

    def plot_exchange_distribution(self) -> None:
        """
        Create pie chart of exchange (country) distribution for LARGE_GAIN stocks.
        """
        if not self.exchange_data:
            logger.warning("No exchange data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        exchanges = list(self.exchange_data.keys())
        counts = list(self.exchange_data.values())

        # Map exchange names to countries for better display
        exchange_to_country = {
            "NYSE": "USA (NYSE)",
            "NASDAQ": "USA (NASDAQ)",
            "London Stock Exchange": "UK (LSE)",
            "Deutsche Börse Xetra": "Germany (Xetra)",
            "Gettex": "Germany (Gettex)",
            "Euronext Paris": "France (Euronext)",
        }

        # Display names with country mapping
        display_names = [
            exchange_to_country.get(ex, ex) or "Unknown" for ex in exchanges
        ]

        # Ensure all display names are strings (replace None with 'Unknown')
        display_names = [
            name if isinstance(name, str) else "Unknown" for name in display_names
        ]

        # Create color palette
        colors = sns.color_palette("Set2", len(exchanges))

        # Create pie chart
        pie_result = ax.pie(
            counts,
            labels=display_names,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=[0.05] * len(exchanges),  # Slightly separate all slices
            shadow=True,
            textprops={"fontsize": 11, "fontweight": "bold"},
        )

        # Enhance text (pie_result contains wedges, texts, and optionally autotexts)
        if len(pie_result) == 3:
            autotexts = pie_result[2]
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontsize(12)
                autotext.set_fontweight("bold")

        ax.set_title(
            f"LARGE_GAIN Stocks - Exchange Distribution ({self.signal_date})\n"
            f"Total: {sum(counts)} stocks across {len(exchanges)} exchanges",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add legend with counts
        legend_labels = [
            f"{display_names[i]}: {count} ({count/sum(counts)*100:.1f}%)"
            for i, (_, count) in enumerate(zip(exchanges, counts))
        ]
        ax.legend(
            legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10
        )

        plt.tight_layout()

        # Save figure
        filename = f"large_gain_exchange_distribution_{self.signal_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved exchange distribution chart: {filename}")

        plt.show()

    def plot_price_growth_5y(self) -> None:
        """
        Create line chart showing 5-year price growth for each LARGE_GAIN stock.
        """
        if not self.large_gain_stocks:
            logger.warning("No stock data to plot")
            return

        fig, ax = plt.subplots(figsize=(16, 10))

        # Calculate date range: 5 years before signal date
        end_date = self.signal_date
        start_date = end_date - timedelta(days=5 * 365)

        logger.info(f"Fetching 5-year historical data from {start_date} to {end_date}")

        # Fetch and plot historical data for each stock
        colors = sns.color_palette("husl", len(self.large_gain_stocks))
        client = YFinanceClient.get_instance()

        successful_plots = 0
        for i, stock in enumerate(self.large_gain_stocks):
            try:
                # Fetch historical data (convert dates to strings)
                hist = client.fetch_history(
                    stock["yf_ticker"],
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )

                if hist is None or hist.empty:
                    logger.warning(f"No historical data for {stock['ticker']}")
                    continue

                # Normalize prices to show percentage growth from start
                normalized_prices = (hist["Close"] / hist["Close"].iloc[0]) * 100

                # Plot line
                ax.plot(
                    normalized_prices.index,
                    normalized_prices.values,
                    label=f"{stock['ticker']} ({stock['sector']})",
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8,
                )

                successful_plots += 1
                logger.debug(f"Plotted {stock['ticker']}: {len(hist)} data points")

            except Exception as e:
                logger.warning(f"Error fetching data for {stock['ticker']}: {e}")
                continue

        if successful_plots == 0:
            logger.error("No historical data could be fetched for any stock")
            plt.close()
            return

        # Customize plot
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Price Growth (Indexed to 100)", fontsize=12, fontweight="bold")
        ax.set_title(
            f'LARGE_GAIN Stocks - 5-Year Price Growth ({start_date.strftime("%Y-%m-%d")} to {end_date})\n'
            f"Successfully plotted: {successful_plots}/{len(self.large_gain_stocks)} stocks\n"
            f"Prices normalized to 100 at start date",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add horizontal line at 100 (starting point)
        ax.axhline(
            y=100,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Starting Point (100)",
        )

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Legend
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=9,
            ncol=1 if len(self.large_gain_stocks) <= 15 else 2,
        )

        plt.tight_layout()

        # Save figure
        filename = f"large_gain_5y_growth_{self.signal_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved 5-year growth chart: {filename}")

        plt.show()

    def print_summary(self) -> None:
        """
        Print summary statistics of LARGE_GAIN stocks.
        """
        if not self.large_gain_stocks:
            print("\nNo LARGE_GAIN stocks to summarize")
            return

        print("\n" + "=" * 80)
        print(f"LARGE_GAIN STOCKS SUMMARY - {self.signal_date}")
        print("=" * 80)
        print(f"Total stocks:    {len(self.large_gain_stocks)}")
        print(f"Total sectors:   {len(self.sector_data)}")
        print(f"Total exchanges: {len(self.exchange_data)}")
        print()

        print("Top 5 stocks by close price:")
        for i, stock in enumerate(self.large_gain_stocks[:5], 1):
            print(
                f"  {i}. {stock['ticker']:10} ${stock['close_price']:8.2f}  "
                f"Sector: {stock['sector']:20}  "
                f"Exchange: {stock['exchange']:25}  "
                f"Confidence: {stock['confidence']}"
            )

        print()
        print("Sector distribution:")
        for sector, count in sorted(
            self.sector_data.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(self.large_gain_stocks)) * 100
            print(f"  {sector:30} {count:3d} stocks ({percentage:5.1f}%)")

        print()
        print("Exchange distribution:")
        for exchange, count in sorted(
            self.exchange_data.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(self.large_gain_stocks)) * 100
            print(f"  {exchange:30} {count:3d} stocks ({percentage:5.1f}%)")

        print("=" * 80)

    def visualize(self) -> None:
        """
        Run complete visualization pipeline.
        """
        logger.info(f"Starting visualization for {self.signal_date}")

        # Prepare data
        self.prepare_data()

        if not self.large_gain_stocks:
            print(f"\nNo LARGE_GAIN stocks found for {self.signal_date}")
            return

        # Print summary
        self.print_summary()

        # Create visualizations
        self.plot_close_prices()
        self.plot_sector_distribution()
        self.plot_exchange_distribution()
        # self.plot_price_growth_5y()  # Disabled - 5-year historical chart not needed

        logger.info("Visualization complete")


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
    Main function - visualize LARGE_GAIN stocks for today.
    """
    print("=" * 80)
    print("LARGE_GAIN STOCKS VISUALIZATION")
    print("=" * 80)

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

        # Create visualizer
        visualizer = LargeGainVisualizer(signal_date=signal_date)

        # Run visualization
        visualizer.visualize()

        logger.info("Visualization completed successfully")

    except KeyboardInterrupt:
        logger.info("\nVisualization interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
