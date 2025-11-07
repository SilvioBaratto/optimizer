#!/usr/bin/env python3
"""
Large Gain Stock Charts - ASCII Price Charts for Top Performers
================================================================
Generates ASCII price charts for all LARGE_GAIN stocks showing 1-year performance.

Features:
- ASCII charts with arrows showing price movement
- Key metrics: total return, volatility, Sharpe ratio, max drawdown
- Risk-adjusted performance metrics
- Only shows last 1 year of data for each stock

Usage:
    python src/data_visualization/large_gain_charts.py
"""

import sys
import logging
from datetime import date as date_type, timedelta
from typing import List, Tuple
import time

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from dotenv import load_dotenv
load_dotenv()

from src.yfinance import YFinanceClient

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


def plot_ascii_graph(ticker: str, prices, dates, width=70, height=15, writer=None):
    """
    Plot compact ASCII graph using arrows and characters.

    Args:
        ticker: Stock ticker symbol
        prices: List of prices
        dates: List of dates corresponding to prices
        width: Width of the graph in characters
        height: Height of the graph in lines
        writer: Optional function to write output (defaults to print)
    """
    if writer is None:
        writer = print

    if len(prices) < 2:
        writer(f"  {ticker}: Not enough data to plot")
        return

    # Normalize prices to fit in height
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price

    if price_range == 0:
        price_range = 1

    # Find peak and trough
    peak_idx = prices.index(max_price)
    trough_idx = prices.index(min_price)
    peak_date = dates[peak_idx]
    trough_date = dates[trough_idx]

    # Downsample if we have more data points than width
    step = max(1, len(prices) // width)
    sampled_prices = prices[::step][:width]
    sampled_dates = dates[::step][:width]

    # Normalize to height
    normalized = [int((p - min_price) / price_range * (height - 1)) for p in sampled_prices]

    # Create canvas
    canvas = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw the line with arrows
    for i in range(len(normalized) - 1):
        y1 = height - 1 - normalized[i]
        y2 = height - 1 - normalized[i + 1]

        # Draw vertical connection
        if y1 < y2:  # Going down
            for y in range(y1, y2 + 1):
                canvas[y][i] = '↓' if y == y2 else '|'
        elif y1 > y2:  # Going up
            for y in range(y2, y1 + 1):
                canvas[y][i] = '↑' if y == y2 else '|'
        else:  # Horizontal
            canvas[y1][i] = '→'

    # Print the graph
    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')

    writer(f"\n{'='*width}")
    writer(f"{ticker}: {start_date} to {end_date}")
    writer(f"Range: ${min_price:.2f} - ${max_price:.2f}")
    writer(f"{'='*width}")

    # Print y-axis labels and canvas (compact - no blank line)
    for i, row in enumerate(canvas):
        price_at_line = max_price - (i / (height - 1)) * price_range
        writer(f"${price_at_line:6.2f} | {''.join(row)}")

    writer(f"{'':>8} {''.join(['-' for _ in range(width)])}")

    # Print date markers along x-axis (only 3 labels for compactness)
    num_labels = 3
    label_positions = [i * (len(sampled_dates) - 1) // (num_labels - 1) for i in range(num_labels)]
    date_labels = [sampled_dates[pos].strftime('%Y-%m') for pos in label_positions]

    # Build x-axis dates string
    spacing = width // (num_labels - 1)
    date_line = f"{'':>8} "
    for i, label in enumerate(date_labels):
        if i == 0:
            date_line += f"{label}"
        else:
            padding = spacing - len(date_labels[i-1]) // 2 - len(label) // 2
            date_line += f"{' ' * padding}{label}"
    writer(date_line)

    # Print compact summary
    total_return = ((prices[-1] / prices[0]) - 1) * 100
    writer(f"\nPeak: ${max_price:.2f} ({peak_date.strftime('%Y-%m-%d')})  |  " +
           f"Trough: ${min_price:.2f} ({trough_date.strftime('%Y-%m-%d')})")
    writer(f"Start: ${prices[0]:.2f}  →  End: ${prices[-1]:.2f}  |  " +
           f"Return: {total_return:+.2f}%")


def calculate_risk_metrics(prices, risk_free_rate=0.045):
    """
    Calculate compact risk metrics.

    Args:
        prices: List of prices
        risk_free_rate: Annual risk-free rate (default: 4.5%)

    Returns:
        Dictionary of risk metrics
    """
    if len(prices) < 2:
        return None

    returns = np.diff(prices) / prices[:-1]

    # Annualized metrics (assuming 252 trading days)
    total_return = (prices[-1] / prices[0] - 1)
    years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Downside deviation (for Sortino ratio)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) * np.sqrt(252) if len(downside_returns) > 1 else 0
    sortino = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
    }


def print_compact_metrics(metrics, width=70, writer=None):
    """Print compact risk metrics."""
    if writer is None:
        writer = print

    if not metrics:
        writer("  Insufficient data for metrics")
        return

    writer(f"{'─'*width}")
    writer(f"Return: {metrics['total_return']*100:>6.2f}% (Total)  |  " +
           f"{metrics['annualized_return']*100:>6.2f}% (Annualized)")
    writer(f"Risk:   {metrics['volatility']*100:>6.2f}% (Volatility)  |  " +
           f"{metrics['max_drawdown']*100:>6.2f}% (Max Drawdown)")
    writer(f"Ratios: {metrics['sharpe_ratio']:>6.2f} (Sharpe)  |  " +
           f"{metrics['sortino_ratio']:>6.2f} (Sortino)")
    writer(f"{'─'*width}")


class LargeGainChartsGenerator:
    """
    Generates ASCII charts for all LARGE_GAIN stocks.
    """

    def __init__(self, signal_date: date_type, max_stocks: int | None = None):
        """
        Initialize generator.

        Args:
            signal_date: Date to analyze (required)
            max_stocks: Maximum number of stocks to chart (None = all)
        """
        self.signal_date = signal_date
        self.max_stocks = max_stocks
        self.large_gain_stocks = []
        self.output_file = None

    def _write(self, text: str = "") -> None:
        """
        Write text to both console and file.

        Args:
            text: Text to write (default: empty line)
        """
        print(text)
        if self.output_file:
            self.output_file.write(text + '\n')

    def fetch_large_gain_signals(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Fetch LARGE_GAIN signals for the target date.

        Returns:
            List of tuples (StockSignal, Instrument)
        """
        logger.info(f"Fetching LARGE_GAIN signals for {self.signal_date}")

        with database_manager.get_session() as session:
            query = select(StockSignal).where(
                StockSignal.signal_date == self.signal_date,
                StockSignal.signal_type == SignalEnum.LARGE_GAIN
            ).options(
                joinedload(StockSignal.instrument).joinedload(Instrument.exchange)
            ).order_by(
                StockSignal.close_price.desc()
            )

            if self.max_stocks:
                query = query.limit(self.max_stocks)

            signals = session.execute(query).scalars().all()

            if not signals:
                logger.warning(f"No LARGE_GAIN signals found for {self.signal_date}")
                return []

            results = [(signal, signal.instrument) for signal in signals]
            logger.info(f"Found {len(results)} LARGE_GAIN signals")

            return results

    def generate_charts(self) -> str:
        """
        Generate ASCII charts for all LARGE_GAIN stocks and save to file.

        Returns:
            Path to the generated file
        """
        # Create output directory if it doesn't exist
        from pathlib import Path
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Create output filename in the output directory
        filename = output_dir / f"large_gain_charts_{self.signal_date}.txt"

        logger.info(f"Opening output file: {filename}")

        # Open file for writing
        with open(filename, 'w', encoding='utf-8') as f:
            self.output_file = f
            result = self._generate_charts_internal()
            self.output_file = None

        logger.info(f"Charts saved to: {filename}")
        return str(filename)

    def _generate_charts_internal(self) -> None:
        """
        Internal method that generates charts (writes to self.output_file).
        """
        signals_and_instruments = self.fetch_large_gain_signals()

        if not signals_and_instruments:
            self._write(f"\nNo LARGE_GAIN stocks found for {self.signal_date}")
            return

        # Track unique instruments
        seen_instruments = set()
        stocks_to_chart = []

        for signal, instrument in signals_and_instruments:
            if not instrument or not instrument.yfinance_ticker:
                continue

            instrument_id = str(instrument.id)
            if instrument_id in seen_instruments:
                continue

            seen_instruments.add(instrument_id)
            stocks_to_chart.append({
                'ticker': instrument.yfinance_ticker,
                'name': instrument.short_name or instrument.ticker,
                'sector': signal.sector or 'Unknown',
                'exchange': signal.exchange_name or 'Unknown',
            })

        logger.info(f"Generating charts for {len(stocks_to_chart)} unique LARGE_GAIN stocks")

        # Calculate date range: 1 year before signal date
        end_date = self.signal_date
        start_date = end_date - timedelta(days=365)

        self._write(f"\n{'='*80}")
        self._write(f"LARGE_GAIN STOCKS - 1-YEAR PRICE CHARTS ({self.signal_date})")
        self._write(f"{'='*80}")
        self._write(f"\nTotal stocks: {len(stocks_to_chart)}")
        self._write(f"Period: {start_date} to {end_date} (1 year)")
        self._write(f"{'='*80}")

        successful = 0
        failed = 0

        client = YFinanceClient.get_instance()

        for i, stock in enumerate(stocks_to_chart, 1):
            try:
                logger.info(f"[{i}/{len(stocks_to_chart)}] Fetching data for {stock['ticker']}")

                # Fetch historical data (convert dates to strings)
                hist = client.fetch_history(
                    stock['ticker'],
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )

                if hist is None or hist.empty or len(hist) < 10:
                    logger.warning(f"Insufficient data for {stock['ticker']}")
                    self._write(f"\n{'='*70}")
                    self._write(f"[{i}/{len(stocks_to_chart)}] {stock['ticker']} - {stock['name']}")
                    self._write(f"Sector: {stock['sector']} | Exchange: {stock['exchange']}")
                    self._write(f"{'='*70}")
                    self._write("  ⚠️  Insufficient historical data")
                    failed += 1
                    continue

                prices = hist['Close'].tolist()
                dates = hist.index.tolist()

                # Print stock header
                self._write(f"\n{'='*70}")
                self._write(f"[{i}/{len(stocks_to_chart)}] {stock['ticker']} - {stock['name']}")
                self._write(f"Sector: {stock['sector']} | Exchange: {stock['exchange']}")
                self._write(f"{'='*70}")

                # Plot chart
                plot_ascii_graph(stock['ticker'], prices, dates, writer=self._write)

                # Calculate and print metrics
                metrics = calculate_risk_metrics(prices)
                print_compact_metrics(metrics, writer=self._write)

                successful += 1

                # Rate limiting: sleep briefly to avoid overwhelming yfinance
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error generating chart for {stock['ticker']}: {e}")
                self._write(f"\n{'='*70}")
                self._write(f"[{i}/{len(stocks_to_chart)}] {stock['ticker']} - {stock['name']}")
                self._write(f"{'='*70}")
                self._write(f"  ❌ Error: {str(e)}")
                failed += 1
                continue

        # Print summary
        self._write(f"\n{'='*80}")
        self._write("SUMMARY")
        self._write(f"{'='*80}")
        self._write(f"  Total LARGE_GAIN stocks:  {len(stocks_to_chart)}")
        self._write(f"  Charts generated:         {successful}")
        self._write(f"  Failed:                   {failed}")
        self._write(f"{'='*80}")


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
    Main function - generate ASCII charts for all LARGE_GAIN stocks.
    """
    print("=" * 80)
    print("LARGE_GAIN STOCK CHARTS GENERATOR")
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

        # Optional: Limit number of stocks for testing
        # Set to None to chart all LARGE_GAIN stocks
        max_stocks = None  # Change to e.g., 10 for testing

        # Create generator
        generator = LargeGainChartsGenerator(
            signal_date=signal_date,
            max_stocks=max_stocks
        )

        # Generate charts and save to file
        output_file = generator.generate_charts()

        # Print success message
        print(f"\n{'='*80}")
        print(f"✅ Charts successfully generated!")
        print(f"{'='*80}")
        print(f"Output file: {output_file}")
        print(f"{'='*80}")

        logger.info(f"Chart generation completed successfully: {output_file}")

    except KeyboardInterrupt:
        logger.info("\nChart generation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
