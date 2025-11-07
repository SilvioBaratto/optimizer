#!/usr/bin/env python3
"""
Portfolio Growth Charts - ASCII Charts for Concentrated Portfolio
==================================================================
Generates ASCII price charts for all stocks in the concentrated portfolio,
plus aggregate portfolio performance.

Features:
- Individual stock performance charts (1-year history)
- Weighted portfolio aggregate performance
- Risk metrics for each position and overall portfolio
- Conviction tier breakdown
- Sector allocation visualization

Usage:
    python src/data_visualization/portfolio_growth_charts.py [--portfolio-file PATH]
"""

import sys
from pathlib import Path
import argparse
import json
import logging
from datetime import date as date_type, timedelta, datetime
from typing import List, Dict, Any, Optional, Tuple
import time

import numpy as np

from dotenv import load_dotenv
load_dotenv()

from src.yfinance import YFinanceClient

# Import database and models
from app.database import database_manager, init_db
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

    # Print y-axis labels and canvas
    for i, row in enumerate(canvas):
        price_at_line = max_price - (i / (height - 1)) * price_range
        writer(f"${price_at_line:6.2f} | {''.join(row)}")

    writer(f"{'':>8} {''.join(['-' for _ in range(width)])}")

    # Print date markers along x-axis
    num_labels = 3
    label_positions = [i * (len(sampled_dates) - 1) // (num_labels - 1) for i in range(num_labels)]
    date_labels = [sampled_dates[pos].strftime('%Y-%m') for pos in label_positions]

    spacing = width // (num_labels - 1)
    date_line = f"{'':>8} "
    for i, label in enumerate(date_labels):
        if i == 0:
            date_line += f"{label}"
        else:
            padding = spacing - len(date_labels[i-1]) // 2 - len(label) // 2
            date_line += f"{' ' * padding}{label}"
    writer(date_line)

    # Print summary
    total_return = ((prices[-1] / prices[0]) - 1) * 100
    writer(f"\nPeak: ${max_price:.2f} ({peak_date.strftime('%Y-%m-%d')})  |  " +
           f"Trough: ${min_price:.2f} ({trough_date.strftime('%Y-%m-%d')})")
    writer(f"Start: ${prices[0]:.2f}  →  End: ${prices[-1]:.2f}  |  " +
           f"Return: {total_return:+.2f}%")


def calculate_risk_metrics(prices, risk_free_rate=0.045):
    """
    Calculate risk metrics.

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

    # Sortino
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


class PortfolioGrowthCharts:
    """
    Generates ASCII charts for concentrated portfolio holdings.
    """

    def __init__(self, portfolio_file: Path):
        """
        Initialize generator.

        Args:
            portfolio_file: Path to portfolio JSON file
        """
        self.portfolio_file = portfolio_file
        self.portfolio_data = None
        self.positions = []
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

    def load_portfolio(self) -> bool:
        """
        Load portfolio data from JSON file.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading portfolio from: {self.portfolio_file}")

            with open(self.portfolio_file, 'r') as f:
                self.portfolio_data = json.load(f)

            self.positions = self.portfolio_data.get('positions', [])

            logger.info(f"Loaded {len(self.positions)} positions from portfolio")
            return True

        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            return False

    def get_yfinance_ticker(self, ticker: str, instrument_id: str) -> Optional[str]:
        """
        Get yfinance ticker for a portfolio position.

        Args:
            ticker: T212 ticker
            instrument_id: Instrument UUID

        Returns:
            yfinance ticker or None
        """
        try:
            from sqlalchemy import select
            from uuid import UUID

            with database_manager.get_session() as session:
                stmt = select(Instrument).where(Instrument.id == UUID(instrument_id))
                instrument = session.execute(stmt).scalar_one_or_none()

                if instrument and instrument.yfinance_ticker:
                    return instrument.yfinance_ticker
                else:
                    logger.warning(f"No yfinance ticker found for {ticker}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching yfinance ticker for {ticker}: {e}")
            return None

    def fetch_historical_data(self, yf_ticker: str, start_date: date_type, end_date: date_type) -> Optional[Tuple[List[float], List[datetime]]]:
        """
        Fetch historical price data from yfinance.

        Args:
            yf_ticker: Yahoo Finance ticker
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (prices, dates) or None if failed
        """
        try:
            client = YFinanceClient.get_instance()
            hist = client.fetch_history(yf_ticker, start=start_date, end=end_date)

            if hist.empty or len(hist) < 10:
                logger.warning(f"Insufficient data for {yf_ticker}")
                return None

            prices = hist['Close'].tolist()
            dates = hist.index.tolist()

            return prices, dates

        except Exception as e:
            logger.error(f"Error fetching data for {yf_ticker}: {e}")
            return None

    def calculate_portfolio_performance(
        self,
        position_returns: Dict[str, Dict[str, Any]],
        start_date: date_type,
        end_date: date_type
    ) -> Optional[Tuple[List[float], List[datetime]]]:
        """
        Calculate weighted portfolio aggregate performance.

        Args:
            position_returns: Dict mapping ticker to {prices, dates, weight}
            start_date: Start date (currently unused - for future filtering)
            end_date: End date (currently unused - for future filtering)

        Returns:
            Tuple of (portfolio_values, dates) or None
        """
        try:
            # Get all unique dates (intersection of all stocks)
            all_dates = None
            for _ticker, data in position_returns.items():
                dates_set = set([d.date() if hasattr(d, 'date') else d for d in data['dates']])
                if all_dates is None:
                    all_dates = dates_set
                else:
                    all_dates = all_dates.intersection(dates_set)

            if not all_dates:
                logger.warning("No common dates across portfolio positions")
                return None

            common_dates = sorted(list(all_dates))

            # Calculate weighted portfolio returns for each date
            portfolio_values = []

            for date in common_dates:
                weighted_return = 0.0

                for _ticker, data in position_returns.items():
                    # Find price on this date
                    date_prices = {d.date() if hasattr(d, 'date') else d: p for d, p in zip(data['dates'], data['prices'])}

                    if date in date_prices:
                        # Calculate return from start
                        start_price = data['prices'][0]
                        current_price = date_prices[date]
                        stock_return = (current_price / start_price - 1)

                        # Weight by position weight
                        weighted_return += stock_return * data['weight']

                # Portfolio value (starting at 100)
                portfolio_value = 100 * (1 + weighted_return)
                portfolio_values.append(portfolio_value)

            # Convert dates back to datetime
            portfolio_dates = [datetime.combine(d, datetime.min.time()) for d in common_dates]

            return portfolio_values, portfolio_dates

        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return None

    def generate_charts(self) -> Optional[str]:
        """
        Generate ASCII charts for all portfolio positions and save to file.

        Returns:
            Path to the generated file or None if failed
        """
        if not self.load_portfolio():
            return None

        # Create output directory
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Create output filename
        portfolio_name = self.portfolio_file.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{portfolio_name}_charts_{timestamp}.txt"

        logger.info(f"Opening output file: {filename}")

        # Open file for writing
        with open(filename, 'w', encoding='utf-8') as f:
            self.output_file = f
            self._generate_charts_internal()
            self.output_file = None

        logger.info(f"Charts saved to: {filename}")
        return str(filename)

    def _generate_charts_internal(self) -> None:
        """
        Internal method that generates charts (writes to self.output_file).
        """
        if not self.positions:
            self._write("\nNo positions found in portfolio")
            return

        # Initialize database
        init_db()

        # Extract metadata
        metadata = self.portfolio_data.get('metadata') or {}
        portfolio_date = metadata.get('creation_date', 'Unknown')
        total_positions = len(self.positions)

        # Calculate date range: 1 year before portfolio creation
        try:
            if isinstance(portfolio_date, str):
                portfolio_date_obj = datetime.fromisoformat(portfolio_date.replace('Z', '+00:00')).date()
            else:
                portfolio_date_obj = date_type.today()
        except:
            portfolio_date_obj = date_type.today()

        end_date = portfolio_date_obj
        start_date = end_date - timedelta(days=365)

        # Print header
        self._write(f"\n{'='*80}")
        self._write(f"CONCENTRATED PORTFOLIO - GROWTH CHARTS")
        self._write(f"{'='*80}")
        self._write(f"\nPortfolio Date: {portfolio_date}")
        self._write(f"Total Positions: {total_positions}")
        self._write(f"Analysis Period: {start_date} to {end_date} (1 year)")
        self._write(f"{'='*80}")

        # Print portfolio summary
        self._print_portfolio_summary()

        # Generate charts for each position
        position_returns = {}
        successful = 0
        failed = 0

        for i, position in enumerate(self.positions, 1):
            ticker = position.get('ticker', 'Unknown')
            company_name = position.get('company_name', ticker)
            weight = position.get('weight', 0.0)
            conviction = position.get('conviction_tier', 'Unknown')
            sector = position.get('sector', 'Unknown')
            industry = position.get('industry', 'Unknown')
            instrument_id = position.get('instrument_id')

            logger.info(f"[{i}/{total_positions}] Processing {ticker}")

            # Get yfinance ticker
            yf_ticker = self.get_yfinance_ticker(ticker, instrument_id)

            if not yf_ticker:
                self._write(f"\n{'='*70}")
                self._write(f"[{i}/{total_positions}] {ticker} - {company_name}")
                self._write(f"Weight: {weight:.1%} | {conviction} | {sector}")
                self._write(f"{'='*70}")
                self._write("  ⚠️  No yfinance ticker mapping found")
                failed += 1
                continue

            # Fetch historical data
            result = self.fetch_historical_data(yf_ticker, start_date, end_date)

            if not result:
                self._write(f"\n{'='*70}")
                self._write(f"[{i}/{total_positions}] {ticker} - {company_name}")
                self._write(f"Weight: {weight:.1%} | {conviction} | {sector}")
                self._write(f"{'='*70}")
                self._write("  ⚠️  Insufficient historical data")
                failed += 1
                continue

            prices, dates = result

            # Store for portfolio aggregate
            position_returns[ticker] = {
                'prices': prices,
                'dates': dates,
                'weight': weight
            }

            # Print position header
            self._write(f"\n{'='*70}")
            self._write(f"[{i}/{total_positions}] {ticker} - {company_name}")
            self._write(f"Weight: {weight:.1%} | Conviction: {conviction} | Sector: {sector}")
            self._write(f"Industry: {industry}")
            self._write(f"{'='*70}")

            # Plot chart
            plot_ascii_graph(ticker, prices, dates, writer=self._write)

            # Calculate and print metrics
            metrics = calculate_risk_metrics(prices)
            print_compact_metrics(metrics, writer=self._write)

            successful += 1

            # Rate limiting
            time.sleep(0.3)

        # Generate aggregate portfolio performance
        self._write(f"\n{'='*80}")
        self._write("AGGREGATE PORTFOLIO PERFORMANCE (WEIGHTED)")
        self._write(f"{'='*80}")

        if position_returns:
            portfolio_result = self.calculate_portfolio_performance(
                position_returns,
                start_date,
                end_date
            )

            if portfolio_result:
                portfolio_values, portfolio_dates = portfolio_result

                plot_ascii_graph("PORTFOLIO", portfolio_values, portfolio_dates, writer=self._write)

                # Calculate portfolio metrics
                portfolio_metrics = calculate_risk_metrics(portfolio_values)
                print_compact_metrics(portfolio_metrics, writer=self._write)
            else:
                self._write("  ⚠️  Could not calculate aggregate portfolio performance")
        else:
            self._write("  ⚠️  No position data available for aggregate calculation")

        # Print detailed statistics
        self._print_detailed_statistics()

        # Print summary
        self._write(f"\n{'='*80}")
        self._write("SUMMARY")
        self._write(f"{'='*80}")
        self._write(f"  Total positions:          {total_positions}")
        self._write(f"  Charts generated:         {successful}")
        self._write(f"  Failed:                   {failed}")
        self._write(f"{'='*80}")

    def _print_portfolio_summary(self) -> None:
        """Print portfolio composition summary."""
        metrics = self.portfolio_data.get('metrics') or {}

        self._write(f"\n{'─'*80}")
        self._write("PORTFOLIO COMPOSITION")
        self._write(f"{'─'*80}")

        # Conviction breakdown
        self._write("\nConviction Tiers:")
        self._write(f"  HIGH:   {metrics.get('high_conviction_count', 0)} positions ({metrics.get('high_conviction_weight', 0):.1%})")
        self._write(f"  MEDIUM: {metrics.get('medium_conviction_count', 0)} positions ({metrics.get('medium_conviction_weight', 0):.1%})")
        self._write(f"  LOW:    {metrics.get('low_conviction_count', 0)} positions ({metrics.get('low_conviction_weight', 0):.1%})")

        # Diversification
        self._write(f"\nDiversification:")
        self._write(f"  Sectors:    {metrics.get('sector_count', 0)}")
        self._write(f"  Industries: {metrics.get('industry_count', 0)}")
        self._write(f"  Countries:  {metrics.get('country_count', 0)}")

        # Risk metrics
        if metrics.get('avg_sharpe_ratio'):
            self._write(f"\nQuality Metrics:")
            self._write(f"  Avg Sharpe:     {metrics.get('avg_sharpe_ratio', 0):.2f}")
            if metrics.get('avg_volatility'):
                self._write(f"  Avg Volatility: {metrics.get('avg_volatility', 0):.1%}")
            if metrics.get('avg_alpha'):
                self._write(f"  Avg Alpha:      {metrics.get('avg_alpha', 0):+.1%}")

        self._write(f"{'─'*80}")

    def _print_detailed_statistics(self) -> None:
        """
        Print detailed portfolio statistics including country, sector, and industry distributions.
        """
        from collections import defaultdict

        # Initialize counters
        country_counts = defaultdict(int)
        country_weights = defaultdict(float)
        sector_counts = defaultdict(int)
        sector_weights = defaultdict(float)
        industry_counts = defaultdict(int)
        industry_weights = defaultdict(float)

        # Collect statistics from positions
        for position in self.positions:
            country = position.get('country', 'Unknown')
            sector = position.get('sector', 'Unknown')
            industry = position.get('industry', 'Unknown')
            weight = position.get('weight', 0.0)

            country_counts[country] += 1
            country_weights[country] += weight

            sector_counts[sector] += 1
            sector_weights[sector] += weight

            industry_counts[industry] += 1
            industry_weights[industry] += weight

        # Print detailed statistics
        self._write(f"\n{'='*80}")
        self._write("DETAILED PORTFOLIO STATISTICS")
        self._write(f"{'='*80}")

        # ══════════════════════════════════════════════════════════════════
        # COUNTRY DISTRIBUTION
        # ══════════════════════════════════════════════════════════════════
        self._write(f"\n{'─'*80}")
        self._write(f"COUNTRY DISTRIBUTION ({len(country_counts)} countries)")
        self._write(f"{'─'*80}")

        # Sort by weight (descending)
        sorted_countries = sorted(
            country_counts.items(),
            key=lambda x: country_weights[x[0]],
            reverse=True
        )

        self._write(f"\n{'Country':<25} {'Stocks':>8} {'Weight':>12}")
        self._write(f"{'-'*25} {'-'*8} {'-'*12}")

        for country, count in sorted_countries:
            weight = country_weights[country]
            self._write(f"{country:<25} {count:>8} {weight:>11.1%}")

        # ══════════════════════════════════════════════════════════════════
        # SECTOR DISTRIBUTION
        # ══════════════════════════════════════════════════════════════════
        self._write(f"\n{'─'*80}")
        self._write(f"SECTOR DISTRIBUTION ({len(sector_counts)} sectors)")
        self._write(f"{'─'*80}")

        # Sort by weight (descending)
        sorted_sectors = sorted(
            sector_counts.items(),
            key=lambda x: sector_weights[x[0]],
            reverse=True
        )

        self._write(f"\n{'Sector':<30} {'Stocks':>8} {'Weight':>12} {'Type':>12}")
        self._write(f"{'-'*30} {'-'*8} {'-'*12} {'-'*12}")

        # Import sector classifications
        from src.risk_management.sector_allocator import DEFENSIVE_SECTORS

        for sector, count in sorted_sectors:
            weight = sector_weights[sector]
            sector_type = "DEFENSIVE" if sector in DEFENSIVE_SECTORS else "CYCLICAL"
            self._write(f"{sector:<30} {count:>8} {weight:>11.1%} {sector_type:>12}")

        # ══════════════════════════════════════════════════════════════════
        # INDUSTRY DISTRIBUTION
        # ══════════════════════════════════════════════════════════════════
        self._write(f"\n{'─'*80}")
        self._write(f"INDUSTRY DISTRIBUTION ({len(industry_counts)} industries)")
        self._write(f"{'─'*80}")

        # Sort by weight (descending), then alphabetically
        sorted_industries = sorted(
            industry_counts.items(),
            key=lambda x: (-industry_weights[x[0]], x[0])
        )

        self._write(f"\n{'Industry':<45} {'Stocks':>8} {'Weight':>12}")
        self._write(f"{'-'*45} {'-'*8} {'-'*12}")

        for industry, count in sorted_industries:
            weight = industry_weights[industry]
            self._write(f"{industry:<45} {count:>8} {weight:>11.1%}")

        # ══════════════════════════════════════════════════════════════════
        # SUMMARY STATISTICS
        # ══════════════════════════════════════════════════════════════════
        self._write(f"\n{'─'*80}")
        self._write("DIVERSIFICATION SUMMARY")
        self._write(f"{'─'*80}")

        # Find most concentrated areas
        top_country = max(country_counts.items(), key=lambda x: country_weights[x[0]])
        top_sector = max(sector_counts.items(), key=lambda x: sector_weights[x[0]])

        # Count defensive vs cyclical
        defensive_count = sum(count for sector, count in sector_counts.items() if sector in DEFENSIVE_SECTORS)
        cyclical_count = sum(count for sector, count in sector_counts.items() if sector not in DEFENSIVE_SECTORS and sector != 'Unknown')

        defensive_weight = sum(sector_weights[sector] for sector in sector_weights if sector in DEFENSIVE_SECTORS)
        cyclical_weight = sum(sector_weights[sector] for sector in sector_weights if sector not in DEFENSIVE_SECTORS and sector != 'Unknown')

        self._write(f"\nGeographic Concentration:")
        self._write(f"  Largest country:      {top_country[0]} ({top_country[1]} stocks, {country_weights[top_country[0]]:.1%})")
        self._write(f"  Total countries:      {len(country_counts)}")

        self._write(f"\nSector Balance:")
        self._write(f"  Defensive:            {defensive_count} stocks ({defensive_weight:.1%})")
        self._write(f"  Cyclical:             {cyclical_count} stocks ({cyclical_weight:.1%})")
        self._write(f"  Largest sector:       {top_sector[0]} ({top_sector[1]} stocks, {sector_weights[top_sector[0]]:.1%})")

        self._write(f"\nIndustry Concentration:")
        self._write(f"  Total industries:     {len(industry_counts)}")
        self._write(f"  Avg stocks/industry:  {len(self.positions) / len(industry_counts):.1f}")

        # Check if any industry has more than 2 stocks (violation)
        over_concentrated = [(ind, cnt) for ind, cnt in industry_counts.items() if cnt > 2]
        if over_concentrated:
            self._write(f"\n  ⚠️  Industries with >2 stocks:")
            for ind, cnt in over_concentrated:
                self._write(f"      - {ind}: {cnt} stocks ({industry_weights[ind]:.1%})")
        else:
            self._write(f"  ✅ No industry has more than 2 stocks (well diversified)")

        self._write(f"{'─'*80}")


def find_latest_portfolio() -> Optional[Path]:
    """
    Find the most recent portfolio JSON file in the data/portfolios directory.

    Returns:
        Path to the latest portfolio file or None
    """
    # Portfolio files are saved to project_root/data/portfolios by concentrated_portfolio_builder.py
    portfolio_dir = project_root / "data" / "portfolios"

    if not portfolio_dir.exists():
        logger.warning(f"Portfolio directory not found: {portfolio_dir}")
        logger.info("Attempting to create directory...")
        try:
            portfolio_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {portfolio_dir}")
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return None

    # Find all portfolio JSON files (any name pattern)
    portfolio_files = list(portfolio_dir.glob("*.json"))

    if not portfolio_files:
        logger.warning(f"No portfolio files found in: {portfolio_dir}")
        logger.info("Please run concentrated_portfolio_builder.py first to generate a portfolio.")
        return None

    # Sort by modification time (most recent first)
    latest_file = max(portfolio_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Found latest portfolio: {latest_file}")
    return latest_file


def main():
    """
    Main function - generate ASCII charts for portfolio holdings.
    """
    parser = argparse.ArgumentParser(
        description="Generate growth charts for concentrated portfolio"
    )
    parser.add_argument(
        '--portfolio-file',
        type=Path,
        help='Path to portfolio JSON file (default: latest in portfolios/)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PORTFOLIO GROWTH CHARTS GENERATOR")
    print("=" * 80)

    try:
        # Get portfolio file
        if args.portfolio_file:
            portfolio_file = args.portfolio_file
        else:
            portfolio_file = find_latest_portfolio()

        if not portfolio_file or not portfolio_file.exists():
            logger.error("No portfolio file found. Run concentrated_portfolio_builder.py first.")
            sys.exit(1)

        logger.info(f"Using portfolio file: {portfolio_file}")

        # Create generator
        generator = PortfolioGrowthCharts(portfolio_file)

        # Generate charts
        output_file = generator.generate_charts()

        if output_file:
            # Print success message
            print(f"\n{'='*80}")
            print(f"✅ Charts successfully generated!")
            print(f"{'='*80}")
            print(f"Output file: {output_file}")
            print(f"{'='*80}")
            logger.info(f"Chart generation completed successfully: {output_file}")
        else:
            logger.error("Chart generation failed")
            sys.exit(1)

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
