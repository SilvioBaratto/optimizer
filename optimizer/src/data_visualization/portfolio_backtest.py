#!/usr/bin/env python3
"""
Portfolio Backtesting - Compare Portfolio vs S&P 500
=====================================================

Backtests a saved portfolio against the S&P 500 benchmark over the last 5 years.

Features:
- Fetches portfolio from database
- Downloads historical prices for all positions
- Calculates portfolio returns using optimized weights
- Compares against S&P 500 (^GSPC)
- Generates performance metrics and visualization charts

Usage:
    python src/data_visualization/portfolio_backtest.py [portfolio_id]

    If no portfolio_id provided, uses the most recent portfolio.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import select

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.database import init_db, database_manager
from app.models.portfolio import Portfolio, PortfolioPosition
from src.yfinance.client import YFinanceClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_portfolio(portfolio_id: str = None) -> Tuple[Portfolio, List[PortfolioPosition]]:
    """
    Fetch portfolio and positions from database.

    Args:
        portfolio_id: UUID of portfolio (None = most recent)

    Returns:
        Tuple of (Portfolio, List[PortfolioPosition])
    """
    with database_manager.get_session() as session:
        if portfolio_id:
            stmt = select(Portfolio).where(Portfolio.id == uuid.UUID(portfolio_id))
            portfolio = session.execute(stmt).scalar_one()
        else:
            # Get most recent portfolio
            stmt = select(Portfolio).order_by(Portfolio.created_at.desc()).limit(1)
            portfolio = session.execute(stmt).scalar_one_or_none()

            if not portfolio:
                raise ValueError("No portfolios found in database")

        # Fetch positions
        positions_stmt = select(PortfolioPosition).where(
            PortfolioPosition.portfolio_id == portfolio.id
        ).order_by(PortfolioPosition.weight.desc())

        positions = session.execute(positions_stmt).scalars().all()

        # Convert to lists to avoid detached instance errors
        return portfolio, list(positions)


def fetch_historical_prices(
    positions: List[PortfolioPosition],
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical prices for all positions.

    Args:
        positions: List of portfolio positions
        start_date: Start date for historical data
        end_date: End date for historical data

    Returns:
        DataFrame with adjusted close prices (dates × tickers)
    """
    logger.info(f"Fetching historical prices from {start_date.date()} to {end_date.date()}")

    yf_client = YFinanceClient.get_instance()

    prices = {}
    failed_tickers = []

    for position in positions:
        # Skip zero-weight positions
        if float(position.weight) == 0:
            continue

        yf_ticker = position.yfinance_ticker
        if not yf_ticker:
            logger.warning(f"No yfinance ticker for {position.ticker}, skipping")
            failed_tickers.append(position.ticker)
            continue

        try:
            logger.info(f"Fetching {yf_ticker}...")

            # Calculate days to fetch (with buffer)
            days = (end_date - start_date).days + 100

            data = yf_client.fetch_history(
                yf_ticker,
                period=f"{days}d",
                min_rows=int(days * 0.5)
            )

            if data is None or data.empty:
                logger.warning(f"No data returned for {yf_ticker}")
                failed_tickers.append(position.ticker)
                continue

            # Use adjusted close
            if 'Adj Close' in data.columns:
                prices[position.ticker] = data['Adj Close']
            else:
                prices[position.ticker] = data['Close']

        except Exception as e:
            logger.warning(f"Failed to fetch {yf_ticker}: {e}")
            failed_tickers.append(position.ticker)

    if not prices:
        raise ValueError("Failed to fetch any price data")

    # Create DataFrame
    price_df = pd.DataFrame(prices)

    # Make start_date and end_date timezone-aware to match price data
    import pytz
    if price_df.index.tz is not None:
        # Price data is timezone-aware, convert comparison dates
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)

    # Filter to date range
    price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]

    # Forward fill missing data (max 5 days)
    price_df = price_df.ffill(limit=5)

    # Drop rows with any NaN
    price_df = price_df.dropna()

    logger.info(f"✓ Fetched prices: {len(price_df)} days × {len(price_df.columns)} stocks")

    if failed_tickers:
        logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

    return price_df


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    positions: List[PortfolioPosition]
) -> pd.Series:
    """
    Calculate portfolio returns using position weights.

    Args:
        prices: DataFrame of historical prices
        positions: List of portfolio positions with weights

    Returns:
        Series of portfolio returns (indexed by date)
    """
    # Create weight mapping
    weights = {}
    for position in positions:
        if position.ticker in prices.columns and float(position.weight) > 0:
            weights[position.ticker] = float(position.weight)

    # Normalize weights (in case some positions are missing)
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {ticker: w / total_weight for ticker, w in weights.items()}

    logger.info(f"Portfolio weights (normalized):")
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {ticker:15s}: {weight:.2%}")

    # Calculate daily returns for each stock
    returns = prices.pct_change()

    # Calculate weighted portfolio returns
    portfolio_returns = pd.Series(0.0, index=returns.index)

    for ticker, weight in weights.items():
        if ticker in returns.columns:
            portfolio_returns += weight * returns[ticker]

    # Drop first row (NaN from pct_change)
    portfolio_returns = portfolio_returns.dropna()

    logger.info(f"✓ Calculated portfolio returns: {len(portfolio_returns)} days")

    return portfolio_returns


def fetch_benchmark_returns(
    start_date: datetime,
    end_date: datetime,
    benchmark_ticker: str = "^GSPC"
) -> pd.Series:
    """
    Fetch benchmark (S&P 500) returns.

    Args:
        start_date: Start date
        end_date: End date
        benchmark_ticker: Benchmark ticker (default: ^GSPC for S&P 500)

    Returns:
        Series of benchmark returns
    """
    logger.info(f"Fetching benchmark ({benchmark_ticker})...")

    yf_client = YFinanceClient.get_instance()

    days = (end_date - start_date).days + 100

    data = yf_client.fetch_history(
        benchmark_ticker,
        period=f"{days}d",
        min_rows=int(days * 0.5)
    )

    if data is None or data.empty:
        raise ValueError(f"Failed to fetch benchmark data for {benchmark_ticker}")

    # Use adjusted close
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']

    # Make dates timezone-aware if needed
    import pytz
    if prices.index.tz is not None:
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)

    # Filter to date range
    prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]

    # Calculate returns
    returns = prices.pct_change().dropna()

    logger.info(f"✓ Fetched benchmark returns: {len(returns)} days")

    return returns


def calculate_metrics(returns: pd.Series, name: str = "Portfolio") -> Dict:
    """
    Calculate performance metrics.

    Args:
        returns: Series of daily returns
        name: Name for logging

    Returns:
        Dictionary of metrics
    """
    # Cumulative return
    cumulative_return = (1 + returns).cumprod().iloc[-1] - 1

    # Annualized return (assuming 252 trading days)
    n_years = len(returns) / 252
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1

    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming 4% risk-free rate)
    risk_free_rate = 0.04
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    metrics = {
        'name': name,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_days': len(returns)
    }

    return metrics


def plot_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str,
    output_path: str
):
    """
    Create performance comparison charts.

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
        portfolio_name: Name of portfolio
        output_path: Path to save chart
    """
    # Align returns to common dates
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns[common_dates]
    benchmark_returns = benchmark_returns[common_dates]

    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Portfolio Backtest: {portfolio_name} vs S&P 500', fontsize=16, fontweight='bold')

    # 1. Cumulative returns
    ax1 = axes[0]
    ax1.plot(portfolio_cumulative.index, portfolio_cumulative.values,
             label='Portfolio', linewidth=2, color='#2E86AB')
    ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values,
             label='S&P 500', linewidth=2, color='#A23B72', linestyle='--')
    ax1.set_ylabel('Cumulative Return (1 = 0%)', fontsize=11)
    ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 2. Rolling 252-day volatility
    ax2 = axes[1]
    portfolio_vol = portfolio_returns.rolling(252).std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.rolling(252).std() * np.sqrt(252)

    ax2.plot(portfolio_vol.index, portfolio_vol.values,
             label='Portfolio', linewidth=2, color='#2E86AB')
    ax2.plot(benchmark_vol.index, benchmark_vol.values,
             label='S&P 500', linewidth=2, color='#A23B72', linestyle='--')
    ax2.set_ylabel('Annualized Volatility', fontsize=11)
    ax2.set_title('Rolling 1-Year Volatility', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 3. Drawdown
    ax3 = axes[2]
    portfolio_cumulative_full = (1 + portfolio_returns).cumprod()
    benchmark_cumulative_full = (1 + benchmark_returns).cumprod()

    portfolio_drawdown = (portfolio_cumulative_full - portfolio_cumulative_full.expanding().max()) / portfolio_cumulative_full.expanding().max()
    benchmark_drawdown = (benchmark_cumulative_full - benchmark_cumulative_full.expanding().max()) / benchmark_cumulative_full.expanding().max()

    ax3.fill_between(portfolio_drawdown.index, 0, portfolio_drawdown.values,
                      label='Portfolio', alpha=0.5, color='#2E86AB')
    ax3.fill_between(benchmark_drawdown.index, 0, benchmark_drawdown.values,
                      label='S&P 500', alpha=0.3, color='#A23B72')
    ax3.set_ylabel('Drawdown', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved chart: {output_path}")

    plt.close()


def print_metrics_comparison(portfolio_metrics: Dict, benchmark_metrics: Dict):
    """Print performance metrics comparison."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON (5-Year Backtest)")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Portfolio':>15} {'S&P 500':>15} {'Difference':>15}")
    print("-" * 80)

    # Cumulative return
    print(f"{'Total Return':<30} "
          f"{portfolio_metrics['cumulative_return']:>14.2%} "
          f"{benchmark_metrics['cumulative_return']:>14.2%} "
          f"{portfolio_metrics['cumulative_return'] - benchmark_metrics['cumulative_return']:>+14.2%}")

    # Annualized return
    print(f"{'Annualized Return':<30} "
          f"{portfolio_metrics['annualized_return']:>14.2%} "
          f"{benchmark_metrics['annualized_return']:>14.2%} "
          f"{portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return']:>+14.2%}")

    # Volatility
    print(f"{'Volatility (Annual)':<30} "
          f"{portfolio_metrics['volatility']:>14.2%} "
          f"{benchmark_metrics['volatility']:>14.2%} "
          f"{portfolio_metrics['volatility'] - benchmark_metrics['volatility']:>+14.2%}")

    # Sharpe ratio
    print(f"{'Sharpe Ratio':<30} "
          f"{portfolio_metrics['sharpe_ratio']:>14.2f} "
          f"{benchmark_metrics['sharpe_ratio']:>14.2f} "
          f"{portfolio_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']:>+14.2f}")

    # Sortino ratio
    print(f"{'Sortino Ratio':<30} "
          f"{portfolio_metrics['sortino_ratio']:>14.2f} "
          f"{benchmark_metrics['sortino_ratio']:>14.2f} "
          f"{portfolio_metrics['sortino_ratio'] - benchmark_metrics['sortino_ratio']:>+14.2f}")

    # Max drawdown
    print(f"{'Max Drawdown':<30} "
          f"{portfolio_metrics['max_drawdown']:>14.2%} "
          f"{benchmark_metrics['max_drawdown']:>14.2%} "
          f"{portfolio_metrics['max_drawdown'] - benchmark_metrics['max_drawdown']:>+14.2%}")

    # Win rate
    print(f"{'Win Rate (Daily)':<30} "
          f"{portfolio_metrics['win_rate']:>14.2%} "
          f"{benchmark_metrics['win_rate']:>14.2%} "
          f"{portfolio_metrics['win_rate'] - benchmark_metrics['win_rate']:>+14.2%}")

    print("\n" + "=" * 80)

    # Conclusion
    alpha = portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return']

    if alpha > 0:
        print(f"✓ Portfolio OUTPERFORMED S&P 500 by {alpha:+.2%} annually")
    else:
        print(f"✗ Portfolio UNDERPERFORMED S&P 500 by {alpha:+.2%} annually")

    print("=" * 80)


def main(portfolio_id: str = None):
    """Run backtest."""
    print("\n" + "=" * 80)
    print("PORTFOLIO BACKTESTING")
    print("=" * 80)

    # Initialize database
    init_db()

    # Fetch portfolio
    logger.info("Fetching portfolio from database...")
    portfolio, positions = fetch_portfolio(portfolio_id)

    print(f"\nPortfolio: {portfolio.name}")
    print(f"  ID: {portfolio.id}")
    print(f"  Date: {portfolio.portfolio_date}")
    print(f"  Positions: {len(positions)} ({portfolio.total_positions} non-zero)")
    print(f"  Total Weight: {float(portfolio.total_weight):.2%}")

    # Define backtest period (5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365 + 50)  # 5 years + buffer

    print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")

    # Fetch historical prices
    prices = fetch_historical_prices(positions, start_date, end_date)

    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(prices, positions)

    # Fetch benchmark returns
    benchmark_returns = fetch_benchmark_returns(start_date, end_date)

    # Align to common dates before calculating metrics
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns_aligned = portfolio_returns[common_dates]
    benchmark_returns_aligned = benchmark_returns[common_dates]

    logger.info(f"\n✓ Aligned to common dates: {len(common_dates)} trading days")
    logger.info(f"  Date range: {common_dates[0].date()} to {common_dates[-1].date()}")

    # Calculate metrics on aligned returns
    portfolio_metrics = calculate_metrics(portfolio_returns_aligned, "Portfolio")
    benchmark_metrics = calculate_metrics(benchmark_returns_aligned, "S&P 500")

    # Print comparison
    print_metrics_comparison(portfolio_metrics, benchmark_metrics)

    # Create chart (use aligned returns)
    output_path = f"outputs/portfolio_backtest_{portfolio.id}.png"
    plot_performance(portfolio_returns_aligned, benchmark_returns_aligned, portfolio.name, output_path)

    print(f"\n✓ Backtest complete!")
    print(f"  Chart saved: {output_path}")


if __name__ == "__main__":
    portfolio_id = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        main(portfolio_id)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
