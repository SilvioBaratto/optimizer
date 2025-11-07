#!/usr/bin/env python3
"""
Position Sizing Calculator - Convert Portfolio Weights to Trading Plan
=======================================================================

Takes a saved portfolio and investment amount, returns exact position sizes:
- Amount in EUR to invest per stock
- Number of shares to buy
- Total invested amount
- Remaining cash

Usage:
    python src/black_litterman/position_sizer.py --capital 1000 [--portfolio_id <uuid>]

Example:
    python src/black_litterman/position_sizer.py --capital 1000
    python src/black_litterman/position_sizer.py --capital 5000 --portfolio_id d7e1ffaa-87af-466e-908a-bb9825eef5d2
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
from decimal import Decimal

import pandas as pd
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

        # Fetch positions (non-zero weights only)
        positions_stmt = select(PortfolioPosition).where(
            PortfolioPosition.portfolio_id == portfolio.id,
            PortfolioPosition.weight > 0
        ).order_by(PortfolioPosition.weight.desc())

        positions = session.execute(positions_stmt).scalars().all()

        return portfolio, list(positions)


def fetch_current_prices(positions: List[PortfolioPosition]) -> Dict[str, float]:
    """
    Fetch current stock prices.

    Args:
        positions: List of portfolio positions

    Returns:
        Dictionary mapping ticker -> current price in local currency
    """
    logger.info("Fetching current prices...")

    yf_client = YFinanceClient.get_instance()
    prices = {}
    failed_tickers = []

    for position in positions:
        yf_ticker = position.yfinance_ticker
        if not yf_ticker:
            logger.warning(f"No yfinance ticker for {position.ticker}, skipping")
            failed_tickers.append(position.ticker)
            continue

        try:
            # Fetch latest data (1 day)
            data = yf_client.fetch_history(yf_ticker, period="1d", min_rows=1)

            if data is None or data.empty:
                logger.warning(f"No data returned for {yf_ticker}")
                failed_tickers.append(position.ticker)
                continue

            # Get most recent close price
            if 'Close' in data.columns:
                current_price = data['Close'].iloc[-1]
                prices[position.ticker] = float(current_price)
                logger.info(f"  {position.ticker:15s} ({yf_ticker:10s}): {current_price:>10.2f}")
            else:
                logger.warning(f"No Close price for {yf_ticker}")
                failed_tickers.append(position.ticker)

        except Exception as e:
            logger.warning(f"Failed to fetch price for {yf_ticker}: {e}")
            failed_tickers.append(position.ticker)

    if failed_tickers:
        logger.warning(f"Failed to fetch prices for {len(failed_tickers)} tickers: {failed_tickers}")

    logger.info(f"✓ Fetched prices for {len(prices)} stocks")

    return prices


def calculate_position_sizes(
    positions: List[PortfolioPosition],
    prices: Dict[str, float],
    total_capital: float,
    min_position_value: float = 10.0  # Minimum position size (EUR)
) -> pd.DataFrame:
    """
    Calculate exact position sizes using portfolio weights.

    Since Trading212 allows fractional shares, we simply allocate capital
    according to the exact portfolio weights.

    Args:
        positions: Portfolio positions with weights
        prices: Current prices per ticker
        total_capital: Total investment capital (EUR)
        min_position_value: Minimum position value (default 10 EUR)

    Returns:
        DataFrame with position sizing details
    """
    logger.info(f"\nCalculating position sizes for €{total_capital:,.2f} capital...")

    sizing_data = []

    for position in positions:
        ticker = position.ticker
        weight = float(position.weight)

        # Skip zero-weight positions
        if weight == 0:
            continue

        # Skip if no price available
        if ticker not in prices:
            logger.warning(f"No price available for {ticker}, skipping")
            continue

        price = prices[ticker]

        # Calculate exact amounts using portfolio weight
        target_amount = total_capital * weight
        actual_amount = target_amount  # With fractional shares, we can achieve exact allocation
        shares = actual_amount / price  # Can be fractional

        # Skip positions below minimum value threshold
        if actual_amount < min_position_value:
            logger.debug(f"Skipping {ticker}: amount €{actual_amount:.2f} below minimum €{min_position_value:.2f}")
            continue

        sizing_data.append({
            'ticker': ticker,
            'yfinance_ticker': position.yfinance_ticker,
            'company_name': position.company_name or '',
            'target_weight': weight,
            'actual_weight': weight,  # Exact match with fractional shares
            'price': price,
            'target_amount': target_amount,
            'shares': shares,
            'actual_amount': actual_amount,
            'weight_diff': 0.0  # Perfect allocation
        })

    df = pd.DataFrame(sizing_data)

    # Sort by actual amount (descending)
    df = df.sort_values('actual_amount', ascending=False)

    total_invested = df['actual_amount'].sum()
    logger.info(f"✓ Allocated {len(df)} positions using €{total_invested:.2f} of €{total_capital:.2f} ({(total_invested/total_capital)*100:.1f}%)")

    return df


def print_trading_plan(df: pd.DataFrame, total_capital: float):
    """Print trading plan summary."""
    print("\n" + "=" * 140)
    print(" " * 50 + "TRADING PLAN")
    print("=" * 140)

    print(f"\nTotal Capital: €{total_capital:,.2f}")
    print(f"Number of Positions: {len(df)}")

    print("\n" + "-" * 140)
    print(f"{'#':<3} {'Ticker':<15} {'Company':<30} {'Price':>12} {'Shares':>14} {'Cost':>14} {'Weight%':>10}")
    print("-" * 140)

    for i, row in df.iterrows():
        print(
            f"{i+1:<3} "
            f"{row['ticker']:<15} "
            f"{row['company_name'][:28]:<30} "
            f"€{row['price']:>11.2f} "
            f"{row['shares']:>13.4f} "
            f"€{row['actual_amount']:>12.2f} "
            f"{row['actual_weight']:>9.2%}"
        )

    print("-" * 140)

    # Summary statistics
    total_invested = df['actual_amount'].sum()
    remaining_cash = total_capital - total_invested
    total_weight = df['actual_weight'].sum()

    print(f"\n{'SUMMARY:':<25}")
    print(f"{'Total Invested:':<25} €{total_invested:>12,.2f} ({total_weight:>6.2%} of capital)")
    print(f"{'Remaining Cash:':<25} €{remaining_cash:>12,.2f} ({remaining_cash/total_capital:>6.2%} of capital)")
    print(f"{'Number of Positions:':<25} {len(df):>12.0f}")

    print("=" * 140)


def save_trading_plan(df: pd.DataFrame, portfolio_id: str, capital: float):
    """Save trading plan to CSV."""
    output_path = f"outputs/trading_plan_{portfolio_id}_{int(capital)}eur.csv"

    # Select columns for export
    export_df = df[[
        'ticker', 'yfinance_ticker', 'company_name',
        'target_weight', 'actual_weight', 'weight_diff',
        'price', 'shares', 'target_amount', 'actual_amount'
    ]].copy()

    # Rename columns for clarity
    export_df.columns = [
        'Ticker', 'YF_Ticker', 'Company',
        'Target_Weight_%', 'Actual_Weight_%', 'Weight_Difference_%',
        'Price_EUR', 'Shares', 'Target_Amount_EUR', 'Actual_Amount_EUR'
    ]

    # Format percentages
    export_df['Target_Weight_%'] = (export_df['Target_Weight_%'] * 100).round(2)
    export_df['Actual_Weight_%'] = (export_df['Actual_Weight_%'] * 100).round(2)
    export_df['Weight_Difference_%'] = (export_df['Weight_Difference_%'] * 100).round(2)

    # Round amounts
    export_df['Target_Amount_EUR'] = export_df['Target_Amount_EUR'].round(2)
    export_df['Actual_Amount_EUR'] = export_df['Actual_Amount_EUR'].round(2)
    export_df['Price_EUR'] = export_df['Price_EUR'].round(2)

    export_df.to_csv(output_path, index=False)

    logger.info(f"✓ Trading plan saved: {output_path}")


def main():
    """Run position sizing calculator."""
    parser = argparse.ArgumentParser(
        description="Calculate exact position sizes from portfolio weights"
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=2000.0,
        help='Total investment capital in EUR (e.g., 1000)'
    )
    parser.add_argument(
        '--portfolio_id',
        type=str,
        default=None,
        help='Portfolio UUID (if not provided, uses most recent)'
    )
    parser.add_argument(
        '--min_position',
        type=float,
        default=10.0,
        help='Minimum position size in EUR (default: 10)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 120)
    print(" " * 40 + "POSITION SIZING CALCULATOR")
    print("=" * 120)

    # Initialize database
    init_db()

    # Fetch portfolio
    logger.info("Fetching portfolio from database...")
    portfolio, positions = fetch_portfolio(args.portfolio_id)

    print(f"\nPortfolio: {portfolio.name}")
    print(f"  ID: {portfolio.id}")
    print(f"  Date: {portfolio.portfolio_date}")
    print(f"  Positions: {len(positions)} (non-zero)")
    print(f"  Total Weight: {float(portfolio.total_weight):.2%}")

    print(f"\nInvestment Capital: €{args.capital:,.2f}")

    # Fetch current prices
    prices = fetch_current_prices(positions)

    if not prices:
        logger.error("Failed to fetch any prices")
        return 1

    # Calculate position sizes
    sizing_df = calculate_position_sizes(
        positions,
        prices,
        args.capital,
        min_position_value=args.min_position
    )

    if sizing_df.empty:
        logger.error("No positions calculated (all below minimum?)")
        return 1

    # Print trading plan
    print_trading_plan(sizing_df, args.capital)

    # Save to CSV
    save_trading_plan(sizing_df, str(portfolio.id), args.capital)

    print("\n✓ Position sizing complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
