#!/usr/bin/env python3
"""
Momentum Benchmark Comparison
==============================

Compares your Black-Litterman portfolio against a pure momentum benchmark.

The Test:
- Momentum strategy: Long top 20 stocks by 12-1 month return (equal-weighted)
- Your strategy: Current BL-optimized portfolio from database
- Backtest period: 5 years (2020-2025)
- Universe: Same stocks as your portfolio

Key Questions:
1. Do you beat momentum on risk-adjusted returns (Sharpe ratio)?
2. What's your alpha after controlling for momentum exposure?
3. Are you just replicating momentum (high correlation/beta)?
4. Is the added complexity justified?

CRITICAL: Cross-Market Timezone Handling
----------------------------------------
This script handles mixed US/European portfolios correctly by:
- Normalizing all price data to timezone-naive format
- US stocks: America/New_York → removed
- European stocks: Europe/London/Paris → removed
- This enables proper date alignment by calendar date
- Without this fix, same calendar dates won't align due to TZ offsets

Usage:
    python src/data_visualization/momentum_benchmark.py [portfolio_id]
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
from scipy import stats
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


def fetch_portfolio(portfolio_id: str | None = None) -> Tuple[Portfolio, List[PortfolioPosition]]:
    """Fetch portfolio and positions from database."""
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

        return portfolio, list(positions)


def calculate_momentum_scores(
    tickers: List[str],
    calculation_date: datetime,
    lookback_days: int = 365,
    exclude_recent_days: int = 21
) -> pd.DataFrame:
    """
    Calculate momentum scores for a universe of stocks.

    Momentum = Total return from [t-365 to t-21]
    (Excludes last 21 days to avoid short-term reversals)

    Args:
        tickers: List of yfinance tickers
        calculation_date: Date to calculate momentum as of
        lookback_days: Lookback period for momentum (default: 365 = 12 months)
        exclude_recent_days: Days to exclude at end (default: 21 = 1 month)

    Returns:
        DataFrame with columns: ticker, momentum, price_data_quality
    """
    logger.info(f"Calculating momentum scores for {len(tickers)} stocks...")
    logger.info(f"  Lookback: {lookback_days} days, excluding last {exclude_recent_days} days")

    yf_client = YFinanceClient.get_instance()

    momentum_data = []
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(tickers)}")

        try:
            # Fetch enough data to ensure we have lookback period
            total_days = lookback_days + exclude_recent_days + 100  # Buffer

            data = yf_client.fetch_history(
                ticker,
                period=f"{total_days}d",
                min_rows=int((lookback_days + exclude_recent_days) * 0.6)
            )

            if data is None or data.empty:
                failed_tickers.append(ticker)
                continue

            # Use adjusted close
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

            # CRITICAL FIX: Normalize timezone for cross-market portfolios
            # US stocks use America/New_York, European stocks use Europe/Paris, etc.
            # Remove timezone to enable proper date alignment by calendar date
            if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
                prices = prices.copy()
                prices.index = prices.index.tz_localize(None)

            # Filter to dates before calculation_date (now both are timezone-naive)
            calculation_date_naive = calculation_date.replace(tzinfo=None)
            prices = prices[prices.index <= calculation_date_naive]

            if len(prices) < (lookback_days + exclude_recent_days) * 0.6:
                failed_tickers.append(ticker)
                continue

            # Calculate momentum: [t-365] to [t-21]
            end_idx = -exclude_recent_days if exclude_recent_days > 0 else len(prices)
            start_idx = -(lookback_days + exclude_recent_days)

            # Ensure indices are valid
            if abs(start_idx) > len(prices):
                start_idx = 0
            if isinstance(end_idx, int) and end_idx < 0 and abs(end_idx) > len(prices):
                end_idx = -1

            price_start = prices.iloc[start_idx]
            price_end = prices.iloc[end_idx] if isinstance(end_idx, int) else prices.iloc[-1]

            momentum = (price_end / price_start) - 1

            # Data quality: ratio of available days to expected days
            expected_days = lookback_days + exclude_recent_days
            quality = min(1.0, len(prices) / expected_days)

            momentum_data.append({
                'ticker': ticker,
                'momentum': momentum,
                'price_data_quality': quality,
                'price_start': price_start,
                'price_end': price_end
            })

        except Exception as e:
            logger.warning(f"Failed to calculate momentum for {ticker}: {e}")
            failed_tickers.append(ticker)

    if not momentum_data:
        raise ValueError("Failed to calculate momentum for any stocks")

    df = pd.DataFrame(momentum_data)

    logger.info(f"✓ Calculated momentum: {len(df)} stocks")
    logger.info(f"  Failed: {len(failed_tickers)} stocks")
    logger.info(f"  Top momentum: {df.nlargest(5, 'momentum')[['ticker', 'momentum']].to_dict('records')}")

    return df


def create_momentum_portfolio(
    momentum_scores: pd.DataFrame,
    n_stocks: int = 20
) -> Dict[str, float]:
    """
    Create equal-weighted momentum portfolio.

    Args:
        momentum_scores: DataFrame with momentum scores
        n_stocks: Number of stocks to hold (default: 20)

    Returns:
        Dictionary mapping ticker -> weight
    """
    # Select top N by momentum
    top_stocks = momentum_scores.nlargest(n_stocks, 'momentum')

    # Equal weight
    weight = 1.0 / n_stocks
    weights = {row['ticker']: weight for _, row in top_stocks.iterrows()}

    logger.info(f"✓ Created momentum portfolio: {n_stocks} stocks, equal-weighted")
    logger.info(f"  Top 5 holdings:")
    for ticker in list(weights.keys())[:5]:
        mom = momentum_scores[momentum_scores['ticker'] == ticker]['momentum'].iloc[0]
        logger.info(f"    {ticker}: {weight:.2%} (momentum: {mom:+.2%})")

    return weights


def fetch_historical_prices_bulk(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical prices for multiple tickers.

    Args:
        tickers: List of yfinance tickers
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with adjusted close prices (dates × tickers)
    """
    logger.info(f"Fetching historical prices for {len(tickers)} stocks...")
    logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")

    yf_client = YFinanceClient.get_instance()

    prices = {}
    failed = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(tickers)}")

        try:
            days = (end_date - start_date).days + 100

            data = yf_client.fetch_history(
                ticker,
                period=f"{days}d",
                min_rows=int(days * 0.5)
            )

            if data is None or data.empty:
                failed.append(ticker)
                continue

            # Use adjusted close
            if 'Adj Close' in data.columns:
                price_series = data['Adj Close']
            else:
                price_series = data['Close']

            # CRITICAL FIX: Normalize timezone for cross-market portfolios
            # US stocks use America/New_York, European stocks use Europe/Paris, etc.
            # Remove timezone to enable proper date alignment by calendar date
            if isinstance(price_series.index, pd.DatetimeIndex) and price_series.index.tz is not None:
                price_series = price_series.copy()
                price_series.index = price_series.index.tz_localize(None)

            prices[ticker] = price_series

        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            failed.append(ticker)

    if not prices:
        raise ValueError("Failed to fetch any price data")

    # Create DataFrame (all prices now timezone-naive after normalization)
    price_df = pd.DataFrame(prices)

    # Ensure dates are timezone-naive for filtering
    start_date_naive = start_date.replace(tzinfo=None)
    end_date_naive = end_date.replace(tzinfo=None)

    # Filter to date range (now all using same timezone-naive format)
    price_df = price_df[(price_df.index >= start_date_naive) & (price_df.index <= end_date_naive)]

    # Forward fill missing data (max 5 days)
    price_df = price_df.ffill(limit=5)

    # Drop rows with any NaN
    price_df = price_df.dropna()

    logger.info(f"✓ Fetched prices: {len(price_df)} days × {len(price_df.columns)} stocks")

    if failed:
        logger.warning(f"  Failed tickers ({len(failed)}): {failed[:10]}...")

    return price_df


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.Series:
    """
    Calculate portfolio returns using position weights.

    Args:
        prices: DataFrame of historical prices
        weights: Dictionary mapping ticker -> weight

    Returns:
        Series of portfolio returns (indexed by date)
    """
    # Filter to tickers that exist in both weights and prices
    common_tickers = set(weights.keys()) & set(prices.columns)

    if not common_tickers:
        raise ValueError("No common tickers between weights and prices")

    # Normalize weights for available tickers
    weights_filtered = {t: weights[t] for t in common_tickers}
    total_weight = sum(weights_filtered.values())
    weights_normalized = {t: w / total_weight for t, w in weights_filtered.items()}

    # Calculate daily returns
    returns = prices[list(common_tickers)].pct_change()

    # Calculate weighted portfolio returns
    portfolio_returns = pd.Series(0.0, index=returns.index)

    for ticker, weight in weights_normalized.items():
        portfolio_returns += weight * returns[ticker]

    # Drop first row (NaN from pct_change)
    portfolio_returns = portfolio_returns.dropna()

    logger.info(f"✓ Calculated returns: {len(portfolio_returns)} days, {len(common_tickers)} stocks")

    return portfolio_returns


def calculate_metrics(returns: pd.Series, name: str = "Portfolio") -> Dict:
    """Calculate performance metrics."""
    # Cumulative return
    cumulative_return = (1 + returns).cumprod().iloc[-1] - 1

    # Annualized return
    n_years = len(returns) / 252
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1

    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (4% risk-free rate)
    risk_free_rate = 0.04
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    return {
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


def regression_analysis(
    your_returns: pd.Series,
    momentum_returns: pd.Series
) -> Tuple[float, float, float, float, float]:
    """
    Run regression: Your_Returns = alpha + beta * Momentum_Returns + epsilon

    Returns:
        (alpha, beta, r_squared, t_stat_alpha, p_value_alpha)
    """
    # Align to common dates
    common_dates = your_returns.index.intersection(momentum_returns.index)
    y = your_returns[common_dates].values
    x = momentum_returns[common_dates].values

    # Add constant for intercept
    X = np.column_stack([np.ones(len(x)), x])

    # OLS regression
    from numpy.linalg import inv
    beta_hat = inv(X.T @ X) @ X.T @ y

    alpha = float(beta_hat[0])
    beta = float(beta_hat[1])

    # R-squared
    y_pred = X @ beta_hat
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - (ss_res / ss_tot)

    # T-statistic for alpha
    se_residuals = np.sqrt(ss_res / (len(y) - 2))
    se_alpha = float(se_residuals * np.sqrt((X.T @ X)[0, 0] / (len(y) - 2)))
    t_stat_alpha = alpha / se_alpha if se_alpha > 0 else 0.0

    # P-value for alpha (two-tailed)
    p_value_alpha = float(2 * (1 - stats.t.cdf(abs(t_stat_alpha), len(y) - 2)))

    logger.info(f"✓ Regression analysis:")
    logger.info(f"  Alpha: {alpha*252:.4f} (annualized)")
    logger.info(f"  Beta: {beta:.4f}")
    logger.info(f"  R²: {r_squared:.4f}")
    logger.info(f"  T-stat (alpha): {t_stat_alpha:.2f}")
    logger.info(f"  P-value (alpha): {p_value_alpha:.4f}")

    return alpha * 252, beta, r_squared, t_stat_alpha, p_value_alpha  # Annualize alpha


def print_comparison(
    your_metrics: Dict,
    momentum_metrics: Dict,
    alpha: float,
    beta: float,
    correlation: float,
    t_stat: float,
    p_value: float
):
    """Print detailed comparison and verdict."""
    print("\n" + "=" * 90)
    print("MOMENTUM BENCHMARK COMPARISON")
    print("=" * 90)

    print(f"\n{'Metric':<30} {'Your Portfolio':>18} {'Momentum':>18} {'Difference':>18}")
    print("-" * 90)

    # Returns
    print(f"{'Total Return':<30} "
          f"{your_metrics['cumulative_return']:>17.2%} "
          f"{momentum_metrics['cumulative_return']:>17.2%} "
          f"{your_metrics['cumulative_return'] - momentum_metrics['cumulative_return']:>+17.2%}")

    print(f"{'Annualized Return':<30} "
          f"{your_metrics['annualized_return']:>17.2%} "
          f"{momentum_metrics['annualized_return']:>17.2%} "
          f"{your_metrics['annualized_return'] - momentum_metrics['annualized_return']:>+17.2%}")

    # Risk
    print(f"{'Volatility (Annual)':<30} "
          f"{your_metrics['volatility']:>17.2%} "
          f"{momentum_metrics['volatility']:>17.2%} "
          f"{your_metrics['volatility'] - momentum_metrics['volatility']:>+17.2%}")

    # Risk-adjusted
    print(f"{'Sharpe Ratio':<30} "
          f"{your_metrics['sharpe_ratio']:>17.2f} "
          f"{momentum_metrics['sharpe_ratio']:>17.2f} "
          f"{your_metrics['sharpe_ratio'] - momentum_metrics['sharpe_ratio']:>+17.2f}")

    print(f"{'Sortino Ratio':<30} "
          f"{your_metrics['sortino_ratio']:>17.2f} "
          f"{momentum_metrics['sortino_ratio']:>17.2f} "
          f"{your_metrics['sortino_ratio'] - momentum_metrics['sortino_ratio']:>+17.2f}")

    # Drawdown
    print(f"{'Max Drawdown':<30} "
          f"{your_metrics['max_drawdown']:>17.2%} "
          f"{momentum_metrics['max_drawdown']:>17.2%} "
          f"{your_metrics['max_drawdown'] - momentum_metrics['max_drawdown']:>+17.2%}")

    # Win rate
    print(f"{'Win Rate (Daily)':<30} "
          f"{your_metrics['win_rate']:>17.2%} "
          f"{momentum_metrics['win_rate']:>17.2%} "
          f"{your_metrics['win_rate'] - momentum_metrics['win_rate']:>+17.2%}")

    print("\n" + "=" * 90)
    print("STATISTICAL ANALYSIS")
    print("=" * 90)

    print(f"\nRegression: Your_Returns = alpha + beta × Momentum_Returns + error")
    print(f"  Alpha (annualized):     {alpha:>8.2%}  (excess return unexplained by momentum)")
    print(f"  Beta:                   {beta:>8.2f}  (momentum exposure: 1.0 = pure replication)")
    print(f"  Correlation:            {correlation:>8.2f}  (similarity: 1.0 = identical)")
    print(f"  T-statistic (alpha):    {t_stat:>8.2f}  (significance: >2.0 = likely real)")
    print(f"  P-value (alpha):        {p_value:>8.4f}  (probability alpha = 0)")

    # Interpretation
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    is_significant = p_value < 0.05
    is_high_correlation = correlation > 0.7
    is_high_beta = beta > 0.8
    beats_sharpe = your_metrics['sharpe_ratio'] > momentum_metrics['sharpe_ratio']

    if alpha > 0 and is_significant and beats_sharpe:
        print("\n✓ YOUR SYSTEM ADDS VALUE BEYOND MOMENTUM")
        print(f"  - Alpha: {alpha:+.2%} per year (statistically significant)")
        print(f"  - Sharpe ratio: {your_metrics['sharpe_ratio']:.2f} vs {momentum_metrics['sharpe_ratio']:.2f}")

        if is_high_correlation:
            print(f"  - WARNING: High correlation ({correlation:.2f}) suggests you're still momentum-heavy")
            print(f"  - Consider: Are your other signals (quality, value, macro) contributing?")
        else:
            print(f"  - Low correlation ({correlation:.2f}) confirms orthogonal strategy")

        print("\n  RECOMMENDATION: Your complexity is justified. Keep the system.")

    elif alpha > 0 and not is_significant:
        print("\n⚠ INCONCLUSIVE: Possible alpha but not statistically significant")
        print(f"  - Alpha: {alpha:+.2%} per year (t-stat: {t_stat:.2f}, need >2.0)")
        print(f"  - This could be luck or insufficient data")
        print("\n  RECOMMENDATION: Collect more data (6-12 months) before concluding.")

    elif is_high_beta and is_high_correlation:
        print("\n✗ YOU'RE JUST REPLICATING MOMENTUM")
        print(f"  - Beta: {beta:.2f} (you move almost identically with momentum)")
        print(f"  - Correlation: {correlation:.2f} (very similar returns)")
        print(f"  - Alpha: {alpha:+.2%} (not worth the complexity)")

        print("\n  RECOMMENDATION: Simplify dramatically.")
        print("    Option 1: Use pure momentum (cheaper, simpler)")
        print("    Option 2: Add truly orthogonal signals (value, low-vol, quality)")
        print("    Option 3: Focus on areas you clearly add value (tail risk, regime timing)")

    else:
        print("\n⚠ MIXED RESULTS")
        print(f"  - Alpha: {alpha:+.2%}")
        print(f"  - Correlation: {correlation:.2f}")
        print(f"  - Sharpe: {your_metrics['sharpe_ratio']:.2f} vs {momentum_metrics['sharpe_ratio']:.2f}")

        if not beats_sharpe:
            print("\n  You're NOT beating momentum on risk-adjusted basis.")
            print("  RECOMMENDATION: Investigate why Sharpe is lower despite different strategy.")
        else:
            print("\n  RECOMMENDATION: Analyze which components add value vs. which are noise.")

    print("\n" + "=" * 90)


def plot_comparison(
    your_returns: pd.Series,
    momentum_returns: pd.Series,
    sp500_returns: pd.Series,
    your_name: str,
    output_path: str
):
    """Create comparison charts."""
    # Align to common dates
    common_dates = your_returns.index.intersection(momentum_returns.index).intersection(sp500_returns.index)
    your_returns = your_returns[common_dates]
    momentum_returns = momentum_returns[common_dates]
    sp500_returns = sp500_returns[common_dates]

    # Calculate cumulative returns
    your_cum = (1 + your_returns).cumprod()
    momentum_cum = (1 + momentum_returns).cumprod()
    sp500_cum = (1 + sp500_returns).cumprod()

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f'Momentum Benchmark Comparison: {your_name}', fontsize=16, fontweight='bold')

    # 1. Cumulative returns
    ax1 = axes[0]
    ax1.plot(your_cum.index, your_cum.values, label='Your Portfolio (BL)', linewidth=2, color='#2E86AB')
    ax1.plot(momentum_cum.index, momentum_cum.values, label='Momentum Benchmark', linewidth=2, color='#E63946', linestyle='--')
    ax1.plot(sp500_cum.index, sp500_cum.values, label='S&P 500', linewidth=1.5, color='#A8A8A8', alpha=0.7)
    ax1.set_ylabel('Cumulative Return (1 = 0%)', fontsize=11)
    ax1.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 2. Rolling correlation (252-day)
    ax2 = axes[1]
    rolling_corr = your_returns.rolling(252).corr(momentum_returns)
    ax2.plot(rolling_corr.index, rolling_corr.values, linewidth=2, color='#F77F00')
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High correlation threshold (0.7)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium correlation threshold (0.5)')
    ax2.set_ylabel('Correlation', fontsize=11)
    ax2.set_title('Rolling 1-Year Correlation with Momentum', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.2, 1.0)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 3. Excess returns (Your portfolio - Momentum)
    ax3 = axes[2]
    excess_returns = your_returns - momentum_returns
    excess_cum = (1 + excess_returns).cumprod()
    ax3.plot(excess_cum.index, excess_cum.values, linewidth=2, color='#06A77D')
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    ax3.fill_between(excess_cum.index, 1.0, excess_cum.values,
                      where=(excess_cum.values >= 1.0), alpha=0.3, color='green', label='Outperformance')
    ax3.fill_between(excess_cum.index, 1.0, excess_cum.values,
                      where=(excess_cum.values < 1.0), alpha=0.3, color='red', label='Underperformance')
    ax3.set_ylabel('Cumulative Excess Return', fontsize=11)
    ax3.set_title('Your Portfolio vs Momentum (Cumulative Difference)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 4. Drawdown comparison
    ax4 = axes[3]
    your_dd = (your_cum - your_cum.expanding().max()) / your_cum.expanding().max()
    momentum_dd = (momentum_cum - momentum_cum.expanding().max()) / momentum_cum.expanding().max()

    ax4.fill_between(your_dd.index, 0, your_dd.values, alpha=0.5, color='#2E86AB', label='Your Portfolio')
    ax4.fill_between(momentum_dd.index, 0, momentum_dd.values, alpha=0.3, color='#E63946', label='Momentum')
    ax4.set_ylabel('Drawdown', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved chart: {output_path}")

    plt.close()


def main(portfolio_id: str | None = None):
    """Run momentum benchmark comparison."""
    print("\n" + "=" * 90)
    print("MOMENTUM BENCHMARK COMPARISON")
    print("=" * 90)

    # Initialize database
    init_db()

    # Fetch portfolio
    logger.info("Fetching portfolio from database...")
    portfolio, positions = fetch_portfolio(portfolio_id)

    print(f"\nYour Portfolio: {portfolio.name}")
    print(f"  ID: {portfolio.id}")
    print(f"  Date: {portfolio.portfolio_date}")
    print(f"  Positions: {len(positions)} ({portfolio.total_positions} non-zero)")

    # Extract portfolio weights (using yfinance tickers)
    your_weights = {}
    yf_tickers = []

    for pos in positions:
        if float(pos.weight) > 0 and pos.yfinance_ticker:
            your_weights[pos.yfinance_ticker] = float(pos.weight)
            yf_tickers.append(pos.yfinance_ticker)

    logger.info(f"✓ Extracted {len(your_weights)} positions with yfinance tickers")

    # Define backtest period (5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365 + 50)

    print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")

    # Calculate momentum scores using same universe
    print(f"\nStep 1: Calculate momentum scores for your universe ({len(yf_tickers)} stocks)")
    momentum_scores = calculate_momentum_scores(
        tickers=yf_tickers,
        calculation_date=end_date,
        lookback_days=365,
        exclude_recent_days=21
    )

    # Create momentum benchmark portfolio
    print(f"\nStep 2: Construct momentum benchmark (top 20 by 12-1 month return)")
    momentum_weights = create_momentum_portfolio(momentum_scores, n_stocks=20)

    # Fetch historical prices for all stocks
    all_tickers = list(set(your_weights.keys()) | set(momentum_weights.keys()))
    print(f"\nStep 3: Fetch historical prices for {len(all_tickers)} stocks")
    prices = fetch_historical_prices_bulk(all_tickers, start_date, end_date)

    # Calculate returns for both strategies
    print(f"\nStep 4: Calculate portfolio returns")
    your_returns = calculate_portfolio_returns(prices, your_weights)
    momentum_returns = calculate_portfolio_returns(prices, momentum_weights)

    # Fetch S&P 500 for reference
    print(f"\nStep 5: Fetch S&P 500 benchmark")
    yf_client = YFinanceClient.get_instance()
    days = (end_date - start_date).days + 100
    sp500_data = yf_client.fetch_history("^GSPC", period=f"{days}d")

    if sp500_data is not None and not sp500_data.empty:
        sp500_prices = sp500_data['Adj Close'] if 'Adj Close' in sp500_data.columns else sp500_data['Close']

        # CRITICAL FIX: Normalize timezone for S&P 500
        if isinstance(sp500_prices.index, pd.DatetimeIndex) and sp500_prices.index.tz is not None:
            sp500_prices = sp500_prices.copy()
            sp500_prices.index = sp500_prices.index.tz_localize(None)

        # Filter to date range (all timezone-naive now)
        start_date_naive = start_date.replace(tzinfo=None)
        end_date_naive = end_date.replace(tzinfo=None)
        sp500_prices = sp500_prices[(sp500_prices.index >= start_date_naive) & (sp500_prices.index <= end_date_naive)]
        sp500_returns = sp500_prices.pct_change().dropna()
    else:
        logger.warning("Failed to fetch S&P 500 data, will skip in visualization")
        sp500_returns = pd.Series(dtype=float)

    # Calculate metrics
    print(f"\nStep 6: Calculate performance metrics")
    your_metrics = calculate_metrics(your_returns, "Your Portfolio")
    momentum_metrics = calculate_metrics(momentum_returns, "Momentum Benchmark")

    # Regression analysis
    print(f"\nStep 7: Run regression analysis")
    alpha, beta, r_squared, t_stat, p_value = regression_analysis(your_returns, momentum_returns)

    # Correlation
    common_dates = your_returns.index.intersection(momentum_returns.index)
    correlation = your_returns[common_dates].corr(momentum_returns[common_dates])

    # Print comparison and verdict
    print_comparison(your_metrics, momentum_metrics, alpha, beta, correlation, t_stat, p_value)

    # Create charts
    output_path = f"outputs/momentum_benchmark_{portfolio.id}.png"
    print(f"\nStep 8: Generate comparison charts")
    plot_comparison(your_returns, momentum_returns, sp500_returns, portfolio.name, output_path)

    print(f"\n✓ Analysis complete!")
    print(f"  Chart saved: {output_path}")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    portfolio_id = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        main(portfolio_id)
    except Exception as e:
        logger.error(f"Benchmark comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
