#!/usr/bin/env python3
"""
Institutional-grade AAPL price chart with risk metrics and factor analysis.

Usage:
    python src/utils/growth.py
"""

import numpy as np
import pandas as pd
from datetime import datetime
from optimizer.src.yfinance import YFinanceClient


def plot_ascii_graph(prices, dates, width=80, height=30):
    """
    Plot ASCII graph using arrows and characters.

    Args:
        prices: List of prices
        dates: List of dates corresponding to prices
        width: Width of the graph in characters
        height: Height of the graph in lines
    """
    if len(prices) < 2:
        print("Not enough data to plot")
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

    print(f"\n{'='*width}")
    print(f"AAPL Price Chart: {start_date} to {end_date}")
    print(f"Price Range: ${min_price:.2f} - ${max_price:.2f}")
    print(f"{'='*width}")
    print()

    # Print y-axis labels and canvas
    for i, row in enumerate(canvas):
        price_at_line = max_price - (i / (height - 1)) * price_range
        print(f"${price_at_line:7.2f} | {''.join(row)}")

    print(f"{'':>10} {''.join(['-' for _ in range(width)])}")

    # Print date markers along x-axis
    num_labels = 5
    label_positions = [i * (len(sampled_dates) - 1) // (num_labels - 1) for i in range(num_labels)]
    date_labels = [sampled_dates[pos].strftime('%Y-%m') for pos in label_positions]

    # Print x-axis dates
    spacing = width // (num_labels - 1)
    print(f"{'':>10} ", end='')
    for i, label in enumerate(date_labels):
        if i == 0:
            print(f"{label}", end='')
        else:
            padding = spacing - len(date_labels[i-1]) // 2 - len(label) // 2
            print(f"{' ' * padding}{label}", end='')
    print()

    # Print summary
    print()
    print(f"{'='*width}")
    print(f"TIME DIMENSION")
    print(f"{'='*80}")
    print(f"Period:          {start_date} to {end_date}")
    print(f"Total days:      {len(prices)} trading days")
    print(f"Data frequency:  Daily prices")
    print(f"Each bar:        ~{len(prices) // width} trading days")
    print()
    print(f"Peak:            ${max_price:.2f} on {peak_date.strftime('%Y-%m-%d')}")
    print(f"Trough:          ${min_price:.2f} on {trough_date.strftime('%Y-%m-%d')}")
    print(f"Start price:     ${prices[0]:.2f}")
    print(f"End price:       ${prices[-1]:.2f}")
    print(f"Change:          ${prices[-1] - prices[0]:+.2f} ({((prices[-1] / prices[0]) - 1) * 100:+.2f}%)")
    print(f"{'='*width}")


def calculate_risk_metrics(prices, benchmark_prices=None, risk_free_rate=0.045):
    """
    Calculate institutional-grade risk metrics.

    Args:
        prices: List of prices
        benchmark_prices: List of benchmark prices (e.g., SPY)
        risk_free_rate: Annual risk-free rate (default: 4.5% = current US 10Y)

    Returns:
        Dictionary of risk metrics
    """
    returns = np.diff(prices) / prices[:-1]

    # Annualized metrics (assuming 252 trading days)
    total_return = (prices[-1] / prices[0] - 1)
    years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1

    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Calmar Ratio (return / max drawdown)
    calmar = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

    # Downside deviation (for Sortino ratio) - FIXED: Use ddof=1
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) * np.sqrt(252) if len(downside_returns) > 1 else 0
    sortino = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'risk_free_rate': risk_free_rate,
    }

    # Benchmark comparison if provided
    if benchmark_prices is not None:
        bench_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
        bench_total_return = (benchmark_prices[-1] / benchmark_prices[0] - 1)
        bench_annualized = (1 + bench_total_return) ** (1 / years) - 1

        # IMPROVED: Regression-based beta and alpha (more accurate)
        from scipy import stats

        # Daily excess returns
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        excess_benchmark = bench_returns - daily_rf

        # Time-series regression: r_t - r_f = α + β(r_m - r_f) + ε_t
        # Use regression to get beta and R² (statistical fit quality)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            excess_benchmark,
            excess_returns
        )

        beta_regression = slope

        # FIXED: Calculate alpha from annual returns to ensure attribution sums correctly
        # α = R_p - R_f - β(R_m - R_f)
        # This avoids compounding issues with annualizing daily regression intercept
        # The daily intercept * 252 doesn't account for geometric compounding
        alpha = annualized_return - risk_free_rate - beta_regression * (bench_annualized - risk_free_rate)

        # Covariance-based beta (for comparison)
        covariance = np.cov(returns, bench_returns)[0, 1]
        bench_variance = np.var(bench_returns)
        beta_covariance = covariance / bench_variance if bench_variance > 0 else 1.0

        # Use regression-based beta
        beta = beta_regression

        # Information Ratio (alpha / tracking error)
        tracking_error = np.std(returns - bench_returns) * np.sqrt(252)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0

        metrics.update({
            'benchmark_return': bench_total_return,
            'benchmark_annualized': bench_annualized,
            'beta': beta,
            'beta_covariance': beta_covariance,  # For comparison
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'r_squared': r_value ** 2,  # How much variance explained by market
        })

    return metrics


def calculate_moving_averages(prices, window_short=50, window_long=200):
    """Calculate moving averages."""
    prices_series = pd.Series(prices)
    ma_short = prices_series.rolling(window=window_short).mean().tolist()
    ma_long = prices_series.rolling(window=window_long).mean().tolist()
    return ma_short, ma_long


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index using Wilder's smoothing method.

    FIXED: Initialize with NaN for first N values instead of setting all to same value.
    """
    deltas = np.diff(prices)

    # Initialize gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # FIXED: Initialize with NaN instead of zeros
    rsi = np.full(len(prices), np.nan)

    # First average (Simple Moving Average)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate first RSI value
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi[period] = 100 - (100 / (1 + rs))

    # Wilder's smoothing for subsequent values
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def print_risk_metrics(metrics, width=80):
    """Print institutional risk metrics."""
    print(f"\n{'='*width}")
    print("RISK-ADJUSTED PERFORMANCE METRICS")
    print(f"{'='*width}")
    print()
    print("RETURN METRICS:")
    print(f"  Total Return:         {metrics['total_return']*100:>8.2f}%")
    print(f"  Annualized Return:    {metrics['annualized_return']*100:>8.2f}%")

    if 'benchmark_return' in metrics:
        print(f"  Benchmark Return:     {metrics['benchmark_return']*100:>8.2f}%")
        print(f"  Benchmark Ann. Ret:   {metrics['benchmark_annualized']*100:>8.2f}%")
        excess = (metrics['annualized_return'] - metrics['benchmark_annualized']) * 100
        print(f"  Excess Return:        {excess:>8.2f}%")

    print()
    print("RISK METRICS:")
    print(f"  Volatility (ann.):    {metrics['volatility']*100:>8.2f}%")
    print(f"  Max Drawdown:         {metrics['max_drawdown']*100:>8.2f}%")

    if 'tracking_error' in metrics:
        print(f"  Tracking Error:       {metrics['tracking_error']*100:>8.2f}%")

    print()
    print("RISK-ADJUSTED RATIOS:")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:>8.2f}")
    print(f"  Calmar Ratio:         {metrics['calmar_ratio']:>8.2f}")

    if 'information_ratio' in metrics:
        print(f"  Information Ratio:    {metrics['information_ratio']:>8.2f}")

    if 'beta' in metrics and 'alpha' in metrics:
        print()
        print("FACTOR DECOMPOSITION:")
        print(f"  Beta (systematic):    {metrics['beta']:>8.2f}")
        if 'beta_covariance' in metrics:
            print(f"  Beta (covariance):    {metrics['beta_covariance']:>8.2f}")
        print(f"  Alpha (excess):       {metrics['alpha']*100:>8.2f}%")
        if 'r_squared' in metrics:
            print(f"  R² (expl. variance):  {metrics['r_squared']:>8.2f}")

        # FIXED: Correct return attribution including risk-free component
        # Total return = Risk-free + Beta × Market Risk Premium + Alpha
        # R_p = R_f + β(R_m - R_f) + α
        risk_free_component = metrics['risk_free_rate']
        market_risk_premium = metrics['benchmark_annualized'] - metrics['risk_free_rate']
        systematic_return = metrics['beta'] * market_risk_premium
        alpha_component = metrics['alpha']

        total_explained = risk_free_component + systematic_return + alpha_component

        print()
        print("RETURN ATTRIBUTION (Decomposition):")
        print(f"  Risk-Free Rate:       {risk_free_component*100:>8.2f}% ({risk_free_component/metrics['annualized_return']*100:>6.1f}%)")
        print(f"  Market Beta:          {systematic_return*100:>8.2f}% ({systematic_return/metrics['annualized_return']*100:>6.1f}%)")
        print(f"  Alpha:                {alpha_component*100:>8.2f}% ({alpha_component/metrics['annualized_return']*100:>6.1f}%)")
        print(f"  {'─'*40}")
        print(f"  Total Explained:      {total_explained*100:>8.2f}%")
        print(f"  Actual Return:        {metrics['annualized_return']*100:>8.2f}%")
        diff = abs(total_explained - metrics['annualized_return'])
        print(f"  Difference:           {diff*100:>8.4f}% {'✓' if diff < 0.001 else '⚠'}")


def calculate_percentile_rank(stock_return, benchmark_constituents_returns):
    """Calculate percentile rank vs universe."""
    if not benchmark_constituents_returns:
        return None

    sorted_returns = sorted(benchmark_constituents_returns)
    rank = sum(1 for r in sorted_returns if r < stock_return)
    percentile = (rank / len(sorted_returns)) * 100
    return percentile


def main():
    print("="*80)
    print("INSTITUTIONAL-GRADE EQUITY ANALYSIS: AAPL")
    print("="*80)
    print()

    # Fetch AAPL data
    print("Fetching AAPL data...")
    client = YFinanceClient.get_instance()
    hist = client.fetch_history("AAPL", period="5y")

    # Fetch benchmark (S&P 500)
    print("Fetching S&P 500 benchmark...")
    spy_hist = client.fetch_history("SPY", period="5y")

    # Align dates
    common_dates = hist.index.intersection(spy_hist.index)
    hist = hist.loc[common_dates]
    spy_hist = spy_hist.loc[common_dates]

    prices = hist['Close'].tolist()
    dates = hist.index.tolist()
    spy_prices = spy_hist['Close'].tolist()

    # Calculate risk metrics
    print("Calculating risk metrics...")
    metrics = calculate_risk_metrics(prices, spy_prices)

    # Calculate moving averages
    ma_50, ma_200 = calculate_moving_averages(prices)

    # Calculate RSI
    rsi = calculate_rsi(prices)

    # Plot ASCII graph
    plot_ascii_graph(prices, dates)

    # Print risk metrics
    print_risk_metrics(metrics)

    # Technical indicators summary
    current_rsi = rsi[-1]
    print()
    print(f"{'='*80}")
    print("TECHNICAL INDICATORS (Current)")
    print(f"{'='*80}")

    # Handle NaN values in moving averages
    if not np.isnan(ma_50[-1]):
        print(f"  50-day MA:            ${ma_50[-1]:>8.2f}")
        price_vs_ma50 = ((prices[-1] / ma_50[-1]) - 1) * 100
        print(f"  Price vs 50-MA:       {price_vs_ma50:>+7.2f}%")
    else:
        print(f"  50-day MA:            N/A (insufficient data)")

    if not np.isnan(ma_200[-1]):
        print(f"  200-day MA:           ${ma_200[-1]:>8.2f}")
        price_vs_ma200 = ((prices[-1] / ma_200[-1]) - 1) * 100
        print(f"  Price vs 200-MA:      {price_vs_ma200:>+7.2f}%")
    else:
        print(f"  200-day MA:           N/A (insufficient data)")

    # RSI with proper NaN handling
    if not np.isnan(current_rsi):
        print(f"  RSI (14):             {current_rsi:>8.2f} ", end='')
        if current_rsi > 70:
            print("(Overbought)")
        elif current_rsi < 30:
            print("(Oversold)")
        else:
            print("(Neutral)")
    else:
        print(f"  RSI (14):             N/A (insufficient data)")

    print()
    print(f"{'='*80}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*80}")
    print("Sharpe Ratio:   >1.0 = Good, >2.0 = Very Good, >3.0 = Excellent")
    print("Information:    >0.5 = Good, >0.75 = Very Good, >1.0 = Excellent")
    print("Beta:           <1.0 = Less volatile than market, >1.0 = More volatile")
    print("Alpha:          Positive = Outperforming risk-adjusted benchmark")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
