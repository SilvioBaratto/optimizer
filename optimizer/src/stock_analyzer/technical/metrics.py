from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats

from .indicators import (
    calculate_rsi,
    calculate_ewma_volatility,
    calculate_moving_averages,
    calculate_momentum,
)


def calculate_technical_metrics(
    stock_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame], risk_free_rate: float
) -> Dict[str, float]:
    """Calculate institutional-grade technical/risk metrics."""
    prices = np.array(stock_data["Close"].values, dtype=np.float64)
    returns = np.diff(prices) / prices[:-1]

    # Annualized metrics
    total_return = prices[-1] / prices[0] - 1
    years = len(prices) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Volatility: Use EWMA with 84-day half-life (Section 5.3.3)
    volatility = calculate_ewma_volatility(returns, halflife=84, annualize=True)

    # Sharpe Ratio
    sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Calmar Ratio
    calmar = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = (
        np.std(downside_returns, ddof=1) * np.sqrt(252) if len(downside_returns) > 1 else 0
    )
    sortino = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    metrics: Dict[str, Any] = {
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "risk_free_rate": risk_free_rate,
    }

    # Benchmark comparison with date alignment (Section 5.3.1)
    if benchmark_data is not None and len(benchmark_data) > 0:
        benchmark_metrics = _calculate_benchmark_metrics(
            stock_data, benchmark_data, risk_free_rate, annualized_return
        )
        metrics.update(benchmark_metrics)

    # RSI (14-day)
    if len(prices) >= 14:
        rsi_values = calculate_rsi(prices, period=14)
        metrics["rsi"] = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0
    else:
        metrics["rsi"] = 50.0  # Neutral if insufficient data

    # Moving Averages
    ma_dict = calculate_moving_averages(prices, windows=[50, 200])
    if 50 in ma_dict and not np.isnan(ma_dict[50]):
        metrics["ma_50"] = ma_dict[50]
        metrics["price_vs_ma50"] = prices[-1] / ma_dict[50] - 1

    if 200 in ma_dict and not np.isnan(ma_dict[200]):
        metrics["ma_200"] = ma_dict[200]
        metrics["price_vs_ma200"] = prices[-1] / ma_dict[200] - 1

    # Momentum with 1-month skip (Section 5.2.5: 12-1 month returns)
    if len(prices) >= 42:  # Need 2 months of data
        metrics["momentum_1m"] = calculate_momentum(prices, lookback_days=21, skip_days=0)

    if len(prices) >= 63:  # Need 3 months
        metrics["momentum_3m"] = calculate_momentum(prices, lookback_days=42, skip_days=21)

    if len(prices) >= 252:  # Need 1 year
        # 12-month return excluding most recent month (industry standard)
        metrics["momentum_12m_minus_1m"] = calculate_momentum(
            prices, lookback_days=231, skip_days=21
        )

    # ADV liquidity metrics (Section 4.4.5)
    if "Volume" in stock_data.columns and len(stock_data) >= 20:
        adv_20 = stock_data["Volume"].tail(20).mean()
        adv_dollars = adv_20 * prices[-1]

        metrics["adv_shares"] = adv_20
        metrics["adv_dollars"] = adv_dollars

    # Convert all numpy types to Python native types for database compatibility
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics[key] = float(value)
        elif isinstance(value, np.bool_):
            metrics[key] = bool(value)

    return metrics


def _calculate_benchmark_metrics(
    stock_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    risk_free_rate: float,
    stock_annualized_return: float,
) -> Dict[str, float]:
    """Calculate benchmark comparison metrics (beta, alpha, R², etc.)."""
    metrics: Dict[str, Any] = {}

    # Prepare dataframes
    stock_df = stock_data[["Close"]].copy()
    bench_df = benchmark_data[["Close"]].copy()

    # Remove timezone information for matching
    stock_df.index = pd.DatetimeIndex(stock_df.index).tz_localize(None)
    bench_df.index = pd.DatetimeIndex(bench_df.index).tz_localize(None)

    # Merge on normalized date indexes
    aligned_data = pd.merge(
        stock_df,
        bench_df,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=("_stock", "_bench"),
    )

    if len(aligned_data) > 30:  # Need sufficient data for regression
        aligned_stock_prices = aligned_data["Close_stock"].values
        aligned_bench_prices = aligned_data["Close_bench"].values

        # Calculate returns on aligned data
        stock_returns_aligned = np.diff(
            np.array(aligned_stock_prices, dtype=np.float64)
        ) / np.array(aligned_stock_prices[:-1], dtype=np.float64)
        bench_returns_aligned = np.diff(
            np.array(aligned_bench_prices, dtype=np.float64)
        ) / np.array(aligned_bench_prices[:-1], dtype=np.float64)

        # Annualized benchmark return
        bench_total_return = aligned_bench_prices[-1] / aligned_bench_prices[0] - 1
        bench_years = len(aligned_bench_prices) / 252
        bench_annualized = (1 + bench_total_return) ** (1 / bench_years) - 1

        # Regression-based beta and alpha
        daily_rf = risk_free_rate / 252
        excess_stock = stock_returns_aligned - daily_rf
        excess_bench = bench_returns_aligned - daily_rf

        if len(excess_stock) > 2:
            # Perform linear regression to calculate beta
            regression_result: Any = stats.linregress(excess_bench, excess_stock)

            beta = float(regression_result.slope)
            # Alpha from annual returns (avoids compounding errors from daily alpha)
            alpha = (
                stock_annualized_return
                - risk_free_rate
                - beta * (bench_annualized - risk_free_rate)
            )
            r_squared = float(regression_result.rvalue) ** 2

            # Tracking error and information ratio
            tracking_error = np.std(stock_returns_aligned - bench_returns_aligned) * np.sqrt(252)
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0

            # Statistical significance (Section 5.3.1)
            if len(excess_stock) > 30:
                # Standard error of alpha (annualized)
                se_alpha_annual = float(regression_result.stderr) * np.sqrt(252)
                alpha_t_stat = alpha / se_alpha_annual if se_alpha_annual > 0 else 0

                # 95% confidence interval
                alpha_lower_95 = alpha - 1.96 * se_alpha_annual
                alpha_upper_95 = alpha + 1.96 * se_alpha_annual
            else:
                alpha_t_stat = 0
                alpha_lower_95 = alpha
                alpha_upper_95 = alpha

            # Risk decomposition (Section 5.3.1)
            # Calculate volatility from aligned returns
            stock_volatility = np.std(stock_returns_aligned, ddof=1) * np.sqrt(252)
            total_variance = stock_volatility**2
            systematic_variance = r_squared * total_variance
            specific_variance = (1 - r_squared) * total_variance

            systematic_volatility = np.sqrt(systematic_variance)
            specific_volatility = np.sqrt(specific_variance)

            metrics.update(
                {
                    "beta": beta,
                    "alpha": alpha,
                    "r_squared": r_squared,
                    "information_ratio": information_ratio,
                    "benchmark_return": bench_annualized,
                    # Statistical significance
                    "alpha_t_stat": alpha_t_stat,
                    "alpha_lower_95": alpha_lower_95,
                    "alpha_upper_95": alpha_upper_95,
                    # Risk decomposition
                    "systematic_volatility": systematic_volatility,
                    "specific_volatility": specific_volatility,
                }
            )

    # Convert numpy types to native Python types
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics[key] = float(value)

    return metrics
