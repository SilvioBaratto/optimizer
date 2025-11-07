"""
Technical Indicators
===================

Provides calculation of standard technical indicators used in quantitative analysis.

Indicators:
- RSI (Relative Strength Index): 14-day momentum oscillator
- EWMA Volatility: Exponentially weighted moving average volatility with 84-day half-life
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate RSI (Relative Strength Index) indicator.

    RSI measures momentum by comparing recent gains to recent losses.
    Values range from 0-100:
    - Above 70: Overbought
    - Below 30: Oversold

    Args:
        prices: Array of historical prices
        period: Lookback period for RSI calculation (default: 14)

    Returns:
        Array of RSI values (same length as prices, with NaN for insufficient data)
    """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi = np.full(len(prices), np.nan)

    if len(gains) < period:
        return rsi

    # Initial average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi[period] = 100 - (100 / (1 + rs))

    # Exponential moving average for subsequent values
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_ewma_volatility(
    returns: np.ndarray,
    halflife: int = 84,
    annualize: bool = True
) -> float:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.

    Implements Section 5.3.3 of institutional framework:
    - Uses 84-day half-life (MSCI standard for volatility estimation)
    - Exponential weighting gives more importance to recent data
    - More responsive to regime changes than simple historical volatility

    Args:
        returns: Array of daily returns
        halflife: Half-life in days (default: 84, per MSCI standard)
        annualize: Whether to annualize the result (default: True)

    Returns:
        EWMA volatility (annualized if annualize=True)
    """
    if len(returns) < halflife:
        # Fallback to simple volatility if insufficient data
        std = np.std(returns, ddof=1)
        return std * np.sqrt(252) if annualize else std

    # Decay factor: lambda = 2^(-1/halflife)
    lambda_decay = 2 ** (-1 / halflife)

    # EWMA using pandas for efficiency
    returns_series = pd.Series(returns)
    ewma_var = returns_series.ewm(alpha=1 - lambda_decay, adjust=False).var()

    # Get most recent variance
    current_var = ewma_var.iloc[-1]

    # Annualize if requested (252 trading days per year)
    if annualize:
        return float(np.sqrt(current_var * 252))
    else:
        return float(np.sqrt(current_var))


def calculate_moving_averages(
    prices: np.ndarray,
    windows: list[int] = [50, 200]
) -> dict[int, float]:
    """
    Calculate simple moving averages for multiple windows.

    Args:
        prices: Array of historical prices
        windows: List of window sizes (default: [50, 200])

    Returns:
        Dictionary mapping window size to moving average value
        Returns NaN for windows larger than available data
    """
    result = {}
    prices_series = pd.Series(prices)

    for window in windows:
        if len(prices) >= window:
            ma = prices_series.rolling(window=window).mean().iloc[-1]
            result[window] = float(ma) if not np.isnan(ma) else np.nan
        else:
            result[window] = np.nan

    return result


def calculate_momentum(
    prices: np.ndarray,
    lookback_days: int,
    skip_days: int = 0
) -> float:
    """
    Calculate price momentum with optional skip period.

    Institutional standard (Section 5.2.5):
    - 12-1 month momentum: Skip most recent 21 days to avoid reversal
    - Measures intermediate-term trend

    Args:
        prices: Array of historical prices (most recent last)
        lookback_days: Number of days to look back
        skip_days: Number of most recent days to skip (default: 0)

    Returns:
        Momentum as percentage return (e.g., 0.15 = 15% return)
        Returns 0.0 if insufficient data
    """
    required_days = lookback_days + skip_days

    if len(prices) < required_days:
        return 0.0

    # Calculate return from (t-lookback_days-skip_days) to (t-skip_days)
    if skip_days > 0:
        end_price = prices[-skip_days - 1]  # -1 because -0 doesn't work
        start_price = prices[-required_days]
    else:
        end_price = prices[-1]
        start_price = prices[-lookback_days]

    return (end_price / start_price - 1.0) if start_price > 0 else 0.0
