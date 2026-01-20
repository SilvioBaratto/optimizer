import numpy as np
import pandas as pd


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI (Relative Strength Index) indicator."""
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
    returns: np.ndarray, halflife: int = 84, annualize: bool = True
) -> float:
    """Calculate EWMA (Exponentially Weighted Moving Average) volatility."""
    if len(returns) < halflife:
        # Fallback to simple volatility if insufficient data
        std = float(np.std(returns, ddof=1))
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
    prices: np.ndarray, windows: list[int] = [50, 200]
) -> dict[int, float]:
    """Calculate simple moving averages for multiple windows."""
    result = {}
    prices_series = pd.Series(prices)

    for window in windows:
        if len(prices) >= window:
            ma = prices_series.rolling(window=window).mean().iloc[-1]
            result[window] = float(ma) if not np.isnan(ma) else np.nan
        else:
            result[window] = np.nan

    return result


def calculate_momentum(prices: np.ndarray, lookback_days: int, skip_days: int = 0) -> float:
    """Calculate price momentum with optional skip period."""
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
