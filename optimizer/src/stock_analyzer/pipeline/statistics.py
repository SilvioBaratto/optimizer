"""
Cross-Sectional Statistics
===========================

Calculate universe-level statistics for true cross-sectional standardization.

This module is a simplified placeholder. The full implementation with robust
outlier detection and iterative refinement is available in the original
run_signal_analysis.py if needed.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

from .utils import safe_float

logger = logging.getLogger(__name__)


def calculate_cross_sectional_stats(
    fundamentals_list: List[Tuple],
    enable_stats: bool = True
) -> Optional[Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    Calculate cross-sectional statistics from raw fundamentals.

    This is a simplified version. For production use, consider implementing:
    - Robust outlier detection (iterative z-score filtering)
    - Winsorization for extreme values
    - Minimum sample size validation

    Args:
        fundamentals_list: List of (technical_metrics, info, country) tuples
        enable_stats: If False, return None (use static norms)

    Returns:
        Dictionary of statistics in format:
        {
            'value': {'book_price': (mean, std), 'earnings_price': (mean, std), ...},
            'momentum': {'momentum_12m': (mean, std)},
            'quality': {'roe': (mean, std), 'profit_margin': (mean, std), ...},
            'growth': {'revenue_growth': (mean, std), 'earnings_growth': (mean, std)}
        }
        or None if enable_stats is False
    """
    if not enable_stats:
        logger.info("Cross-sectional statistics disabled. Using static market norms.")
        return None

    if len(fundamentals_list) < 50:
        logger.warning(
            f"Insufficient data for cross-sectional stats ({len(fundamentals_list)} < 50). "
            f"Using static norms."
        )
        return None

    logger.info(f"Calculating cross-sectional statistics from {len(fundamentals_list)} stocks...")

    # Extract metrics
    value_metrics = _extract_value_metrics(fundamentals_list)
    momentum_metrics = _extract_momentum_metrics(fundamentals_list)
    quality_metrics = _extract_quality_metrics(fundamentals_list)
    growth_metrics = _extract_growth_metrics(fundamentals_list)

    # Calculate statistics
    stats = {
        'value': _calculate_metric_stats(value_metrics),
        'momentum': _calculate_metric_stats(momentum_metrics),
        'quality': _calculate_metric_stats(quality_metrics),
        'growth': _calculate_metric_stats(growth_metrics),
    }

    logger.info("Cross-sectional statistics calculated successfully.")
    return stats


def _extract_value_metrics(fundamentals_list: List[Tuple]) -> Dict[str, List[float]]:
    """
    Extract value factor metrics from fundamentals per Chapter 5.

    Chapter 5 specifies 4 metrics: B/P, E/P, S/P, D/P
    """
    metrics = {
        'book_price': [],
        'earnings_price': [],
        'sales_price': [],
        'dividend_price': []
    }

    for technical_metrics, info, country in fundamentals_list:
        if not info:
            continue

        # 1. Book/Price (inverse of P/B)
        price_to_book = safe_float(info.get('priceToBook'))
        if price_to_book > 0:
            metrics['book_price'].append(1 / price_to_book)

        # 2. Earnings/Price (inverse of P/E)
        trailing_pe = safe_float(info.get('trailingPE'))
        if trailing_pe > 0:
            metrics['earnings_price'].append(1 / trailing_pe)

        # 3. Sales/Price (inverse of P/S)
        price_to_sales = safe_float(info.get('priceToSalesTrailing12Months'))
        if price_to_sales > 0:
            metrics['sales_price'].append(1 / price_to_sales)

        # 4. Dividend/Price (dividend yield)
        # Note: ~40% of stocks don't pay dividends - this is structural, not missing
        dividend_yield = safe_float(info.get('dividendYield'))
        if dividend_yield > 0:
            metrics['dividend_price'].append(dividend_yield)

    return metrics


def _extract_momentum_metrics(fundamentals_list: List[Tuple]) -> Dict[str, List[float]]:
    """Extract momentum factor metrics from fundamentals."""
    metrics = {'momentum_12m': []}

    for technical_metrics, info, _ in fundamentals_list:
        if technical_metrics and 'momentum_12m_minus_1m' in technical_metrics:
            mom_12m = safe_float(technical_metrics.get('momentum_12m_minus_1m'))
            metrics['momentum_12m'].append(mom_12m)

    return metrics


def _extract_quality_metrics(fundamentals_list: List[Tuple]) -> Dict[str, List[float]]:
    """Extract quality factor metrics from fundamentals."""
    metrics = {'roe': [], 'profit_margin': [], 'sharpe_ratio': []}

    for technical_metrics, info, _ in fundamentals_list:
        if info:
            if 'returnOnEquity' in info:
                roe = safe_float(info.get('returnOnEquity'))
                metrics['roe'].append(roe)

            if 'profitMargins' in info:
                margin = safe_float(info.get('profitMargins'))
                metrics['profit_margin'].append(margin)

        if technical_metrics and 'sharpe_ratio' in technical_metrics:
            sharpe = safe_float(technical_metrics.get('sharpe_ratio'))
            metrics['sharpe_ratio'].append(sharpe)

    return metrics


def _extract_growth_metrics(fundamentals_list: List[Tuple]) -> Dict[str, List[float]]:
    """Extract growth factor metrics from fundamentals."""
    metrics = {'revenue_growth': [], 'earnings_growth': []}

    for _, info, _ in fundamentals_list:
        if not info:
            continue

        if 'revenueGrowth' in info:
            rev_growth = safe_float(info.get('revenueGrowth'))
            # Clip extreme growth values
            rev_growth_clipped = float(np.clip(rev_growth, -0.50, 1.00))
            metrics['revenue_growth'].append(rev_growth_clipped)

        if 'earningsGrowth' in info:
            earn_growth = safe_float(info.get('earningsGrowth'))
            # Clip extreme growth values
            earn_growth_clipped = float(np.clip(earn_growth, -0.50, 1.00))
            metrics['earnings_growth'].append(earn_growth_clipped)

    return metrics


def _calculate_metric_stats(
    metrics_dict: Dict[str, List[float]]
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate (mean, std) for each metric in dictionary.

    Filters out NaN/inf values and requires minimum 10 valid samples.

    Args:
        metrics_dict: Dictionary mapping metric name to list of values

    Returns:
        Dictionary mapping metric name to (mean, std) tuple
    """
    stats = {}

    for metric_name, values in metrics_dict.items():
        if not values or len(values) < 10:
            continue

        # Filter finite values
        arr = np.array(values)
        valid_arr = arr[np.isfinite(arr)]

        if len(valid_arr) < 10:
            continue

        mean = float(np.mean(valid_arr))
        std = float(np.std(valid_arr, ddof=1))

        if std > 0:
            stats[metric_name] = (mean, std)
            logger.debug(
                f"{metric_name}: n={len(valid_arr)}, mean={mean:.4f}, std={std:.4f}"
            )

    return stats
