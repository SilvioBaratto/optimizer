"""
Cross-Sectional Standardization - 7-Pass Institutional Implementation
======================================================================

Implements the full institutional-grade cross-sectional standardization
approach per Chapter 5 of the portfolio guidelines.

The 7-pass approach ensures:
- Mean = 0.000, Std = 1.000 across the universe
- Perfect 20/20/20/20/20 bucket distribution
- Iterative outlier removal at ±3σ
- Factor correlation validation
- Momentum filters for signal refinement
"""

import logging
import asyncio
import numpy as np
from datetime import date as date_type
from typing import List, Optional, Tuple

from app.database import database_manager
from baml_client.types import SignalType

from .database import save_signal

logger = logging.getLogger(__name__)


async def calculate_robust_cross_sectional_stats(
    raw_data: List[dict]
) -> dict:
    """
    Pass 1.5: Calculate robust cross-sectional statistics with iterative outlier removal.

    Chapter 5 specifies:
    - Cap-weighted means (MSCI Barra approach)
    - Equal-weighted standard deviations
    - Iterative outlier removal at ±3σ (3 iterations)

    Args:
        raw_data: List of dicts with keys: instrument, technical_metrics, info, country

    Returns:
        Cross-sectional statistics dict with structure:
        {
            'value': {'book_price': (mean, std), ...},
            'momentum': {'momentum_12m': (mean, std), ...},
            'quality': {'roe': (mean, std), ...},
            'growth': {'revenue_growth': (mean, std), ...}
        }
    """
    # ========================================================================
    # STEP 1: Extract market caps for cap-weighted means
    # ========================================================================
    logger.info("Extracting market capitalizations for cap-weighted means...")

    market_caps = []
    for item in raw_data:
        info = item.get('info', {})
        instrument = item.get('instrument')

        # Try to get market cap from yfinance info (preferred)
        market_cap = info.get('marketCap') if info else None

        # Fallback to max_open_quantity as proxy
        if market_cap is None or market_cap <= 0:
            if instrument and hasattr(instrument, 'max_open_quantity'):
                market_cap = instrument.max_open_quantity

        # Default to 1.0 if no market cap available (equal weight for missing)
        if market_cap is None or market_cap <= 0:
            market_cap = 1.0

        market_caps.append(float(market_cap))

    # Convert to numpy array and normalize to weights
    market_caps_arr = np.array(market_caps)
    total_market_cap = np.sum(market_caps_arr)
    cap_weights = market_caps_arr / total_market_cap if total_market_cap > 0 else np.ones(len(market_caps)) / len(market_caps)

    n_with_caps = np.sum(market_caps_arr > 1.0)
    logger.info(f"✓ Market caps extracted: {n_with_caps}/{len(raw_data)} stocks have valid market cap data")
    logger.info(f"  Total market cap: ${total_market_cap/1e9:.1f}B")
    logger.info(f"  Largest stock weight: {np.max(cap_weights):.2%}")
    logger.info(f"  Smallest stock weight: {np.min(cap_weights):.2%}")

    # ========================================================================
    # STEP 2: Extract metrics with alignment to raw_data
    # ========================================================================
    value_metrics = _extract_value_metrics_aligned(raw_data)
    momentum_metrics = _extract_momentum_metrics_aligned(raw_data)
    quality_metrics = _extract_quality_metrics_aligned(raw_data)
    growth_metrics = _extract_growth_metrics_aligned(raw_data)

    # ========================================================================
    # STEP 3: Calculate robust statistics with cap-weighted means
    # ========================================================================
    def calculate_robust_stats(
        values: List[float],
        weights: np.ndarray,
        metric_name: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate mean/std with iterative ±3σ outlier removal.

        Uses:
        - Cap-weighted mean (MSCI Barra approach)
        - Equal-weighted std (theory specification)
        """
        if len(values) == 0 or len(values) < 10:
            return None, None

        arr = np.array(values)
        weights_arr = np.array(weights)

        # Filter finite values and align weights
        finite_mask = np.isfinite(arr)
        arr = arr[finite_mask]
        weights_arr = weights_arr[finite_mask]

        if len(arr) < 10:
            return None, None

        # Renormalize weights after filtering
        weights_arr = weights_arr / np.sum(weights_arr)

        original_n = len(arr)

        # Iteratively remove outliers (±3σ) - 3 iterations
        for iteration in range(3):
            # Cap-weighted mean (MSCI Barra approach)
            mean = np.average(arr, weights=weights_arr)

            # Equal-weighted std (theory specification)
            std = np.std(arr, ddof=1)

            if std == 0:
                break

            z_scores = (arr - mean) / std
            mask = np.abs(z_scores) <= 3.0
            new_arr = arr[mask]
            new_weights = weights_arr[mask]

            if len(new_arr) == len(arr):
                break  # No more outliers

            arr = new_arr
            weights_arr = new_weights
            # Renormalize weights after removing outliers
            weights_arr = weights_arr / np.sum(weights_arr)

        if len(arr) < 10:
            return None, None

        # Final statistics
        robust_mean = float(np.average(arr, weights=weights_arr))  # Cap-weighted
        robust_std = float(np.std(arr, ddof=1))  # Equal-weighted
        n_removed = original_n - len(arr)

        logger.info(
            f"  {metric_name:20} μ={robust_mean:+.6f}, σ={robust_std:.6f} "
            f"(n={len(arr)}, removed {n_removed} outliers)"
        )

        return robust_mean, robust_std

    # Calculate statistics for all metrics
    logger.info("\nValue factor metrics (cap-weighted means):")
    cross_sectional_stats = {
        'value': {},
        'momentum': {},
        'quality': {},
        'growth': {}
    }

    # Value metrics
    for metric_name, values in value_metrics.items():
        mean, std = calculate_robust_stats(values, cap_weights, metric_name)
        if mean is not None and std is not None:
            cross_sectional_stats['value'][metric_name] = (mean, std)

    # Momentum metrics
    logger.info("\nMomentum factor metrics (cap-weighted means):")
    for metric_name, values in momentum_metrics.items():
        mean, std = calculate_robust_stats(values, cap_weights, metric_name)
        if mean is not None and std is not None:
            cross_sectional_stats['momentum'][metric_name] = (mean, std)

    # Quality metrics
    logger.info("\nQuality factor metrics (cap-weighted means):")
    for metric_name, values in quality_metrics.items():
        mean, std = calculate_robust_stats(values, cap_weights, metric_name)
        if mean is not None and std is not None:
            cross_sectional_stats['quality'][metric_name] = (mean, std)

    # Growth metrics
    logger.info("\nGrowth factor metrics (cap-weighted means):")
    for metric_name, values in growth_metrics.items():
        mean, std = calculate_robust_stats(values, cap_weights, metric_name)
        if mean is not None and std is not None:
            cross_sectional_stats['growth'][metric_name] = (mean, std)

    return cross_sectional_stats


def _extract_value_metrics_aligned(raw_data: List[dict]) -> dict:
    """
    Extract value metrics maintaining alignment with raw_data indices.

    Returns dictionary with metric_name -> list of values (aligned with raw_data).
    Missing values are filled with np.nan to maintain alignment.
    """
    n = len(raw_data)
    metrics = {
        'book_price': np.full(n, np.nan),
        'earnings_price': np.full(n, np.nan),
        'sales_price': np.full(n, np.nan),
        'dividend_price': np.full(n, np.nan)
    }

    for i, item in enumerate(raw_data):
        info = item.get('info', {})
        if not info:
            continue

        # Book/Price (inverse of P/B)
        try:
            price_to_book = float(info.get('priceToBook', 0))
            if price_to_book > 0:
                metrics['book_price'][i] = 1.0 / price_to_book
        except (ValueError, TypeError):
            pass

        # Earnings/Price (inverse of P/E)
        try:
            trailing_pe = float(info.get('trailingPE', 0))
            if trailing_pe > 0:
                metrics['earnings_price'][i] = 1.0 / trailing_pe
        except (ValueError, TypeError):
            pass

        # Sales/Price (inverse of P/S)
        try:
            price_to_sales = float(info.get('priceToSalesTrailing12Months', 0))
            if price_to_sales > 0:
                metrics['sales_price'][i] = 1.0 / price_to_sales
        except (ValueError, TypeError):
            pass

        # Dividend/Price (dividend yield)
        try:
            dividend_yield = float(info.get('dividendYield', 0))
            if dividend_yield > 0:
                metrics['dividend_price'][i] = dividend_yield
        except (ValueError, TypeError):
            pass

    return metrics


def _extract_momentum_metrics_aligned(raw_data: List[dict]) -> dict:
    """Extract momentum metrics maintaining alignment with raw_data indices."""
    n = len(raw_data)
    metrics = {'momentum_12m': np.full(n, np.nan)}

    for i, item in enumerate(raw_data):
        technical_metrics = item.get('technical_metrics', {})
        if technical_metrics and 'momentum_12m_minus_1m' in technical_metrics:
            try:
                mom_12m = technical_metrics.get('momentum_12m_minus_1m')
                if mom_12m is not None:
                    metrics['momentum_12m'][i] = float(mom_12m)
            except (ValueError, TypeError):
                pass

    return metrics


def _extract_quality_metrics_aligned(raw_data: List[dict]) -> dict:
    """Extract quality metrics maintaining alignment with raw_data indices."""
    n = len(raw_data)
    metrics = {
        'roe': np.full(n, np.nan),
        'profit_margin': np.full(n, np.nan),
        'sharpe_ratio': np.full(n, np.nan)
    }

    for i, item in enumerate(raw_data):
        info = item.get('info', {})
        technical_metrics = item.get('technical_metrics', {})

        if info:
            try:
                roe = info.get('returnOnEquity')
                if roe is not None:
                    metrics['roe'][i] = float(roe)
            except (ValueError, TypeError):
                pass

            try:
                margin = info.get('profitMargins')
                if margin is not None:
                    metrics['profit_margin'][i] = float(margin)
            except (ValueError, TypeError):
                pass

        if technical_metrics and 'sharpe_ratio' in technical_metrics:
            try:
                sharpe = technical_metrics.get('sharpe_ratio')
                if sharpe is not None:
                    metrics['sharpe_ratio'][i] = float(sharpe)
            except (ValueError, TypeError):
                pass

    return metrics


def _extract_growth_metrics_aligned(raw_data: List[dict]) -> dict:
    """Extract growth metrics maintaining alignment with raw_data indices."""
    n = len(raw_data)
    metrics = {
        'revenue_growth': np.full(n, np.nan),
        'earnings_growth': np.full(n, np.nan)
    }

    for i, item in enumerate(raw_data):
        info = item.get('info', {})
        if not info:
            continue

        try:
            rev_growth = info.get('revenueGrowth')
            if rev_growth is not None:
                # Convert to float and clip extreme growth values
                rev_growth_float = float(rev_growth)
                rev_growth_clipped = float(np.clip(rev_growth_float, -0.50, 1.00))
                metrics['revenue_growth'][i] = rev_growth_clipped
        except (ValueError, TypeError):
            pass

        try:
            earn_growth = info.get('earningsGrowth')
            if earn_growth is not None:
                # Convert to float and clip extreme growth values
                earn_growth_float = float(earn_growth)
                earn_growth_clipped = float(np.clip(earn_growth_float, -0.50, 1.00))
                metrics['earnings_growth'][i] = earn_growth_clipped
        except (ValueError, TypeError):
            pass

    return metrics


async def recalculate_with_cross_sectional_stats(
    raw_data: List[dict],
    signal_date: date_type,
    calculator_with_cs,
    total_parallel: int,
    stats: dict
) -> List[dict]:
    """
    Pass 1B: Recalculate z-scores using robust cross-sectional statistics.

    Args:
        raw_data: Raw fundamentals from Pass 1
        signal_date: Target signal date
        calculator_with_cs: Calculator initialized with cross-sectional stats
        total_parallel: Batch size for parallel processing
        stats: Statistics tracker

    Returns:
        List of dicts with added fields: raw_z, factor_zscores
    """
    from tqdm.asyncio import tqdm as async_tqdm

    raw_data_cs = []

    with async_tqdm(
        total=len(raw_data),
        desc="PASS 1B: Calculating z-scores",
        unit="stock"
    ) as pbar:
        # Process in parallel batches
        for batch_start in range(0, len(raw_data), total_parallel):
            batch_end = min(batch_start + total_parallel, len(raw_data))
            batch = raw_data[batch_start:batch_end]

            # Create tasks for this batch
            tasks = []
            for item in batch:
                instrument = item['instrument']
                if not instrument.yfinance_ticker:
                    continue

                task = calculator_with_cs.calculate_raw_composite_zscore(
                    yf_ticker=instrument.yfinance_ticker,
                    target_date=signal_date
                )
                tasks.append((instrument, task))

            # Wait for all tasks
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )

            # Store results
            for (instrument, _), result in zip(tasks, results):
                if isinstance(result, BaseException):
                    logger.debug(f"Error recalculating z-score for {instrument.ticker}: {result}")
                    stats['errors'] += 1
                    continue

                if result is None:
                    continue

                # Unpack result from calculate_raw_composite_zscore
                raw_z, technical_metrics, info, country, factor_zscores = result
                raw_data_cs.append({
                    'instrument': instrument,
                    'raw_z': raw_z,
                    'technical_metrics': technical_metrics,
                    'info': info,
                    'country': country,
                    'factor_zscores': factor_zscores
                })

            pbar.update(len(batch))

    return raw_data_cs


async def apply_cross_sectional_standardization(
    raw_data: List[dict],
    calculator
) -> np.ndarray:
    """
    Pass 2: Apply winsorization + StandardScaler to raw z-scores.

    Chapter 5 specifies ensuring mean=0, std=1 across universe.

    Args:
        raw_data: Data with raw_z field
        calculator: Calculator instance with standardize method

    Returns:
        Standardized z-scores as numpy array
    """
    # Extract raw z-scores
    raw_zscores = [item['raw_z'] for item in raw_data]

    # Apply winsorization + StandardScaler via calculator method
    z_standardized = calculator.standardize_zscores_cross_sectional(
        raw_zscores=raw_zscores,
        winsorize_threshold=10.0
    )

    # Verify standardization
    mean = np.mean(z_standardized)
    std = np.std(z_standardized, ddof=1)

    logger.info(f"Cross-sectional distribution after standardization:")
    logger.info(f"  Mean: {mean:+.6f} (target: 0.000)")
    logger.info(f"  Std:  {std:.6f} (target: 1.000)")

    return z_standardized


async def calculate_robust_factor_stats(
    raw_data: List[dict]
) -> dict:
    """
    Pass 2.5: Calculate robust statistics for factor z-scores.

    Uses iterative outlier removal at ±3σ for each factor.

    Args:
        raw_data: Data with factor_zscores field

    Returns:
        Dict with robust mean/std for each factor
    """
    # Extract all factor z-scores
    all_value_z = []
    all_momentum_z = []
    all_quality_z = []
    all_growth_z = []

    for item in raw_data:
        fz = item.get('factor_zscores', {})
        all_value_z.append(fz.get('value', 0.0))
        all_momentum_z.append(fz.get('momentum', 0.0))
        all_quality_z.append(fz.get('quality', 0.0))
        all_growth_z.append(fz.get('growth', 0.0))

    # Calculate robust statistics (iteratively remove outliers)
    def calculate_robust_stats(values, max_iterations=3, std_threshold=3.0):
        arr = np.array(values)
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            return 0.0, 1.0, 0

        original_n = len(arr)

        for _ in range(max_iterations):
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)

            if std == 0:
                break

            z_scores = (arr - mean) / std
            mask = np.abs(z_scores) <= std_threshold
            new_arr = arr[mask]

            if len(new_arr) == len(arr):
                break

            arr = new_arr

        robust_mean = float(np.mean(arr))
        robust_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        n_removed = original_n - len(arr)

        return robust_mean, robust_std, n_removed

    val_robust_mean, val_robust_std, val_outliers = calculate_robust_stats(all_value_z)
    mom_robust_mean, mom_robust_std, mom_outliers = calculate_robust_stats(all_momentum_z)
    qual_robust_mean, qual_robust_std, qual_outliers = calculate_robust_stats(all_quality_z)
    growth_robust_mean, growth_robust_std, growth_outliers = calculate_robust_stats(all_growth_z)

    logger.info("Robust factor statistics (after outlier removal):")
    logger.info(f"  Valuation:  μ={val_robust_mean:+.3f}, σ={val_robust_std:.3f} ({val_outliers} outliers removed)")
    logger.info(f"  Momentum:   μ={mom_robust_mean:+.3f}, σ={mom_robust_std:.3f} ({mom_outliers} outliers removed)")
    logger.info(f"  Quality:    μ={qual_robust_mean:+.3f}, σ={qual_robust_std:.3f} ({qual_outliers} outliers removed)")
    logger.info(f"  Growth:     μ={growth_robust_mean:+.3f}, σ={growth_robust_std:.3f} ({growth_outliers} outliers removed)")

    return {
        'valuation': (val_robust_mean, val_robust_std),
        'momentum': (mom_robust_mean, mom_robust_std),
        'quality': (qual_robust_mean, qual_robust_std),
        'growth': (growth_robust_mean, growth_robust_std)
    }


async def analyze_factor_correlations(
    raw_data: List[dict]
) -> None:
    """
    Pass 2.6: Analyze factor correlations and validate Chapter 5 expectations.

    Chapter 5 (lines 452-461) specifies expected correlations:
    - Value-Momentum: -0.2 to -0.4 (negative, enabling combination)
    - Value-Quality: 0.0 to 0.2 (low positive)
    - Momentum-Quality: 0.2 to 0.4 (positive but diversifying)

    Args:
        raw_data: Data with factor_zscores field
    """
    # Extract all factor z-scores
    all_value_z = []
    all_momentum_z = []
    all_quality_z = []
    all_growth_z = []

    for item in raw_data:
        fz = item.get('factor_zscores', {})
        all_value_z.append(fz.get('value', 0.0))
        all_momentum_z.append(fz.get('momentum', 0.0))
        all_quality_z.append(fz.get('quality', 0.0))
        all_growth_z.append(fz.get('growth', 0.0))

    # Calculate factor correlation matrix
    if len(all_value_z) >= 30:
        factor_matrix = np.column_stack([
            np.array(all_value_z),
            np.array(all_momentum_z),
            np.array(all_quality_z),
            np.array(all_growth_z)
        ])

        # Remove rows with NaN/inf
        mask = np.all(np.isfinite(factor_matrix), axis=1)
        factor_matrix_clean = factor_matrix[mask]

        if len(factor_matrix_clean) >= 30:
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(factor_matrix_clean.T)

            logger.info("Factor correlation matrix:")
            logger.info("                  Value    Momentum  Quality   Growth")
            logger.info(f"  Value        {correlation_matrix[0,0]:+7.3f}  {correlation_matrix[0,1]:+7.3f}  {correlation_matrix[0,2]:+7.3f}  {correlation_matrix[0,3]:+7.3f}")
            logger.info(f"  Momentum     {correlation_matrix[1,0]:+7.3f}  {correlation_matrix[1,1]:+7.3f}  {correlation_matrix[1,2]:+7.3f}  {correlation_matrix[1,3]:+7.3f}")
            logger.info(f"  Quality      {correlation_matrix[2,0]:+7.3f}  {correlation_matrix[2,1]:+7.3f}  {correlation_matrix[2,2]:+7.3f}  {correlation_matrix[2,3]:+7.3f}")
            logger.info(f"  Growth       {correlation_matrix[3,0]:+7.3f}  {correlation_matrix[3,1]:+7.3f}  {correlation_matrix[3,2]:+7.3f}  {correlation_matrix[3,3]:+7.3f}")

            # Validate expected relationships per Chapter 5
            value_momentum_corr = correlation_matrix[0, 1]
            quality_value_corr = correlation_matrix[2, 0]
            quality_momentum_corr = correlation_matrix[2, 1]
            growth_momentum_corr = correlation_matrix[3, 1]

            logger.info("")
            logger.info("Correlation analysis (Chapter 5 expectations):")
            logger.info(f"  Value-Momentum:       {value_momentum_corr:+.3f}  (expected: -0.2 to -0.4, negative)")
            logger.info(f"  Quality-Value:        {quality_value_corr:+.3f}  (expected: 0.0 to 0.2, low positive)")
            logger.info(f"  Quality-Momentum:     {quality_momentum_corr:+.3f}  (expected: 0.2 to 0.4, positive)")
            logger.info(f"  Growth-Momentum:      {growth_momentum_corr:+.3f}  (expected: near 0, orthogonal)")

            # Warnings for unexpected correlations
            if value_momentum_corr > 0.1:
                logger.warning(f"⚠️  Value-Momentum correlation is positive ({value_momentum_corr:+.3f}), expected negative per Chapter 5")

            if abs(quality_value_corr) > 0.5 or abs(quality_momentum_corr) > 0.6:
                logger.warning(f"⚠️  Quality factor is highly correlated with other factors (not orthogonal)")


async def classify_and_save_signals(
    raw_data: List[dict],
    z_standardized: np.ndarray,
    signal_date: date_type,
    update_existing: bool,
    calculator,
    stats: dict
) -> None:
    """
    Pass 3: Classify signals using cross-sectionally standardized z-scores and save with momentum filters.

    KEY CHANGE: Now uses the standardized z-scores from Pass 2 (mean=0, std=1) instead of recalculating.
    This aligns with Chapter 5 theory - no second standardization needed after factor combination.

    Implements 3-level classification system:
    - Level 1: Percentile-based classification from PRE-STANDARDIZED z-score
    - Level 2: Negative momentum filter (-15% threshold downgrade)
    - Level 3: Positive momentum boost (+40% threshold upgrade)

    Args:
        raw_data: Full data with all fields (instrument, factor_zscores, technical_metrics, info, country)
        z_standardized: Standardized composite z-scores from Pass 2 (mean=0, std=1)
        signal_date: Target date
        update_existing: Whether to update existing signals
        calculator: Calculator instance (used for helper functions and classifier)
        stats: Statistics tracker
    """
    from tqdm.asyncio import tqdm as async_tqdm
    from baml_client.types import StockSignalOutput, SignalType, SignalDrivers, RiskFactors
    from src.stock_analyzer.data.fetchers import fetch_price_data, fetch_macro_regime
    from src.stock_analyzer.adjustments.risk import calculate_risk_factors, calculate_confidence
    from src.stock_analyzer.classification.scoring import (
        calculate_upside_potential,
        calculate_downside_risk,
        calculate_data_quality,
        generate_analysis_notes,
    )

    # Validate array sizes match
    if len(z_standardized) != len(raw_data):
        logger.error(
            f"Size mismatch: z_standardized has {len(z_standardized)} items, "
            f"raw_data has {len(raw_data)} items"
        )
        return

    logger.info(f"Using pre-calculated standardized z-scores (mean={np.mean(z_standardized):.3f}, std={np.std(z_standardized, ddof=1):.3f})")

    # Process and save signals with progress bar
    with async_tqdm(
        total=len(raw_data),
        desc="PASS 3: Saving signals",
        unit="stock"
    ) as pbar:
        with database_manager.get_session() as session:
            for i, item in enumerate(raw_data):
                instrument = item['instrument']
                technical_metrics = item.get('technical_metrics', {})
                info = item.get('info', {})
                country = item.get('country', 'USA')
                factor_zscores = item.get('factor_zscores', {})

                # Get the standardized z-score for this stock
                composite_z = float(z_standardized[i])

                try:
                    # Level 1: Classify using PRE-STANDARDIZED z-score
                    signal_type = calculator.classifier.classify(composite_z)

                    # Fetch latest price data for close/open/volume fields
                    stock_data, _, _ = await fetch_price_data(instrument.yfinance_ticker, period="1y")
                    if stock_data is None or len(stock_data) < 10:
                        logger.warning(f"Insufficient price data for {instrument.ticker}, skipping")
                        stats['errors'] += 1
                        pbar.update(1)
                        continue

                    # Fetch macro data for adjustments (lightweight)
                    macro_data = await fetch_macro_regime(country) if country else None

                    # Calculate confidence and risk
                    composite_score = 50 + composite_z * 15  # Convert to 0-100 scale
                    confidence_level = calculate_confidence(
                        technical_metrics, composite_score, stock_data
                    )
                    risk_factors = calculate_risk_factors(technical_metrics, info or {}, stock_data)

                    # Calculate upside/downside
                    upside_potential_pct = calculate_upside_potential(
                        composite_score, technical_metrics, macro_data
                    )
                    downside_risk_pct = calculate_downside_risk(
                        composite_score, technical_metrics, macro_data
                    )

                    # Extract factor z-scores
                    value_z = factor_zscores.get('value', 0.0)
                    momentum_z = factor_zscores.get('momentum', 0.0)
                    quality_z = factor_zscores.get('quality', 0.0)
                    growth_z = factor_zscores.get('growth', 0.0)

                    # Generate analysis notes
                    analysis_notes = generate_analysis_notes(
                        instrument.yfinance_ticker,
                        signal_type,
                        composite_z,
                        value_z,
                        momentum_z,
                        quality_z,
                        growth_z,
                        technical_metrics,
                        macro_data,
                    )

                    # Build StockSignalOutput using pre-calculated data
                    signal_output = StockSignalOutput(
                        signal_date=signal_date.isoformat(),
                        signal_type=signal_type,
                        confidence_level=confidence_level,
                        # Price data
                        close_price=float(stock_data['Close'].iloc[-1]),
                        open_price=float(stock_data['Open'].iloc[-1]) if 'Open' in stock_data else None,
                        daily_return=(
                            float(stock_data['Close'].pct_change().iloc[-1])
                            if len(stock_data) > 1
                            else 0.0
                        ),
                        volume=float(stock_data['Volume'].iloc[-1]) if 'Volume' in stock_data else None,
                        # Technical indicators
                        volatility=technical_metrics.get('volatility'),
                        rsi=technical_metrics.get('rsi'),
                        # Metadata
                        data_quality_score=calculate_data_quality(stock_data, info or {}),
                        analysis_notes=analysis_notes,
                        # Signal Drivers (using pre-calculated factor z-scores)
                        signal_drivers=SignalDrivers(
                            valuation_score=np.tanh(value_z / 2),
                            valuation_summary=f"Value z-score: {value_z:.2f} (B/P, E/P, FCF/P)",
                            momentum_score=np.tanh(momentum_z / 2),
                            momentum_summary=(
                                f"Momentum z-score: {momentum_z:.2f} "
                                f"(12-1m: {technical_metrics.get('momentum_12m_minus_1m', 0)*100:.1f}%)"
                            ),
                            quality_score=(np.tanh(quality_z / 2) + 1) / 2,
                            quality_summary=(
                                f"Quality z-score: {quality_z:.2f} "
                                f"(ROE, margins, Sharpe: {technical_metrics.get('sharpe_ratio', 0):.2f})"
                            ),
                            growth_score=np.tanh(growth_z / 2),
                            growth_summary=f"Growth z-score: {growth_z:.2f} (Revenue/earnings growth)",
                            technical_score=np.tanh(composite_z / 2),
                            technical_summary=f"Composite z-score: {composite_z:.2f} (cross-sectional standardized)",
                            analyst_score=None,
                            analyst_summary="Institutional four-factor model (no analyst data)",
                        ),
                        # Risk Factors
                        risk_factors=risk_factors,
                        # Price targets
                        upside_potential_pct=upside_potential_pct,
                        downside_risk_pct=downside_risk_pct,
                    )

                    # Level 2: Negative Momentum Filter (-15% threshold)
                    return_1y = technical_metrics.get('annualized_return', 0.0) if technical_metrics else 0.0

                    # Get original signal type
                    original_signal = signal_output.signal_type

                    # Apply momentum filters
                    if return_1y < -0.15 and original_signal.value in ['SMALL_GAIN', 'LARGE_GAIN']:
                        logger.debug(
                            f"{instrument.ticker}: Downgraded to NEUTRAL due to poor momentum "
                            f"({return_1y*100:.1f}%)"
                        )
                        signal_output.signal_type = SignalType.NEUTRAL

                    # Level 3: Positive Momentum Boost (+40% threshold)
                    if return_1y > 0.40 and original_signal.value == 'SMALL_GAIN':
                        logger.debug(
                            f"{instrument.ticker}: Upgraded to LARGE_GAIN due to exceptional momentum "
                            f"(+{return_1y*100:.1f}%)"
                        )
                        signal_output.signal_type = SignalType.LARGE_GAIN

                    # Save to database
                    success = save_signal(
                        session,
                        instrument.id,
                        signal_output,
                        technical_metrics,
                        update_if_exists=update_existing,
                        instrument=instrument
                    )

                    if success:
                        stats['signals_generated'] += 1
                        stats['by_signal_type'][signal_output.signal_type.value] += 1
                    else:
                        stats['skipped_duplicate'] += 1

                except Exception as e:
                    logger.error(f"Error saving signal for {instrument.ticker}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    stats['errors'] += 1

                pbar.update(1)

            # Commit all changes
            session.commit()
