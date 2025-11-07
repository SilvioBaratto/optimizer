"""
Factor Calculators
==================

Implements the four-factor institutional framework with z-score standardization.

Each factor calculator:
1. Extracts relevant metrics from stock data/info
2. Standardizes via z-score relative to market norms
3. Returns normalized factor score

Supports both static market norms (default) and dynamic cross-sectional stats.
"""

from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from ..data.fetchers import fetch_economic_forecasts
from ..utils import safe_float

logger = logging.getLogger(__name__)


def calculate_value_factor(
    info: Optional[Dict],
    stock_data: pd.DataFrame,
    cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
) -> float:
    """
    Calculate value factor per Chapter 5, Section 5.2.2.

    Chapter 5 specifies equal-weight combination of 4 metrics:
        Value_n = 0.25·Z(B/P) + 0.25·Z(E/P) + 0.25·Z(S/P) + 0.25·Z(D/P)

    Higher values indicate cheaper/more attractive valuations.

    Z-Score Methodology:
    Production systems use cross-sectional standardization:
        z_i,t = (F_i,t - μ_t) / σ_t
    where μ_t and σ_t are computed across the universe at time t.

    Current implementation uses historical S&P 500 norms as fallback:
        - B/P: μ=0.25 (P/B ~4.0), σ=0.15
        - E/P: μ=0.05 (P/E ~20), σ=0.03
        - S/P: μ=0.50 (P/S ~2.0), σ=0.30
        - D/P: μ=0.02 (2% yield), σ=0.015

    Missing Data Handling (Chapter 5, lines 17-19):
    - Dividend yield may be 0 for growth stocks (structural, not missing)
    - Average only available metrics (no imputation at descriptor level)
    - Typical stock: 3-4 metrics available

    Outlier Handling (MSCI Barra USE4 Three-Group Methodology):
    Layer 1: Data validation - Remove impossible raw values
    Layer 2: Descriptor z-score winsorization (>10σ removed, 3-10σ winsorized to ±3σ)
    Layer 3: Factor z-score winsorization (composite factor capped at ±5σ)

    Args:
        info: Stock info dict from yfinance
        stock_data: Historical price data (not used, included for consistency)
        cross_sectional_stats: Optional universe-level statistics

    Returns:
        Value factor z-score (average of available metrics)
    """
    if not info:
        return 0.0  # Neutral if no data

    z_scores = []
    value_stats = cross_sectional_stats.get('value', {}) if cross_sectional_stats else {}

    # 1. Book/Price ratio (inverse of P/B)
    price_to_book = safe_float(info.get('priceToBook'))
    if price_to_book > 0:
        # LAYER 1: Data validation - Remove impossible values
        if price_to_book < 0.01 or price_to_book > 100:
            logger.debug(
                f"Value factor: Invalid P/B ratio {price_to_book:.4f} "
                f"(outside [0.01, 100]), skipping B/P descriptor"
            )
        else:
            book_price = 1 / price_to_book
            bp_mean, bp_std = value_stats.get('book_price', (0.25, 0.15))
            z_bp = (book_price - bp_mean) / bp_std if bp_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_bp) > 10.0:
                logger.warning(
                    f"Value factor: Removing extreme B/P z-score {z_bp:+.2f} (>10σ, likely data error)"
                )
                # Don't append - remove completely per MSCI USE4
            elif abs(z_bp) > 3.0:
                z_bp_original = z_bp
                z_bp = 3.0 * np.sign(z_bp)
                logger.debug(
                    f"Value factor: Winsorized B/P z-score {z_bp_original:+.2f} → {z_bp:+.2f} (±3σ)"
                )
                z_scores.append(z_bp)
            else:
                z_scores.append(z_bp)

    # 2. Earnings/Price ratio (inverse of P/E)
    trailing_pe = safe_float(info.get('trailingPE'))
    if trailing_pe > 0:
        # LAYER 1: Data validation
        if trailing_pe < 0.1 or trailing_pe > 1000:
            logger.debug(
                f"Value factor: Invalid P/E ratio {trailing_pe:.4f} "
                f"(outside [0.1, 1000]), skipping E/P descriptor"
            )
        else:
            earnings_price = 1 / trailing_pe
            ep_mean, ep_std = value_stats.get('earnings_price', (0.05, 0.03))
            z_ep = (earnings_price - ep_mean) / ep_std if ep_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_ep) > 10.0:
                logger.warning(
                    f"Value factor: Removing extreme E/P z-score {z_ep:+.2f} (>10σ, likely data error)"
                )
            elif abs(z_ep) > 3.0:
                z_ep_original = z_ep
                z_ep = 3.0 * np.sign(z_ep)
                logger.debug(
                    f"Value factor: Winsorized E/P z-score {z_ep_original:+.2f} → {z_ep:+.2f} (±3σ)"
                )
                z_scores.append(z_ep)
            else:
                z_scores.append(z_ep)

    # 3. Sales/Price ratio (inverse of P/S)
    price_to_sales = safe_float(info.get('priceToSalesTrailing12Months'))
    if price_to_sales > 0:
        # LAYER 1: Data validation
        if price_to_sales < 0.01 or price_to_sales > 100:
            logger.debug(
                f"Value factor: Invalid P/S ratio {price_to_sales:.4f} "
                f"(outside [0.01, 100]), skipping S/P descriptor"
            )
        else:
            sales_price = 1 / price_to_sales
            sp_mean, sp_std = value_stats.get('sales_price', (0.50, 0.30))
            z_sp = (sales_price - sp_mean) / sp_std if sp_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_sp) > 10.0:
                logger.warning(
                    f"Value factor: Removing extreme S/P z-score {z_sp:+.2f} (>10σ, likely data error)"
                )
            elif abs(z_sp) > 3.0:
                z_sp_original = z_sp
                z_sp = 3.0 * np.sign(z_sp)
                logger.debug(
                    f"Value factor: Winsorized S/P z-score {z_sp_original:+.2f} → {z_sp:+.2f} (±3σ)"
                )
                z_scores.append(z_sp)
            else:
                z_scores.append(z_sp)

    # 4. Dividend/Price ratio (dividend yield)
    # Note: ~40% of stocks don't pay dividends (growth/tech stocks)
    # This is structural, not missing data - we handle by averaging available metrics
    dividend_yield = safe_float(info.get('dividendYield'))
    if dividend_yield > 0:
        # LAYER 1: Data validation (dividend yields typically 0-10%)
        if dividend_yield > 0.10:  # 10% yield cap (anything higher is likely data error)
            logger.debug(
                f"Value factor: Invalid dividend yield {dividend_yield*100:.1f}% "
                f"(>10%), skipping D/P descriptor"
            )
        else:
            dp_mean, dp_std = value_stats.get('dividend_price', (0.02, 0.015))
            z_dp = (dividend_yield - dp_mean) / dp_std if dp_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_dp) > 10.0:
                logger.warning(
                    f"Value factor: Removing extreme D/P z-score {z_dp:+.2f} (>10σ, likely data error)"
                )
            elif abs(z_dp) > 3.0:
                z_dp_original = z_dp
                z_dp = 3.0 * np.sign(z_dp)
                logger.debug(
                    f"Value factor: Winsorized D/P z-score {z_dp_original:+.2f} → {z_dp:+.2f} (±3σ)"
                )
                z_scores.append(z_dp)
            else:
                z_scores.append(z_dp)

    # Average z-scores of available metrics (equal weight)
    # Typical stock: 3-4 metrics (dividend optional)
    if len(z_scores) == 0:
        return 0.0

    value_z = float(np.mean(z_scores))

    # LAYER 3: Factor-level winsorization (composite factor)
    # More conservative: cap at ±5σ (prevents factor from dominating composite)
    if abs(value_z) > 5.0:
        value_z_original = value_z
        value_z = 5.0 * np.sign(value_z)
        logger.warning(
            f"Value factor: Capped composite value factor {value_z_original:+.2f} → {value_z:+.2f} (±5σ)"
        )

    return value_z


def calculate_momentum_factor(
    metrics: Dict,
    cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
) -> float:
    """
    Calculate momentum factor per Section 5.2.5.

    Uses 12-1 month returns (skip most recent month to avoid reversal).

    Z-Score Methodology:
    Production systems standardize 12-1m returns across the universe:
        z_i,t = (r_i,t - μ_t) / σ_t

    Current implementation uses historical market norms:
        - Mean: 8% annually
        - Std: 20%

    Outlier Handling (MSCI Barra USE4):
    Layer 1: Data validation - Cap extreme returns (>500% or <-95%)
    Layer 2: Z-score winsorization (>10σ removed, 3-10σ winsorized to ±3σ)
    Layer 3: Factor-level winsorization (capped at ±5σ)

    Args:
        metrics: Technical metrics dict (contains momentum_12m_minus_1m)
        cross_sectional_stats: Optional universe-level statistics

    Returns:
        Momentum factor z-score
    """
    # Get 12-1 month momentum
    momentum_12m = metrics.get('momentum_12m_minus_1m', 0)

    # LAYER 1: Data validation - Cap extreme returns (likely data errors)
    if abs(momentum_12m) > 5.0:  # >500% return is suspicious
        logger.debug(
            f"Momentum factor: Extreme return {momentum_12m*100:+.1f}%, "
            f"capping at ±500%"
        )
        momentum_12m = 5.0 * np.sign(momentum_12m)

    # Use cross-sectional stats if available, otherwise static norms
    momentum_stats = cross_sectional_stats.get('momentum', {}) if cross_sectional_stats else {}
    mom_mean, mom_std = momentum_stats.get('momentum_12m', (0.08, 0.20))

    # Standardize
    z_momentum = (momentum_12m - mom_mean) / mom_std if mom_std > 0 else 0.0

    # LAYER 2: MSCI Barra USE4 z-score winsorization
    if abs(z_momentum) > 10.0:
        logger.warning(
            f"Momentum factor: Removing extreme z-score {z_momentum:+.2f} (>10σ, likely data error)"
        )
        return 0.0  # Remove completely
    elif abs(z_momentum) > 3.0:
        z_momentum_original = z_momentum
        z_momentum = 3.0 * np.sign(z_momentum)
        logger.debug(
            f"Momentum factor: Winsorized z-score {z_momentum_original:+.2f} → {z_momentum:+.2f} (±3σ)"
        )

    # LAYER 3: Factor-level cap (redundant for single descriptor, but consistent)
    if abs(z_momentum) > 5.0:
        z_momentum = 5.0 * np.sign(z_momentum)

    return z_momentum


def calculate_quality_factor(
    info: Optional[Dict],
    metrics: Dict,
    country: Optional[str],
    cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
) -> float:
    """
    Calculate quality factor per Section 5.2.4.

    Uses profitability metrics:
    - ROE (Return on Equity)
    - Profit margins
    - Sharpe ratio (earnings quality)

    Enhancements:
    - Inflation risk adjustment (penalizes when inflation_forecast_6m > 3%)
    - Distress penalty for companies showing financial distress

    Args:
        info: Stock info dict from yfinance
        metrics: Technical metrics dict
        country: Country name (for inflation adjustment)
        cross_sectional_stats: Optional universe-level statistics

    Returns:
        Quality factor z-score
    """
    if not info:
        # Fallback to Sharpe ratio only
        sharpe = metrics.get('sharpe_ratio', 0)
        return (sharpe - 0.5) / 0.5

    z_scores = []
    quality_stats = cross_sectional_stats.get('quality', {}) if cross_sectional_stats else {}

    # 1. ROE (Return on Equity)
    roe = safe_float(info.get('returnOnEquity'))
    if roe != 0.0:  # safe_float returns 0.0 for invalid/None values
        # LAYER 1: Data validation - ROE typically in [-200%, +100%]
        if abs(roe) > 2.0:
            logger.debug(
                f"Quality factor: Extreme ROE {roe*100:+.1f}%, capping at ±200%"
            )
            roe = 2.0 * np.sign(roe)

        roe_mean, roe_std = quality_stats.get('roe', (0.15, 0.10))
        z_roe = (roe - roe_mean) / roe_std if roe_std > 0 else 0.0

        # LAYER 2: MSCI Barra USE4 z-score winsorization
        if abs(z_roe) > 10.0:
            logger.warning(
                f"Quality factor: Removing extreme ROE z-score {z_roe:+.2f} (>10σ, likely data error)"
            )
        elif abs(z_roe) > 3.0:
            z_roe_original = z_roe
            z_roe = 3.0 * np.sign(z_roe)
            logger.debug(
                f"Quality factor: Winsorized ROE z-score {z_roe_original:+.2f} → {z_roe:+.2f} (±3σ)"
            )
            z_scores.append(z_roe)
        else:
            z_scores.append(z_roe)

    # 2. Profit margin
    profit_margin = safe_float(info.get('profitMargins'))
    if profit_margin != 0.0:  # safe_float returns 0.0 for invalid/None values
        # LAYER 1: Data validation - Profit margins typically in [-100%, +100%]
        if abs(profit_margin) > 1.0:
            logger.debug(
                f"Quality factor: Extreme profit margin {profit_margin*100:+.1f}%, capping at ±100%"
            )
            profit_margin = 1.0 * np.sign(profit_margin)

        margin_mean, margin_std = quality_stats.get('profit_margin', (0.10, 0.08))
        z_margin = (profit_margin - margin_mean) / margin_std if margin_std > 0 else 0.0

        # LAYER 2: MSCI Barra USE4 z-score winsorization
        if abs(z_margin) > 10.0:
            logger.warning(
                f"Quality factor: Removing extreme margin z-score {z_margin:+.2f} (>10σ, likely data error)"
            )
        elif abs(z_margin) > 3.0:
            z_margin_original = z_margin
            z_margin = 3.0 * np.sign(z_margin)
            logger.debug(
                f"Quality factor: Winsorized margin z-score {z_margin_original:+.2f} → {z_margin:+.2f} (±3σ)"
            )
            z_scores.append(z_margin)
        else:
            z_scores.append(z_margin)

    # 3. Sharpe ratio (earnings quality)
    sharpe = metrics.get('sharpe_ratio', 0)
    # LAYER 1: Data validation - Sharpe ratio typically in [-5, +5]
    if abs(sharpe) > 10.0:
        logger.debug(
            f"Quality factor: Extreme Sharpe ratio {sharpe:+.2f}, capping at ±10"
        )
        sharpe = 10.0 * np.sign(sharpe)

    sharpe_mean, sharpe_std = quality_stats.get('sharpe_ratio', (0.5, 0.5))
    z_sharpe = (sharpe - sharpe_mean) / sharpe_std if sharpe_std > 0 else 0.0

    # LAYER 2: MSCI Barra USE4 z-score winsorization
    if abs(z_sharpe) > 10.0:
        logger.warning(
            f"Quality factor: Removing extreme Sharpe z-score {z_sharpe:+.2f} (>10σ, likely data error)"
        )
    elif abs(z_sharpe) > 3.0:
        z_sharpe_original = z_sharpe
        z_sharpe = 3.0 * np.sign(z_sharpe)
        logger.debug(
            f"Quality factor: Winsorized Sharpe z-score {z_sharpe_original:+.2f} → {z_sharpe:+.2f} (±3σ)"
        )
        z_scores.append(z_sharpe)
    else:
        z_scores.append(z_sharpe)

    # Calculate base quality z-score
    base_quality_z = float(np.mean(z_scores)) if len(z_scores) > 0 else 0.0

    # Apply distressed company penalty
    distress_penalty = _calculate_distress_penalty(info, metrics)
    if distress_penalty < 0:
        logger.debug(f"Quality factor: Distressed company penalty = {distress_penalty:.2f}")
        base_quality_z += distress_penalty

    # Apply inflation risk adjustment
    if country:
        forecasts = fetch_economic_forecasts(country)
        if forecasts and forecasts.get('inflation_forecast_6m') is not None:
            inflation_forecast = forecasts['inflation_forecast_6m']

            # Penalize quality when inflation > 3% (erodes real returns)
            if inflation_forecast > 3.0:
                # Progressive penalty: -0.1 z-score per 1% inflation above 3%
                inflation_penalty = -(inflation_forecast - 3.0) * 0.1
                adjusted_quality_z = base_quality_z + inflation_penalty

                logger.debug(
                    f"Quality factor: Inflation risk adjustment applied. "
                    f"Base z-score={base_quality_z:.2f}, "
                    f"Inflation forecast={inflation_forecast:.1f}%, "
                    f"Penalty={inflation_penalty:.2f}, "
                    f"Adjusted z-score={adjusted_quality_z:.2f}"
                )
                base_quality_z = adjusted_quality_z

    # LAYER 3: Factor-level winsorization (composite factor)
    if abs(base_quality_z) > 5.0:
        quality_z_original = base_quality_z
        base_quality_z = 5.0 * np.sign(base_quality_z)
        logger.warning(
            f"Quality factor: Capped composite quality factor {quality_z_original:+.2f} → {base_quality_z:+.2f} (±5σ)"
        )

    return base_quality_z


def _calculate_distress_penalty(info: Dict, metrics: Dict) -> float:
    """
    Calculate distress penalty for financially distressed companies.

    Detects:
    1. Negative equity (book value < 0)
    2. Negative earnings with high debt
    3. Declining revenue with high leverage
    4. Low Altman Z-score (bankruptcy risk)
    5. Negative operating cash flow

    Args:
        info: Stock info dict from yfinance
        metrics: Technical metrics dict

    Returns:
        Penalty (always <= 0, typically -1.0 to -3.0 for severely distressed)
    """
    penalty = 0.0

    # 1. Negative equity
    book_value = safe_float(info.get('bookValue'))
    if book_value < 0:
        penalty -= 2.0
        logger.debug("Distressed company detected: Negative equity")

    # 2. Negative ROE with high debt
    roe = safe_float(info.get('returnOnEquity'))
    debt_to_equity = safe_float(info.get('debtToEquity'))
    if roe < 0 and debt_to_equity > 2.0:
        penalty -= 1.5
        logger.debug(
            f"Distressed company detected: Negative ROE ({roe:.2%}) "
            f"with high debt (D/E={debt_to_equity:.2f})"
        )

    # 3. Declining revenue with high leverage
    revenue_growth = safe_float(info.get('revenueGrowth'))
    if revenue_growth < -0.10 and debt_to_equity > 1.5:
        penalty -= 1.0
        logger.debug(
            f"Distressed company detected: Declining revenue ({revenue_growth:.1%}) "
            f"with high leverage"
        )

    # 4. Simplified Altman Z-score
    total_assets = safe_float(info.get('totalAssets'))
    total_liabilities = safe_float(info.get('totalLiabilities', info.get('totalDebt')))
    ebitda = safe_float(info.get('ebitda'))
    market_cap = safe_float(info.get('marketCap'))
    total_revenue = safe_float(info.get('totalRevenue'))

    if total_assets > 0 and total_liabilities > 0 and market_cap > 0:
        z_score_components = []

        if ebitda > 0:
            z_score_components.append(3.3 * (ebitda / total_assets))

        z_score_components.append(0.6 * (market_cap / total_liabilities))

        if total_revenue > 0:
            z_score_components.append(1.0 * (total_revenue / total_assets))

        if z_score_components:
            z_score = sum(z_score_components)

            if z_score < 1.1:
                penalty -= 2.0
                logger.debug(f"Distressed company detected: Very low Z-score ({z_score:.2f})")
            elif z_score < 1.81:
                penalty -= 1.0
                logger.debug(f"Distressed company detected: Low Z-score ({z_score:.2f})")

    # 5. Negative operating cash flow with negative margins
    operating_cash_flow = safe_float(info.get('operatingCashflow'))
    profit_margin = safe_float(info.get('profitMargins'))
    if operating_cash_flow < 0 and profit_margin < 0:
        penalty -= 1.5
        logger.debug(
            "Distressed company detected: Negative operating cash flow with negative margins"
        )

    # Cap total penalty at -3.0
    return max(penalty, -3.0)


def calculate_growth_factor(
    info: Optional[Dict],
    country: Optional[str],
    cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
) -> float:
    """
    Calculate growth factor per Section 5.2.3.

    Blends trailing growth + forward-looking forecasts:
    - Revenue growth (trailing) - 60% weight
    - Earnings growth (trailing) - 60% weight
    - GDP forecast 6m (forward-looking) - 40% weight
    - Earnings forecast 12m (forward-looking) - 40% weight

    Forward-looking component helps catch turning points 3-6 months earlier.

    Args:
        info: Stock info dict from yfinance
        country: Country name (for forward-looking forecasts)
        cross_sectional_stats: Optional universe-level statistics

    Returns:
        Growth factor z-score
    """
    if not info:
        return 0.0

    z_scores = []
    growth_stats = cross_sectional_stats.get('growth', {}) if cross_sectional_stats else {}

    # Get forward-looking forecasts
    forecasts = None
    if country:
        forecasts = fetch_economic_forecasts(country)

    # 1. Revenue growth (trailing + forward blend)
    revenue_growth_raw = info.get('revenueGrowth')
    revenue_growth = safe_float(revenue_growth_raw) if revenue_growth_raw is not None else None
    if revenue_growth is not None and revenue_growth != 0.0:
        # LAYER 1: Data validation - Cap extreme growth values
        revenue_growth_capped = np.clip(revenue_growth, -0.50, 1.00)

        if revenue_growth != revenue_growth_capped:
            logger.debug(
                f"Growth factor: Capped extreme revenue growth "
                f"from {revenue_growth*100:.1f}% to {revenue_growth_capped*100:.1f}%"
            )

        # Calculate trailing z-score
        rev_mean, rev_std = growth_stats.get('revenue_growth', (0.05, 0.10))
        z_rev_trailing = (revenue_growth_capped - rev_mean) / rev_std if rev_std > 0 else 0.0

        # Blend with GDP forecast if available
        if forecasts and forecasts.get('gdp_forecast_6m') is not None:
            gdp_forecast = forecasts['gdp_forecast_6m']
            z_gdp_forward = (gdp_forecast - 2.0) / 3.0

            # 60% trailing, 40% forward
            z_rev_blended = 0.6 * z_rev_trailing + 0.4 * z_gdp_forward

            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_rev_blended) > 10.0:
                logger.warning(
                    f"Growth factor: Removing extreme revenue z-score {z_rev_blended:+.2f} (>10σ, likely data error)"
                )
            elif abs(z_rev_blended) > 3.0:
                z_rev_original = z_rev_blended
                z_rev_blended = 3.0 * np.sign(z_rev_blended)
                logger.debug(
                    f"Growth factor: Winsorized revenue z-score {z_rev_original:+.2f} → {z_rev_blended:+.2f} (±3σ)"
                )
                z_scores.append(z_rev_blended)
            else:
                z_scores.append(z_rev_blended)

            logger.debug(
                f"Growth factor: Blended revenue z-score = {z_rev_blended:.2f} "
                f"(60% trailing={z_rev_trailing:.2f}, 40% GDP forecast={z_gdp_forward:.2f})"
            )
        else:
            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_rev_trailing) > 10.0:
                logger.warning(
                    f"Growth factor: Removing extreme revenue z-score {z_rev_trailing:+.2f} (>10σ, likely data error)"
                )
            elif abs(z_rev_trailing) > 3.0:
                z_rev_original = z_rev_trailing
                z_rev_trailing = 3.0 * np.sign(z_rev_trailing)
                logger.debug(
                    f"Growth factor: Winsorized revenue z-score {z_rev_original:+.2f} → {z_rev_trailing:+.2f} (±3σ)"
                )
                z_scores.append(z_rev_trailing)
            else:
                z_scores.append(z_rev_trailing)

    # 2. Earnings growth (trailing + forward blend)
    earnings_growth_raw = info.get('earningsGrowth')
    earnings_growth = safe_float(earnings_growth_raw) if earnings_growth_raw is not None else None
    if earnings_growth is not None and earnings_growth != 0.0:
        # LAYER 1: Data validation - Cap extreme earnings growth
        earnings_growth_capped = np.clip(earnings_growth, -0.50, 1.00)

        if earnings_growth != earnings_growth_capped:
            logger.debug(
                f"Growth factor: Capped extreme earnings growth "
                f"from {earnings_growth*100:.1f}% to {earnings_growth_capped*100:.1f}%"
            )

        # Calculate trailing z-score
        earn_mean, earn_std = growth_stats.get('earnings_growth', (0.07, 0.15))
        z_earn_trailing = (
            (earnings_growth_capped - earn_mean) / earn_std if earn_std > 0 else 0.0
        )

        # Blend with earnings forecast if available
        if forecasts and forecasts.get('earnings_forecast_12m') is not None:
            earnings_forecast = forecasts['earnings_forecast_12m']
            z_earn_forward = (earnings_forecast - 7.0) / 15.0

            # 60% trailing, 40% forward
            z_earn_blended = 0.6 * z_earn_trailing + 0.4 * z_earn_forward

            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_earn_blended) > 10.0:
                logger.warning(
                    f"Growth factor: Removing extreme earnings z-score {z_earn_blended:+.2f} (>10σ, likely data error)"
                )
            elif abs(z_earn_blended) > 3.0:
                z_earn_original = z_earn_blended
                z_earn_blended = 3.0 * np.sign(z_earn_blended)
                logger.debug(
                    f"Growth factor: Winsorized earnings z-score {z_earn_original:+.2f} → {z_earn_blended:+.2f} (±3σ)"
                )
                z_scores.append(z_earn_blended)
            else:
                z_scores.append(z_earn_blended)

            logger.debug(
                f"Growth factor: Blended earnings z-score = {z_earn_blended:.2f} "
                f"(60% trailing={z_earn_trailing:.2f}, 40% forecast={z_earn_forward:.2f})"
            )
        else:
            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_earn_trailing) > 10.0:
                logger.warning(
                    f"Growth factor: Removing extreme earnings z-score {z_earn_trailing:+.2f} (>10σ, likely data error)"
                )
            elif abs(z_earn_trailing) > 3.0:
                z_earn_original = z_earn_trailing
                z_earn_trailing = 3.0 * np.sign(z_earn_trailing)
                logger.debug(
                    f"Growth factor: Winsorized earnings z-score {z_earn_original:+.2f} → {z_earn_trailing:+.2f} (±3σ)"
                )
                z_scores.append(z_earn_trailing)
            else:
                z_scores.append(z_earn_trailing)

    # Fallback: Use forecasts alone if no trailing data
    if len(z_scores) == 0 and forecasts:
        if forecasts.get('gdp_forecast_6m') is not None:
            gdp_forecast = forecasts['gdp_forecast_6m']
            z_gdp = (gdp_forecast - 2.0) / 3.0
            z_scores.append(z_gdp)
            logger.debug(f"Growth factor: Using GDP forecast only, z-score = {z_gdp:.2f}")

        if forecasts.get('earnings_forecast_12m') is not None:
            earnings_forecast = forecasts['earnings_forecast_12m']
            z_earn = (earnings_forecast - 7.0) / 15.0
            z_scores.append(z_earn)
            logger.debug(f"Growth factor: Using earnings forecast only, z-score = {z_earn:.2f}")

    # Average z-scores (equal weight)
    if len(z_scores) == 0:
        return 0.0

    growth_z = float(np.mean(z_scores))

    # LAYER 3: Factor-level winsorization (composite factor)
    if abs(growth_z) > 5.0:
        growth_z_original = growth_z
        growth_z = 5.0 * np.sign(growth_z)
        logger.warning(
            f"Growth factor: Capped composite growth factor {growth_z_original:+.2f} → {growth_z:+.2f} (±5σ)"
        )

    return growth_z


def calculate_ic_weights(historical_ics: Dict[str, list]) -> Dict[str, float]:
    """
    Calculate IC-based weights for factors per Section 5.2.6.

    Information Coefficient (IC) measures the correlation between factor scores
    and subsequent returns. Higher IC = more predictive factor = higher weight.

    Args:
        historical_ics: Dict mapping factor name to list of historical IC values

    Returns:
        Dynamic weights based on historical Information Coefficients.
        Falls back to equal weights if insufficient history (< 12 months).
    """
    if not historical_ics or len(historical_ics.get('value', [])) < 12:
        logger.debug("IC-weighting: Insufficient history, using equal weights")
        return {'value': 0.25, 'momentum': 0.25, 'quality': 0.25, 'growth': 0.25}

    # Calculate weights from historical ICs (use absolute value)
    ics = np.array([
        np.mean(historical_ics['value'][-12:]),
        np.mean(historical_ics['momentum'][-12:]),
        np.mean(historical_ics['quality'][-12:]),
        np.mean(historical_ics['growth'][-12:]),
    ])

    # Use absolute IC (direction matters less than strength)
    abs_ics = np.abs(ics)

    # Normalize to sum to 1
    weights = abs_ics / abs_ics.sum()

    logger.debug(
        f"IC-weighted factors: Value={weights[0]:.2f}, Momentum={weights[1]:.2f}, "
        f"Quality={weights[2]:.2f}, Growth={weights[3]:.2f}"
    )

    return {
        'value': weights[0],
        'momentum': weights[1],
        'quality': weights[2],
        'growth': weights[3],
    }
