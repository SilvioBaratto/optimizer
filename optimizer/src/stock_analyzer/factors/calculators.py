from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from ..data.fetchers import fetch_economic_forecasts
from ..utils import safe_float


def calculate_value_factor(
    info: Optional[Dict],
    stock_data: pd.DataFrame,
    cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> float:
    """Calculate value factor"""
    if not info:
        return 0.0  # Neutral if no data

    z_scores = []
    value_stats = cross_sectional_stats.get("value", {}) if cross_sectional_stats else {}

    # 1. Book/Price ratio (inverse of P/B)
    price_to_book = safe_float(info.get("priceToBook"))
    if price_to_book > 0:
        # LAYER 1: Data validation - Remove impossible values
        if 0.01 <= price_to_book <= 100:
            book_price = 1 / price_to_book
            bp_mean, bp_std = value_stats.get("book_price", (0.25, 0.15))
            z_bp = (book_price - bp_mean) / bp_std if bp_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_bp) <= 10.0:
                if abs(z_bp) > 3.0:
                    z_bp = 3.0 * np.sign(z_bp)
                z_scores.append(z_bp)

    # 2. Earnings/Price ratio (inverse of P/E)
    trailing_pe = safe_float(info.get("trailingPE"))
    if trailing_pe > 0:
        # LAYER 1: Data validation
        if 0.1 <= trailing_pe <= 1000:
            earnings_price = 1 / trailing_pe
            ep_mean, ep_std = value_stats.get("earnings_price", (0.05, 0.03))
            z_ep = (earnings_price - ep_mean) / ep_std if ep_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_ep) <= 10.0:
                if abs(z_ep) > 3.0:
                    z_ep = 3.0 * np.sign(z_ep)
                z_scores.append(z_ep)

    # 3. Sales/Price ratio (inverse of P/S)
    price_to_sales = safe_float(info.get("priceToSalesTrailing12Months"))
    if price_to_sales > 0:
        # LAYER 1: Data validation
        if 0.01 <= price_to_sales <= 100:
            sales_price = 1 / price_to_sales
            sp_mean, sp_std = value_stats.get("sales_price", (0.50, 0.30))
            z_sp = (sales_price - sp_mean) / sp_std if sp_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_sp) <= 10.0:
                if abs(z_sp) > 3.0:
                    z_sp = 3.0 * np.sign(z_sp)
                z_scores.append(z_sp)

    # 4. Dividend/Price ratio (dividend yield)
    dividend_yield = safe_float(info.get("dividendYield"))
    if dividend_yield > 0:
        # LAYER 1: Data validation (dividend yields typically 0-10%)
        if dividend_yield <= 0.10:
            dp_mean, dp_std = value_stats.get("dividend_price", (0.02, 0.015))
            z_dp = (dividend_yield - dp_mean) / dp_std if dp_std > 0 else 0.0

            # LAYER 2: MSCI Barra USE4 descriptor z-score winsorization
            if abs(z_dp) <= 10.0:
                if abs(z_dp) > 3.0:
                    z_dp = 3.0 * np.sign(z_dp)
                z_scores.append(z_dp)

    # Average z-scores of available metrics (equal weight)
    if len(z_scores) == 0:
        return 0.0

    value_z = float(np.mean(z_scores))

    # LAYER 3: Factor-level winsorization (composite factor)
    if abs(value_z) > 5.0:
        value_z = 5.0 * np.sign(value_z)

    return value_z


def calculate_momentum_factor(
    metrics: Dict, cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
) -> float:
    """Calculate momentum factor per Section"""
    # Get 12-1 month momentum
    momentum_12m = metrics.get("momentum_12m_minus_1m", 0)

    # LAYER 1: Data validation - Cap extreme returns (likely data errors)
    if abs(momentum_12m) > 5.0:  # >500% return is suspicious
        momentum_12m = 5.0 * np.sign(momentum_12m)

    # Use cross-sectional stats if available, otherwise static norms
    momentum_stats = cross_sectional_stats.get("momentum", {}) if cross_sectional_stats else {}
    mom_mean, mom_std = momentum_stats.get("momentum_12m", (0.08, 0.20))

    # Standardize
    z_momentum = (momentum_12m - mom_mean) / mom_std if mom_std > 0 else 0.0

    # LAYER 2: MSCI Barra USE4 z-score winsorization
    if abs(z_momentum) > 10.0:
        return 0.0  # Remove completely
    elif abs(z_momentum) > 3.0:
        z_momentum = 3.0 * np.sign(z_momentum)

    # LAYER 3: Factor-level cap (redundant for single descriptor, but consistent)
    if abs(z_momentum) > 5.0:
        z_momentum = 5.0 * np.sign(z_momentum)

    return z_momentum


def calculate_quality_factor(
    info: Optional[Dict],
    metrics: Dict,
    country: Optional[str],
    cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> float:
    """Calculate quality factor"""
    if not info:
        # Fallback to Sharpe ratio only
        sharpe = metrics.get("sharpe_ratio", 0)
        return (sharpe - 0.5) / 0.5

    z_scores = []
    quality_stats = cross_sectional_stats.get("quality", {}) if cross_sectional_stats else {}

    # 1. ROE (Return on Equity)
    roe = safe_float(info.get("returnOnEquity"))
    if roe != 0.0:  # safe_float returns 0.0 for invalid/None values
        # LAYER 1: Data validation - ROE typically in [-200%, +100%]
        if abs(roe) > 2.0:
            roe = 2.0 * np.sign(roe)

        roe_mean, roe_std = quality_stats.get("roe", (0.15, 0.10))
        z_roe = (roe - roe_mean) / roe_std if roe_std > 0 else 0.0

        # LAYER 2: MSCI Barra USE4 z-score winsorization
        if abs(z_roe) <= 10.0:
            if abs(z_roe) > 3.0:
                z_roe = 3.0 * np.sign(z_roe)
            z_scores.append(z_roe)

    # 2. Profit margin
    profit_margin = safe_float(info.get("profitMargins"))
    if profit_margin != 0.0:  # safe_float returns 0.0 for invalid/None values
        # LAYER 1: Data validation - Profit margins typically in [-100%, +100%]
        if abs(profit_margin) > 1.0:
            profit_margin = 1.0 * np.sign(profit_margin)

        margin_mean, margin_std = quality_stats.get("profit_margin", (0.10, 0.08))
        z_margin = (profit_margin - margin_mean) / margin_std if margin_std > 0 else 0.0

        # LAYER 2: MSCI Barra USE4 z-score winsorization
        if abs(z_margin) <= 10.0:
            if abs(z_margin) > 3.0:
                z_margin = 3.0 * np.sign(z_margin)
            z_scores.append(z_margin)

    # 3. Sharpe ratio (earnings quality)
    sharpe = metrics.get("sharpe_ratio", 0)
    # LAYER 1: Data validation - Sharpe ratio typically in [-5, +5]
    if abs(sharpe) > 10.0:
        sharpe = 10.0 * np.sign(sharpe)

    sharpe_mean, sharpe_std = quality_stats.get("sharpe_ratio", (0.5, 0.5))
    z_sharpe = (sharpe - sharpe_mean) / sharpe_std if sharpe_std > 0 else 0.0

    # LAYER 2: MSCI Barra USE4 z-score winsorization
    if abs(z_sharpe) <= 10.0:
        if abs(z_sharpe) > 3.0:
            z_sharpe = 3.0 * np.sign(z_sharpe)
        z_scores.append(z_sharpe)

    # Calculate base quality z-score
    base_quality_z = float(np.mean(z_scores)) if len(z_scores) > 0 else 0.0

    # Apply distressed company penalty
    distress_penalty = _calculate_distress_penalty(info, metrics)
    if distress_penalty < 0:
        base_quality_z += distress_penalty

    # Apply inflation risk adjustment
    if country:
        forecasts = fetch_economic_forecasts(country)
        if forecasts and forecasts.get("inflation_forecast_6m") is not None:
            inflation_forecast = forecasts["inflation_forecast_6m"]

            # Penalize quality when inflation > 3% (erodes real returns)
            if inflation_forecast > 3.0:
                # Progressive penalty: -0.1 z-score per 1% inflation above 3%
                inflation_penalty = -(inflation_forecast - 3.0) * 0.1
                base_quality_z += inflation_penalty

    # LAYER 3: Factor-level winsorization (composite factor)
    if abs(base_quality_z) > 5.0:
        base_quality_z = 5.0 * np.sign(base_quality_z)

    return base_quality_z


def _calculate_distress_penalty(info: Dict, metrics: Dict) -> float:
    """
    Calculate distress penalty for financially distressed companies.
    """
    penalty = 0.0

    # 1. Negative equity
    book_value = safe_float(info.get("bookValue"))
    if book_value < 0:
        penalty -= 2.0

    # 2. Negative ROE with high debt
    roe = safe_float(info.get("returnOnEquity"))
    debt_to_equity = safe_float(info.get("debtToEquity"))
    if roe < 0 and debt_to_equity > 2.0:
        penalty -= 1.5

    # 3. Declining revenue with high leverage
    revenue_growth = safe_float(info.get("revenueGrowth"))
    if revenue_growth < -0.10 and debt_to_equity > 1.5:
        penalty -= 1.0

    # 4. Simplified Altman Z-score
    total_assets = safe_float(info.get("totalAssets"))
    total_liabilities = safe_float(info.get("totalLiabilities", info.get("totalDebt")))
    ebitda = safe_float(info.get("ebitda"))
    market_cap = safe_float(info.get("marketCap"))
    total_revenue = safe_float(info.get("totalRevenue"))

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
            elif z_score < 1.81:
                penalty -= 1.0

    # 5. Negative operating cash flow with negative margins
    operating_cash_flow = safe_float(info.get("operatingCashflow"))
    profit_margin = safe_float(info.get("profitMargins"))
    if operating_cash_flow < 0 and profit_margin < 0:
        penalty -= 1.5

    # Cap total penalty at -3.0
    return max(penalty, -3.0)


def calculate_growth_factor(
    info: Optional[Dict],
    country: Optional[str],
    cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> float:
    """Calculate growth factor"""
    if not info:
        return 0.0

    z_scores = []
    growth_stats = cross_sectional_stats.get("growth", {}) if cross_sectional_stats else {}

    # Get forward-looking forecasts
    forecasts = None
    if country:
        forecasts = fetch_economic_forecasts(country)

    # 1. Revenue growth (trailing + forward blend)
    revenue_growth_raw = info.get("revenueGrowth")
    revenue_growth = safe_float(revenue_growth_raw) if revenue_growth_raw is not None else None
    if revenue_growth is not None and revenue_growth != 0.0:
        # LAYER 1: Data validation - Cap extreme growth values
        revenue_growth_capped = np.clip(revenue_growth, -0.50, 1.00)

        # Calculate trailing z-score
        rev_mean, rev_std = growth_stats.get("revenue_growth", (0.05, 0.10))
        z_rev_trailing = (revenue_growth_capped - rev_mean) / rev_std if rev_std > 0 else 0.0

        # Blend with GDP forecast if available
        if forecasts and forecasts.get("gdp_forecast_6m") is not None:
            gdp_forecast = forecasts["gdp_forecast_6m"]
            z_gdp_forward = (gdp_forecast - 2.0) / 3.0

            # 60% trailing, 40% forward
            z_rev_blended = 0.6 * z_rev_trailing + 0.4 * z_gdp_forward

            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_rev_blended) <= 10.0:
                if abs(z_rev_blended) > 3.0:
                    z_rev_blended = 3.0 * np.sign(z_rev_blended)
                z_scores.append(z_rev_blended)
        else:
            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_rev_trailing) <= 10.0:
                if abs(z_rev_trailing) > 3.0:
                    z_rev_trailing = 3.0 * np.sign(z_rev_trailing)
                z_scores.append(z_rev_trailing)

    # 2. Earnings growth (trailing + forward blend)
    earnings_growth_raw = info.get("earningsGrowth")
    earnings_growth = safe_float(earnings_growth_raw) if earnings_growth_raw is not None else None
    if earnings_growth is not None and earnings_growth != 0.0:
        # LAYER 1: Data validation - Cap extreme earnings growth
        earnings_growth_capped = np.clip(earnings_growth, -0.50, 1.00)

        # Calculate trailing z-score
        earn_mean, earn_std = growth_stats.get("earnings_growth", (0.07, 0.15))
        z_earn_trailing = (earnings_growth_capped - earn_mean) / earn_std if earn_std > 0 else 0.0

        # Blend with earnings forecast if available
        if forecasts and forecasts.get("earnings_forecast_12m") is not None:
            earnings_forecast = forecasts["earnings_forecast_12m"]
            z_earn_forward = (earnings_forecast - 7.0) / 15.0

            # 60% trailing, 40% forward
            z_earn_blended = 0.6 * z_earn_trailing + 0.4 * z_earn_forward

            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_earn_blended) <= 10.0:
                if abs(z_earn_blended) > 3.0:
                    z_earn_blended = 3.0 * np.sign(z_earn_blended)
                z_scores.append(z_earn_blended)
        else:
            # LAYER 2: MSCI Barra USE4 z-score winsorization
            if abs(z_earn_trailing) <= 10.0:
                if abs(z_earn_trailing) > 3.0:
                    z_earn_trailing = 3.0 * np.sign(z_earn_trailing)
                z_scores.append(z_earn_trailing)

    # Fallback: Use forecasts alone if no trailing data
    if len(z_scores) == 0 and forecasts:
        if forecasts.get("gdp_forecast_6m") is not None:
            gdp_forecast = forecasts["gdp_forecast_6m"]
            z_gdp = (gdp_forecast - 2.0) / 3.0
            z_scores.append(z_gdp)

        if forecasts.get("earnings_forecast_12m") is not None:
            earnings_forecast = forecasts["earnings_forecast_12m"]
            z_earn = (earnings_forecast - 7.0) / 15.0
            z_scores.append(z_earn)

    # Average z-scores (equal weight)
    if len(z_scores) == 0:
        return 0.0

    growth_z = float(np.mean(z_scores))

    # LAYER 3: Factor-level winsorization (composite factor)
    if abs(growth_z) > 5.0:
        growth_z = 5.0 * np.sign(growth_z)

    return growth_z


def calculate_ic_weights(historical_ics: Dict[str, list]) -> Dict[str, float]:
    """Calculate IC-based weights for factors"""
    if not historical_ics or len(historical_ics.get("value", [])) < 12:
        return {"value": 0.25, "momentum": 0.25, "quality": 0.25, "growth": 0.25}

    # Calculate weights from historical ICs (use absolute value)
    ics = np.array(
        [
            np.mean(historical_ics["value"][-12:]),
            np.mean(historical_ics["momentum"][-12:]),
            np.mean(historical_ics["quality"][-12:]),
            np.mean(historical_ics["growth"][-12:]),
        ]
    )

    # Use absolute IC (direction matters less than strength)
    abs_ics = np.abs(ics)

    # Normalize to sum to 1
    weights = abs_ics / abs_ics.sum()

    return {
        "value": weights[0],
        "momentum": weights[1],
        "quality": weights[2],
        "growth": weights[3],
    }
