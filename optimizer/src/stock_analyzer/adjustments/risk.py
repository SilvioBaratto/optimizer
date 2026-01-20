from typing import Dict, Optional
import pandas as pd

from baml_client.types import RiskFactors, ConfidenceLevel


def calculate_risk_factors(
    metrics: Dict,
    info: Dict,
    stock_data: Optional[pd.DataFrame] = None,
) -> RiskFactors:
    """
    Calculate institutional-grade risk factors.
    """
    # 1. Volatility Risk
    volatility = metrics.get("volatility", 0.20)
    if volatility < 0.10:
        volatility_level = "MINIMAL"
    elif volatility <= 0.15:
        volatility_level = "LOW"
    elif volatility <= 0.25:
        volatility_level = "MEDIUM"
    elif volatility <= 0.35:
        volatility_level = "HIGH"
    else:
        volatility_level = "EXTREME"

    # 2. Beta Risk
    beta = metrics.get("beta", 1.0)
    r_squared = metrics.get("r_squared", 0.5)

    systematic_risk_pct = r_squared * 100
    specific_risk_pct = (1 - r_squared) * 100

    if beta < 0.8:
        beta_risk = "LOW"
    elif beta <= 1.2:
        beta_risk = "MEDIUM"
    else:
        beta_risk = "HIGH"

    # 3. Liquidity Risk
    liquidity_risk = _calculate_liquidity_risk(stock_data, info)

    # 4. Drawdown Risk
    max_dd = abs(metrics.get("max_drawdown", -0.15))
    if max_dd <= 0.10:
        drawdown_risk = "LOW"
    elif max_dd <= 0.15:
        drawdown_risk = "MEDIUM"
    elif max_dd <= 0.25:
        drawdown_risk = "HIGH"
    else:
        drawdown_risk = "EXTREME"

    # 5. Sharpe Ratio Quality
    sharpe = metrics.get("sharpe_ratio", 0)
    if sharpe >= 1.0:
        risk_adjusted_quality = "EXCELLENT"
    elif sharpe >= 0.5:
        risk_adjusted_quality = "GOOD"
    elif sharpe >= 0:
        risk_adjusted_quality = "WEAK"
    else:
        risk_adjusted_quality = "POOR"

    # 6. Debt/Leverage Risk
    debt_to_equity = info.get("debtToEquity") if info else None
    current_ratio = info.get("currentRatio", 1.0) if info else 1.0

    if debt_to_equity is not None:
        if debt_to_equity > 300 or current_ratio < 0.5:
            debt_risk = "EXTREME"
        elif debt_to_equity > 200 or current_ratio < 1.0:
            debt_risk = "HIGH"
        elif debt_to_equity > 100 or current_ratio < 1.5:
            debt_risk = "MEDIUM"
        else:
            debt_risk = "LOW"
    else:
        debt_risk = "UNKNOWN"

    # 7. Data Quality Gaps
    data_gaps = _identify_data_gaps(metrics, info, stock_data)

    # 8. Primary Risk Summary
    debt_display = f"{debt_to_equity:.0f}" if debt_to_equity is not None else "N/A"
    primary_risks = [
        f"Total volatility: {volatility*100:.1f}% ({volatility_level})",
        f"Systematic risk: {systematic_risk_pct:.0f}% (β={beta:.2f}, {beta_risk})",
        f"Specific risk: {specific_risk_pct:.0f}% (idiosyncratic)",
        f"Max drawdown: {max_dd*100:.1f}% ({drawdown_risk})",
        f"Liquidity: {liquidity_risk} (ADV analysis)",
        f"Leverage: D/E={debt_display}, CR={current_ratio:.1f} ({debt_risk})",
    ]

    if sharpe < 0.3:
        primary_risks.append(f"Risk-adjusted return: Sharpe={sharpe:.2f} ({risk_adjusted_quality})")

    return RiskFactors(
        volatility_level=volatility_level,
        beta_risk=beta_risk,
        debt_risk=debt_risk,
        liquidity_risk=liquidity_risk,
        data_gaps=data_gaps,
        primary_risks=primary_risks,
    )


def _calculate_liquidity_risk(stock_data: Optional[pd.DataFrame], info: Optional[Dict]) -> str:
    """Calculate liquidity risk based on ADV and market cap."""
    if stock_data is None or len(stock_data) < 20 or not info:
        return "MEDIUM"

    try:
        recent_volume = stock_data["Volume"].tail(20).mean()
        current_price = stock_data["Close"].iloc[-1]
        adv_dollars = recent_volume * current_price
        market_cap = info.get("marketCap", 0)

        # Liquidity classification by market cap tier
        if market_cap > 10_000_000_000:  # Large-cap
            if adv_dollars > 100_000_000:
                return "LOW"
            elif adv_dollars > 50_000_000:
                return "MEDIUM"
            else:
                return "HIGH"
        elif market_cap > 2_000_000_000:  # Mid-cap
            if adv_dollars > 50_000_000:
                return "LOW"
            elif adv_dollars > 10_000_000:
                return "MEDIUM"
            else:
                return "HIGH"
        else:  # Small-cap
            if adv_dollars > 10_000_000:
                return "LOW"
            elif adv_dollars > 1_000_000:
                return "MEDIUM"
            else:
                return "HIGH"
    except Exception:
        return "MEDIUM"


def _identify_data_gaps(
    metrics: Dict, info: Optional[Dict], stock_data: Optional[pd.DataFrame]
) -> list[str]:
    """Identify missing critical data."""
    gaps = []

    if "beta" not in metrics:
        gaps.append("No benchmark comparison available")
    if "r_squared" not in metrics:
        gaps.append("Cannot decompose systematic vs specific risk")
    if "ma_200" not in metrics:
        gaps.append("Insufficient data for 200-day moving average")
    if not info or info.get("debtToEquity") is None:
        gaps.append("Debt/leverage data unavailable")
    if stock_data is None or len(stock_data) < 20:
        gaps.append("Insufficient data for liquidity analysis")

    return gaps


def calculate_confidence(
    metrics: Dict, composite_score: float, stock_data: pd.DataFrame
) -> ConfidenceLevel:
    """
    Calculate confidence level using institutional framework.
    """
    # 1. Statistical Significance (40%)
    statistical_confidence = 0.0

    alpha_t = abs(metrics.get("alpha_t_stat", 0))
    if alpha_t >= 2.0:
        statistical_confidence += 0.35
    elif alpha_t >= 1.65:
        statistical_confidence += 0.25
    elif alpha_t >= 1.0:
        statistical_confidence += 0.15

    r_squared = metrics.get("r_squared", 0)
    if r_squared >= 0.5:
        statistical_confidence += 0.25
    elif r_squared >= 0.3:
        statistical_confidence += 0.15
    elif r_squared >= 0.1:
        statistical_confidence += 0.05

    info_ratio = abs(metrics.get("information_ratio", 0))
    if info_ratio >= 0.5:
        statistical_confidence += 0.20
    elif info_ratio >= 0.25:
        statistical_confidence += 0.10

    sharpe = metrics.get("sharpe_ratio", 0)
    if abs(sharpe) >= 1.0:
        statistical_confidence += 0.20
    elif abs(sharpe) >= 0.5:
        statistical_confidence += 0.10

    # 2. Data Sufficiency (30%)
    days_of_data = len(stock_data)
    if days_of_data >= 504:
        data_confidence = 1.0
    elif days_of_data >= 252:
        data_confidence = 0.8
    elif days_of_data >= 84:
        data_confidence = 0.6
    elif days_of_data >= 63:
        data_confidence = 0.4
    else:
        data_confidence = 0.2

    # 3. Metrics Completeness (20%)
    critical_metrics = ["beta", "alpha", "sharpe_ratio", "volatility", "max_drawdown"]
    metrics_present = sum(1 for m in critical_metrics if m in metrics and metrics[m] is not None)
    metrics_completeness = metrics_present / len(critical_metrics)

    # 4. Score Extremity (10%)
    z_score = abs((composite_score - 50) / 15)
    if z_score >= 2.0:
        score_confidence = 1.0
    elif z_score >= 1.5:
        score_confidence = 0.8
    elif z_score >= 1.0:
        score_confidence = 0.6
    elif z_score >= 0.5:
        score_confidence = 0.4
    else:
        score_confidence = 0.2

    # 5. Combine Components
    total_confidence = (
        statistical_confidence * 0.40
        + data_confidence * 0.30
        + metrics_completeness * 0.20
        + score_confidence * 0.10
    )

    # 6. Penalties for Concerning Patterns
    volatility = metrics.get("volatility", 0)
    if volatility > 0.35 and sharpe < 0.3:
        total_confidence *= 0.8

    if "alpha_lower_95" in metrics and "alpha_upper_95" in metrics:
        alpha_range = metrics["alpha_upper_95"] - metrics["alpha_lower_95"]
        if alpha_range > 0.20:
            total_confidence *= 0.9

    # 7. Map to Confidence Levels
    if total_confidence >= 0.70:
        return ConfidenceLevel.HIGH
    elif total_confidence >= 0.40:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
