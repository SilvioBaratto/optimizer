"""
Scoring Utilities
=================

Provides utility functions for signal scoring:
- Upside potential calculation
- Downside risk calculation
- Data quality scoring
- Analysis notes generation
"""

from typing import Dict, Optional
import logging
import numpy as np
import pandas as pd

from baml_client.types import SignalType

logger = logging.getLogger(__name__)


def calculate_upside_potential(
    composite_score: float, metrics: Dict, macro_data: Optional[Dict] = None
) -> float:
    """
    Calculate upside potential based on institutional factor framework.

    Expected returns derived from:
    - Factor premiums (2-6% annually)
    - Market return expectation (~10% historical equity return)
    - Risk-adjusted by Sharpe ratio
    - James-Stein shrinkage toward mean
    - Business cycle positioning

    Args:
        composite_score: Composite score (0-100 scale)
        metrics: Technical metrics dict
        macro_data: Optional macro regime data

    Returns:
        Upside potential as percentage (0-20%)
    """
    # Convert score to z-score
    z_score = (composite_score - 50) / 15

    # Map z-score to factor premium (-2% to +6%)
    if z_score > 1.5:
        base_premium = 6.0
    elif z_score > 0.5:
        base_premium = 4.0 + (z_score - 0.5) * 2.0
    elif z_score > -0.5:
        base_premium = 2.0 + (z_score + 0.5) * 2.0
    elif z_score > -1.5:
        base_premium = 0.0 + (z_score + 1.5) * 2.0
    else:
        base_premium = -2.0

    # Add market return
    market_return = 10.0
    raw_expected_return = market_return + base_premium

    # Sharpe ratio adjustment
    sharpe = metrics.get('sharpe_ratio', 0.5)
    sharpe_multiplier = np.clip(np.exp(0.15 * sharpe), 0.6, 1.5)

    # James-Stein shrinkage
    shrinkage_factor = 0.7
    shrunken_return = shrinkage_factor * raw_expected_return + (1 - shrinkage_factor) * market_return

    # Risk-adjust
    risk_adjusted_return = shrunken_return * sharpe_multiplier

    # Regime adjustment
    regime_multiplier = 1.0
    if macro_data:
        regime_raw = macro_data.get('regime', 'UNCERTAIN')
        regime = regime_raw.value if hasattr(regime_raw, 'value') else regime_raw

        if regime == 'EARLY_CYCLE':
            regime_multiplier = 1.15
        elif regime == 'RECESSION':
            regime_multiplier = 0.85
        elif regime == 'LATE_CYCLE':
            regime_multiplier = 0.95

    final_upside = risk_adjusted_return * regime_multiplier

    # Clip to institutional bounds (0-20%)
    return float(np.clip(final_upside, 0.0, 20.0))


def calculate_downside_risk(
    composite_score: float, metrics: Dict, macro_data: Optional[Dict] = None
) -> float:
    """
    Calculate downside risk based on institutional risk framework.

    Approach:
    1. Calculate downside deviation (Sortino-based)
    2. Blend historical (max DD) and forward-looking (downside dev) risk
    3. Apply quality adjustment via z-score
    4. Apply regime multiplier
    5. Clip to institutional bounds (5-35%)

    Args:
        composite_score: Composite score (0-100 scale)
        metrics: Technical metrics dict
        macro_data: Optional macro regime data

    Returns:
        Downside risk as percentage (5-35%)
    """
    # Base risk from historical metrics
    max_dd = abs(metrics.get('max_drawdown', -0.15))
    volatility = metrics.get('volatility', 0.20)

    # Calculate downside deviation
    sortino = metrics.get('sortino_ratio', 0.5)
    if sortino > 0 and metrics.get('annualized_return') is not None:
        risk_free_rate = metrics.get('risk_free_rate', 0.045)
        excess_return = metrics.get('annualized_return', 0.07) - risk_free_rate
        downside_deviation = abs(excess_return / sortino) if sortino != 0 else volatility
    else:
        downside_deviation = volatility * 0.7

    # Combine max DD and downside deviation
    base_risk = (0.4 * max_dd + 0.6 * downside_deviation) * 100

    # Quality adjustment
    z_score = (composite_score - 50) / 15
    quality_multiplier = np.exp(-0.2 * z_score)

    # Regime adjustment
    regime_multiplier = 1.0
    if macro_data:
        regime_raw = macro_data.get('regime', 'UNCERTAIN')
        regime = regime_raw.value if hasattr(regime_raw, 'value') else regime_raw
        recession_risk_12m = macro_data.get('recession_risk_12m', 0)

        if regime == 'RECESSION':
            regime_multiplier = 1.20
        elif regime == 'LATE_CYCLE':
            regime_multiplier = 1.10
        elif recession_risk_12m > 0.5:
            regime_multiplier = 1.15

    final_risk = base_risk * quality_multiplier * regime_multiplier

    # Clip to institutional bounds (5-35%)
    return float(np.clip(final_risk, 5.0, 35.0))


def calculate_data_quality(stock_data: pd.DataFrame, info: Dict) -> float:
    """
    Calculate data quality score (0-1).

    Components:
    - Price data completeness (0.3)
    - Volume data availability (0.2)
    - Fundamental data availability (0.5)

    Args:
        stock_data: Historical price data
        info: Stock info dict

    Returns:
        Data quality score (0.0 to 1.0)
    """
    quality = 0.0

    # Price data completeness
    if len(stock_data) >= 200:
        quality += 0.3
    elif len(stock_data) >= 100:
        quality += 0.2
    elif len(stock_data) >= 50:
        quality += 0.1

    # Has volume data
    if 'Volume' in stock_data.columns and stock_data['Volume'].sum() > 0:
        quality += 0.2

    # Has fundamental info fields
    if info:
        if info.get('trailingPE') or info.get('forwardPE'):
            quality += 0.2
        if info.get('priceToBook'):
            quality += 0.15
        if info.get('sector'):
            quality += 0.15

    return min(quality, 1.0)


def generate_analysis_notes(
    ticker: str,
    signal_type: SignalType,
    composite_z: float,
    value_z: float,
    momentum_z: float,
    quality_z: float,
    growth_z: float,
    metrics: Dict,
    macro_data: Optional[Dict],
) -> str:
    """
    Generate concise analysis notes based on four-factor framework.

    Args:
        ticker: Stock ticker symbol
        signal_type: Classified signal type
        composite_z: Composite z-score
        value_z: Value factor z-score
        momentum_z: Momentum factor z-score
        quality_z: Quality factor z-score
        growth_z: Growth factor z-score
        metrics: Technical metrics dict
        macro_data: Optional macro regime data

    Returns:
        Concise analysis notes (2-3 sentences)
    """
    # Sentence 1: Signal and composite z-score
    signal_desc = {
        SignalType.LARGE_GAIN: "strong bullish",
        SignalType.SMALL_GAIN: "moderately bullish",
        SignalType.NEUTRAL: "neutral",
        SignalType.SMALL_DECLINE: "moderately bearish",
        SignalType.LARGE_DECLINE: "strong bearish",
    }[signal_type]

    notes = f"{ticker} shows a {signal_desc} signal with composite z-score of {composite_z:+.2f}. "

    # Sentence 2: Leading factor
    factors = [
        ('Value', value_z),
        ('Momentum', momentum_z),
        ('Quality', quality_z),
        ('Growth', growth_z),
    ]
    factors.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factor = factors[0]

    notes += f"Leading factor is {top_factor[0]} (z={top_factor[1]:+.2f}), "

    # Add specific metric highlights
    sharpe = metrics.get('sharpe_ratio', 0)
    mom_12m = metrics.get('momentum_12m_minus_1m', 0) * 100

    if abs(sharpe) > 1.0:
        notes += f"with Sharpe ratio of {sharpe:.2f} "
    if abs(mom_12m) > 10:
        notes += f"and {mom_12m:+.1f}% 12-1m momentum "

    # Sentence 3: Macro context
    if macro_data:
        regime_raw = macro_data.get('regime', 'UNCERTAIN')
        regime = regime_raw.value if hasattr(regime_raw, 'value') else regime_raw
        notes += f"in a {regime} regime. "
    else:
        notes += "with limited macro context. "

    return notes.strip()
