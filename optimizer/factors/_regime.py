"""Macro regime classification and factor group tilts."""

from __future__ import annotations

import pandas as pd

from optimizer.factors._config import (
    FactorGroupType,
    MacroRegime,
    RegimeTiltConfig,
)


def classify_regime(macro_data: pd.DataFrame) -> MacroRegime:
    """Classify the current macro-economic regime.

    Uses a simple heuristic based on GDP growth and leading
    indicators.  The regime is determined by the latest
    observation's position relative to trend.

    Parameters
    ----------
    macro_data : pd.DataFrame
        Macro indicators with columns that may include
        ``gdp_growth``, ``leading_indicator``, ``yield_spread``,
        ``unemployment_rate``.  Index is date.

    Returns
    -------
    MacroRegime
        Current regime classification.
    """
    if len(macro_data) == 0:
        return MacroRegime.EXPANSION

    # Use GDP growth as primary signal
    if "gdp_growth" in macro_data.columns:
        gdp = macro_data["gdp_growth"].dropna()
        if len(gdp) >= 2:
            current = gdp.iloc[-1]
            previous = gdp.iloc[-2]
            trend = gdp.rolling(4, min_periods=1).mean().iloc[-1]

            # Multi-indicator override: rising unemployment + positive GDP → SLOWDOWN
            if "unemployment_rate" in macro_data.columns:
                unemp = macro_data["unemployment_rate"].dropna()
                if len(unemp) >= 2 and unemp.iloc[-1] > unemp.iloc[-2] and current > 0:
                    return MacroRegime.SLOWDOWN

            if current > trend and current > previous:
                return MacroRegime.EXPANSION
            if current > trend and current <= previous:
                return MacroRegime.SLOWDOWN
            if current <= trend and current <= previous:
                return MacroRegime.RECESSION
            return MacroRegime.RECOVERY

    # Fallback: use yield spread if available
    if "yield_spread" in macro_data.columns:
        spread = macro_data["yield_spread"].dropna()
        if len(spread) > 0:
            current_spread = spread.iloc[-1]
            if current_spread > 1.0:
                return MacroRegime.EXPANSION
            if current_spread > 0:
                return MacroRegime.SLOWDOWN
            if current_spread > -0.5:
                return MacroRegime.RECOVERY
            return MacroRegime.RECESSION

    return MacroRegime.EXPANSION


def get_regime_tilts(
    regime: MacroRegime,
    config: RegimeTiltConfig | None = None,
) -> dict[FactorGroupType, float]:
    """Get multiplicative tilts for a given regime.

    Parameters
    ----------
    regime : MacroRegime
        Current macro regime.
    config : RegimeTiltConfig or None
        Tilt configuration.

    Returns
    -------
    dict[FactorGroupType, float]
        Multiplicative tilt per group.  Groups not listed
        get a tilt of 1.0.
    """
    if config is None:
        config = RegimeTiltConfig()

    match regime:
        case MacroRegime.EXPANSION:
            raw_tilts = config.expansion_tilts
        case MacroRegime.SLOWDOWN:
            raw_tilts = config.slowdown_tilts
        case MacroRegime.RECESSION:
            raw_tilts = config.recession_tilts
        case MacroRegime.RECOVERY:
            raw_tilts = config.recovery_tilts

    tilts: dict[FactorGroupType, float] = {}
    for group_name, tilt_value in raw_tilts:
        try:
            group = FactorGroupType(group_name)
            tilts[group] = tilt_value
        except ValueError:
            continue

    return tilts


def apply_regime_tilts(
    group_weights: dict[FactorGroupType, float],
    regime: MacroRegime,
    config: RegimeTiltConfig | None = None,
) -> dict[FactorGroupType, float]:
    """Apply regime-conditional multiplicative tilts to group weights.

    Parameters
    ----------
    group_weights : dict[FactorGroupType, float]
        Base group weights.
    regime : MacroRegime
        Current macro regime.
    config : RegimeTiltConfig or None
        Tilt configuration.

    Returns
    -------
    dict[FactorGroupType, float]
        Tilted group weights (re-normalized to sum to original total).
    """
    if config is None:
        config = RegimeTiltConfig()

    if not config.enable:
        return dict(group_weights)

    tilts = get_regime_tilts(regime, config)

    tilted = {}
    for group, weight in group_weights.items():
        tilt = tilts.get(group, 1.0)
        tilted[group] = weight * tilt

    # Re-normalize to preserve total weight
    original_total = sum(group_weights.values())
    tilted_total = sum(tilted.values())
    if tilted_total > 0 and original_total > 0:
        scale = original_total / tilted_total
        tilted = {g: w * scale for g, w in tilted.items()}

    return tilted
