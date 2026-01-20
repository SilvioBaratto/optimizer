from typing import Optional, Dict

from ..data.fetchers import fetch_pmi_data, fetch_unemployment_rate


def apply_macro_adjustments(
    composite_z: float,
    macro_data: Optional[Dict],
    info: Optional[Dict],
    country: Optional[str],
) -> float:
    """
    Apply conservative macro adjustments to composite z-score.
    """
    if not macro_data or not info or not country:
        return composite_z  # No adjustments if missing data

    sector = info.get("sector")
    if not sector:
        return composite_z

    adjusted_z = composite_z

    # Get regime
    regime_raw = macro_data.get("regime", "UNCERTAIN")
    regime = regime_raw.value if hasattr(regime_raw, "value") else regime_raw

    # GICS-compliant sector classification
    cyclical_sectors = [
        "Consumer Discretionary",
        "Financials",
        "Industrials",
        "Materials",
        "Technology",
    ]
    defensive_sectors = ["Consumer Staples", "Healthcare", "Utilities"]
    mixed_sectors = ["Energy", "Communication Services", "Real Estate"]

    is_cyclical = sector in cyclical_sectors
    is_defensive = sector in defensive_sectors
    is_mixed = sector in mixed_sectors

    # PMI-based sector adjustments
    adjusted_z = _apply_pmi_adjustment(
        adjusted_z, sector, country, regime, macro_data, is_cyclical, is_defensive
    )

    # Unemployment overlay (consumer discretionary only)
    adjusted_z = _apply_unemployment_adjustment(adjusted_z, sector, country)

    # Regime overlay (broader business cycle effects)
    adjusted_z = _apply_regime_overlay(adjusted_z, regime, is_cyclical, is_defensive, composite_z)

    return adjusted_z


def _apply_pmi_adjustment(
    adjusted_z: float,
    sector: str,
    country: str,
    regime: str,
    macro_data: Dict,
    is_cyclical: bool,
    is_defensive: bool,
) -> float:
    """Apply PMI-based sector adjustments."""
    # Get sector-appropriate PMI
    if sector in ["Industrials", "Materials"]:
        ism_pmi = fetch_pmi_data(country, pmi_type="manufacturing")
    elif sector in [
        "Technology",
        "Financials",
        "Consumer Discretionary",
        "Consumer Staples",
        "Healthcare",
        "Utilities",
    ]:
        ism_pmi = fetch_pmi_data(country, pmi_type="services")
    else:
        ism_pmi = fetch_pmi_data(country, pmi_type="composite")

    # Fallback chain
    if ism_pmi is None:
        ism_pmi = fetch_pmi_data(country, pmi_type="composite")
    if ism_pmi is None:
        ism_pmi = fetch_pmi_data(country, pmi_type="manufacturing")

    if ism_pmi is None:
        return adjusted_z

    # Conservative PMI multipliers (±5% max)
    if ism_pmi > 52:  # Strong expansion
        pmi_multiplier = 1.05 if is_cyclical else (0.97 if is_defensive else 1.0)
    elif ism_pmi >= 50:  # Mild expansion
        if is_cyclical:
            pmi_multiplier = 1.0 + (ism_pmi - 50) * 0.025
        elif is_defensive:
            pmi_multiplier = 1.0 - (ism_pmi - 50) * 0.015
        else:
            pmi_multiplier = 1.0
    elif ism_pmi >= 48:  # Mild contraction
        if is_cyclical:
            pmi_multiplier = 1.0 - (50 - ism_pmi) * 0.015
        elif is_defensive:
            pmi_multiplier = 1.0 + (50 - ism_pmi) * 0.025
        else:
            pmi_multiplier = 1.0
    else:  # Strong contraction (PMI < 48)
        pmi_multiplier = 1.05 if is_defensive else (0.95 if is_cyclical else 1.0)

    # Modulate by regime confidence
    regime_confidence = macro_data.get("confidence", 0.7)
    pmi_effect = (pmi_multiplier - 1.0) * regime_confidence
    final_pmi_multiplier = 1.0 + pmi_effect

    return adjusted_z * final_pmi_multiplier


def _apply_unemployment_adjustment(adjusted_z: float, sector: str, country: str) -> float:
    """Apply unemployment overlay (consumer discretionary only)."""
    if sector != "Consumer Discretionary":
        return adjusted_z

    unemployment_rate = fetch_unemployment_rate(country)
    if unemployment_rate is None:
        return adjusted_z

    # Country-specific thresholds
    if country == "USA":
        threshold_high, threshold_low = 7.0, 4.0
    elif country in ["Germany", "France", "UK"]:
        threshold_high, threshold_low = 10.0, 6.0
    elif country == "Japan":
        threshold_high, threshold_low = 4.0, 2.5
    else:
        threshold_high, threshold_low = 7.0, 4.0  # Default to US norms

    unemployment_multiplier = 1.0

    if unemployment_rate > threshold_high:
        # Conservative penalty: -2% per 1% above threshold
        unemployment_multiplier = 1.0 - (unemployment_rate - threshold_high) * 0.02
        unemployment_multiplier = max(unemployment_multiplier, 0.95)  # Floor at -5%
    elif unemployment_rate < threshold_low:
        # Conservative boost: +1.5% per 1% below threshold
        unemployment_multiplier = 1.0 + (threshold_low - unemployment_rate) * 0.015
        unemployment_multiplier = min(unemployment_multiplier, 1.03)  # Cap at +3%

    return adjusted_z * unemployment_multiplier


def _apply_regime_overlay(
    adjusted_z: float,
    regime: str,
    is_cyclical: bool,
    is_defensive: bool,
    composite_z: float,
) -> float:
    """Apply regime overlay for broader business cycle effects."""
    regime_multiplier = 1.0

    if regime == "RECESSION":
        regime_multiplier = 0.97 if is_cyclical else (1.02 if is_defensive else 1.0)
    elif regime == "LATE_CYCLE":
        regime_multiplier = 0.98 if is_cyclical else (1.01 if is_defensive else 1.0)
    elif regime == "EARLY_CYCLE":
        regime_multiplier = 1.03 if is_cyclical else (0.99 if is_defensive else 1.0)

    return adjusted_z * regime_multiplier
