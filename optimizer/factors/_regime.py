"""Macro regime classification and factor group tilts.

This module provides the **macro-indicator** regime system, which classifies
economic conditions into one of five ``MacroRegime`` states (EXPANSION,
SLOWDOWN, RECESSION, RECOVERY, UNKNOWN) based on observable macro indicators (PMI,
yield curve, credit spreads, sentiment).  The resulting regime drives
``RegimeTiltConfig`` multiplicative tilts on factor group weights.

A separate **statistical** regime system exists in ``optimizer.moments``
(``HMMBlendedMu``, ``HMMBlendedCovariance``, ``RegimeRiskConfig``).  That
system fits a Gaussian HMM on return data and produces integer-labelled
latent states that control moment estimation and risk-measure selection.

The two systems are intentionally independent: macro indicators capture
fundamental economic conditions while the HMM captures statistical regimes
in asset returns.  They can and will disagree.  Use
:func:`check_regime_disagreement` to surface such disagreements.
"""

from __future__ import annotations

import logging

import pandas as pd

from optimizer.factors._config import (
    FactorGroupType,
    MacroRegime,
    RegimeThresholdConfig,
    RegimeTiltConfig,
)

logger = logging.getLogger(__name__)


def classify_regime_composite(
    macro_data: pd.DataFrame,
    thresholds: RegimeThresholdConfig | None = None,
) -> MacroRegime:
    """Classify macro regime using the multi-indicator composite score.

    Uses ISM PMI, 2s10s yield curve spread, and HY credit spread to
    compute a composite score S_t as defined in the macroeconomic
    analysis framework (Chapter 7).

    The input DataFrame should contain any of these columns:
    ``pmi`` (Manufacturing PMI), ``spread_2s10s`` (10Y-2Y spread in %),
    ``hy_oas`` (HY OAS in basis points), ``sentiment`` (news score).

    Parameters
    ----------
    macro_data : pd.DataFrame
        Macro indicators indexed by date.
    thresholds : RegimeThresholdConfig or None
        Scoring thresholds.  Defaults to the empirical calibration.

    Returns
    -------
    MacroRegime
        Regime classification based on composite score.
    """
    thresholds = thresholds or RegimeThresholdConfig()

    if len(macro_data) == 0:
        logger.warning(
            "classify_regime_composite: empty macro_data DataFrame; "
            "returning MacroRegime.UNKNOWN with neutral tilts."
        )
        return MacroRegime.UNKNOWN

    latest = macro_data.iloc[-1]

    # Component scores (default 0 when data missing)
    s_pmi = 0
    if "pmi" in macro_data.columns and pd.notna(latest.get("pmi")):
        pmi = float(latest["pmi"])
        s_pmi = (
            1
            if pmi > thresholds.pmi_expansion
            else (-1 if pmi < thresholds.pmi_contraction else 0)
        )

    s_2s10s = 0
    if "spread_2s10s" in macro_data.columns and pd.notna(latest.get("spread_2s10s")):
        spread = float(latest["spread_2s10s"])
        s_2s10s = (
            1
            if spread > thresholds.spread_2s10s_steep
            else (-1 if spread < thresholds.spread_2s10s_inversion else 0)
        )

    s_hy = 0
    if "hy_oas" in macro_data.columns and pd.notna(latest.get("hy_oas")):
        hy = float(latest["hy_oas"])
        s_hy = (
            1
            if hy < thresholds.hy_oas_risk_on
            else (-1 if hy > thresholds.hy_oas_risk_off else 0)
        )

    s_sent = 0
    if "sentiment" in macro_data.columns and pd.notna(latest.get("sentiment")):
        sent = float(latest["sentiment"])
        s_sent = (
            1
            if sent > thresholds.sentiment_positive
            else (-1 if sent < thresholds.sentiment_negative else 0)
        )

    # 3-indicator composite
    s_t = s_pmi + s_2s10s + s_hy

    # Use augmented score when sentiment is available
    if s_sent != 0:
        s_aug = s_t + s_sent
        if s_aug >= 3:
            return MacroRegime.EXPANSION
        if s_aug <= -3:
            return MacroRegime.RECESSION
        if s_aug > 0:
            return MacroRegime.EXPANSION
        if s_aug == 0:
            return MacroRegime.SLOWDOWN
        return MacroRegime.RECOVERY

    # 3-indicator mapping
    if s_t >= 2:
        return MacroRegime.EXPANSION
    if s_t <= -2:
        return MacroRegime.RECESSION
    if s_t > 0:
        return MacroRegime.EXPANSION
    if s_t == 0:
        return MacroRegime.SLOWDOWN
    return MacroRegime.RECOVERY


def classify_regime(
    macro_data: pd.DataFrame,
    thresholds: RegimeThresholdConfig | None = None,
) -> MacroRegime:
    """Classify the current macro-economic regime.

    Uses a simple heuristic based on GDP growth and leading
    indicators.  The regime is determined by the latest
    observation's position relative to trend.

    When richer indicators (``pmi``, ``spread_2s10s``, ``hy_oas``)
    are present, delegates to :func:`classify_regime_composite`.

    Parameters
    ----------
    macro_data : pd.DataFrame
        Macro indicators with columns that may include
        ``gdp_growth``, ``leading_indicator``, ``yield_spread``,
        ``unemployment_rate``.  Index is date.
    thresholds : RegimeThresholdConfig or None
        Scoring thresholds forwarded to the composite classifier.

    Returns
    -------
    MacroRegime
        Current regime classification.
    """
    if len(macro_data) == 0:
        logger.warning(
            "classify_regime: empty macro_data DataFrame; "
            "returning MacroRegime.UNKNOWN with neutral tilts."
        )
        return MacroRegime.UNKNOWN

    # Delegate to composite classifier when richer indicators are available.
    _composite_cols = {"pmi", "spread_2s10s", "hy_oas"}
    if _composite_cols & set(macro_data.columns):
        return classify_regime_composite(macro_data, thresholds=thresholds)

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

    logger.warning(
        "classify_regime: macro_data contains no recognized indicator columns "
        "(gdp_growth, yield_spread, pmi, spread_2s10s, hy_oas); "
        "returning MacroRegime.UNKNOWN with neutral tilts."
    )
    return MacroRegime.UNKNOWN


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
        case MacroRegime.UNKNOWN:
            raw_tilts = config.unknown_tilts

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


def check_regime_disagreement(
    regime_a: MacroRegime,
    regime_b: MacroRegime,
    label_a: str = "composite",
    label_b: str = "hmm",
) -> bool:
    """Check whether two regime classifications disagree.

    When the macro-indicator and HMM-based (or any two) regime systems
    produce different classifications, this function logs a ``WARNING``
    and returns ``True``.  Returns ``False`` when they agree.

    Parameters
    ----------
    regime_a, regime_b : MacroRegime
        Regime classifications from two different subsystems.
    label_a, label_b : str
        Human-readable labels for the two sources (used in the log
        message).

    Returns
    -------
    bool
        ``True`` if the regimes disagree, ``False`` otherwise.
    """
    if regime_a != regime_b:
        logger.warning(
            "Regime disagreement: %s classifies as %r but %s classifies "
            "as %r. Factor tilts use %s classification.",
            label_a,
            regime_a.value,
            label_b,
            regime_b.value,
            label_a,
        )
        return True
    return False
