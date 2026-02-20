"""LLM-augmented moment estimation service.

Three BAML-powered functions that use qualitative macro context to adjust
the quantitative moment estimation pipeline:

  1. calibrate_delta       — risk aversion scalar from macro regime text
  2. adapt_factor_weights  — factor group multipliers from business cycle phase
  3. select_cov_regime     — covariance estimator from news sentiment
"""

from __future__ import annotations

import logging
from typing import Any

from baml_client import b
from baml_client.types import (
    BusinessCyclePhase,
    CovEstimatorChoice,
    CovRegimeSelection,
    DeltaCalibration,
    FactorWeightAdaptation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DELTA_MIN: float = 1.0
DELTA_MAX: float = 10.0


# ---------------------------------------------------------------------------
# (a) Risk aversion calibration
# ---------------------------------------------------------------------------


def calibrate_delta(macro_text: str) -> DeltaCalibration:
    """Return a calibrated Black-Litterman delta from a macro regime description.

    Args:
        macro_text: Free-form text describing the current macro environment
            (e.g. Fed statement, GDP release, credit-spread commentary).

    Returns:
        :class:`DeltaCalibration` with ``delta`` clamped to [1.0, 10.0]
        and a one-sentence ``rationale``.
    """
    result: DeltaCalibration = b.CalibrateDelta(macro_text=macro_text)

    # Enforce hard bounds regardless of LLM output
    clamped = max(DELTA_MIN, min(DELTA_MAX, result.delta))
    if clamped != result.delta:
        logger.warning(
            "delta %.4f out of [%.1f, %.1f]; clamped to %.4f",
            result.delta,
            DELTA_MIN,
            DELTA_MAX,
            clamped,
        )
        result = DeltaCalibration(delta=clamped, rationale=result.rationale)

    return result


# ---------------------------------------------------------------------------
# (b) Factor weight adaptation
# ---------------------------------------------------------------------------


def adapt_factor_weights(
    macro_indicators: str,
    factor_groups: list[str],
) -> FactorWeightAdaptation:
    """Return factor group weight multipliers calibrated to the business cycle.

    Args:
        macro_indicators: Textual summary of macro indicators (PMI, unemployment,
            yield-curve slope, leading indicators index, etc.).
        factor_groups: Ordered list of factor group names to weight
            (e.g. ``["momentum", "value", "quality", "low_volatility"]``).

    Returns:
        :class:`FactorWeightAdaptation` with ``phase``, ``weights``
        (dict factor_group → multiplier), and ``rationale``.

    Notes:
        Multipliers are post-processed to guarantee:
        - All values are positive (min-clipped at 0.1).
        - They sum to ``len(factor_groups)`` (scale-normalised).
    """
    result: FactorWeightAdaptation = b.AdaptFactorWeights(
        macro_indicators=macro_indicators,
        factor_groups=factor_groups,
    )

    weights = _normalise_weights(result.weights, factor_groups)
    return FactorWeightAdaptation(
        phase=result.phase,
        weights=weights,
        rationale=result.rationale,
    )


def _normalise_weights(
    raw: dict[str, Any],
    factor_groups: list[str],
) -> dict[str, float]:
    """Clip to positive and rescale so the sum equals len(factor_groups)."""
    n = len(factor_groups)
    weights: dict[str, float] = {}
    for group in factor_groups:
        weights[group] = max(0.1, float(raw.get(group, 1.0)))

    total = sum(weights.values())
    if total <= 0:
        return {g: 1.0 for g in factor_groups}

    scale = n / total
    return {g: w * scale for g, w in weights.items()}


# ---------------------------------------------------------------------------
# (c) Covariance regime selection
# ---------------------------------------------------------------------------

# Mapping from BAML enum to optimizer CovEstimatorType string values
_COV_CHOICE_TO_ESTIMATOR_TYPE: dict[CovEstimatorChoice, str] = {
    CovEstimatorChoice.EMPIRICAL: "empirical",
    CovEstimatorChoice.LEDOIT_WOLF: "ledoit_wolf",
    CovEstimatorChoice.OAS: "oas",
    CovEstimatorChoice.SHRUNK: "shrunk",
    CovEstimatorChoice.EW: "ew",
    CovEstimatorChoice.GERBER: "gerber",
    CovEstimatorChoice.GRAPHICAL_LASSO_CV: "graphical_lasso_cv",
    CovEstimatorChoice.DENOISE: "denoise",
    CovEstimatorChoice.DETONE: "detone",
    CovEstimatorChoice.IMPLIED: "implied",
}


def select_cov_regime(
    news_headlines: list[str],
    avg_sentiment_score: float,
    realized_vol_30d: float,
) -> CovRegimeSelection:
    """Select the appropriate covariance estimator for the current market regime.

    Args:
        news_headlines: Recent market news headlines used for sentiment context.
        avg_sentiment_score: Average sentiment score in [-1.0, 1.0].
            Negative = bearish, positive = bullish.
        realized_vol_30d: Annualised 30-day realised volatility (e.g. 0.15 = 15%).

    Returns:
        :class:`CovRegimeSelection` with ``estimator`` (:class:`CovEstimatorChoice`),
        ``confidence`` in [0.0, 1.0], and ``rationale``.
    """
    result: CovRegimeSelection = b.SelectCovRegime(
        news_headlines=news_headlines,
        avg_sentiment_score=float(avg_sentiment_score),
        realized_vol_30d=float(realized_vol_30d),
    )

    # Clamp confidence
    confidence = max(0.0, min(1.0, result.confidence))
    return CovRegimeSelection(
        estimator=result.estimator,
        confidence=confidence,
        rationale=result.rationale,
    )


def cov_estimator_type_str(selection: CovRegimeSelection) -> str:
    """Convert a :class:`CovRegimeSelection` to an optimizer ``CovEstimatorType`` value.

    Returns the string value (e.g. ``"ledoit_wolf"``) that maps directly to
    :class:`optimizer.moments.CovEstimatorType`.
    """
    return _COV_CHOICE_TO_ESTIMATOR_TYPE.get(selection.estimator, "ledoit_wolf")
