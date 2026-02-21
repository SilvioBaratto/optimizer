"""Macro regime calibration service.

Workflow:
  1. Fetch recent macro indicators from the DB (EconomicIndicator, BondYield,
     TradingEconomicsIndicator) for a target country/region.
  2. Format them into a compact text summary.
  3. Call the BAML ``ClassifyMacroRegime`` function.
  4. Post-process: clamp delta ∈ [1.0, 10.0], tau ∈ [0.001, 0.1].
  5. Return calibrated (delta, tau) along with a typed ``MacroRegimeCalibration``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from baml_client import b
from baml_client.types import BusinessCyclePhase, MacroRegimeCalibration
from sqlalchemy.orm import Session

from app.repositories.macro_regime_repository import MacroRegimeRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

DELTA_MIN: float = 1.0
DELTA_MAX: float = 10.0
TAU_MIN: float = 0.001
TAU_MAX: float = 0.1

# Phase-based default parameters (used as fallback when BAML returns out-of-range)
_PHASE_DEFAULTS: dict[str, tuple[float, float]] = {
    BusinessCyclePhase.EARLY_EXPANSION: (2.25, 0.05),
    BusinessCyclePhase.MID_EXPANSION: (2.75, 0.025),
    BusinessCyclePhase.LATE_EXPANSION: (3.5, 0.01),
    BusinessCyclePhase.RECESSION: (5.0, 0.05),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Calibrated Black-Litterman parameters with supporting context."""

    phase: BusinessCyclePhase
    delta: float
    tau: float
    confidence: float
    rationale: str
    macro_summary: str  # the text fed to the LLM


# ---------------------------------------------------------------------------
# DB fetch helpers
# ---------------------------------------------------------------------------

# TE indicator keys considered most relevant for cycle classification
_KEY_TE_INDICATORS = {
    "Manufacturing PMI",
    "Services PMI",
    "Composite PMI",
    "Unemployment Rate",
    "Inflation Rate",
    "GDP Growth Rate",
    "Leading Economic Index",
    "Credit Default Swaps",
    "Government Bond 10Y",
    "Government Bond 2Y",
}


def _build_macro_summary(
    repo: MacroRegimeRepository,
    country: str,
) -> str:
    """Assemble a compact textual macro summary for the LLM from DB rows."""
    lines: list[str] = [f"Country/Region: {country}"]

    # Economic indicators (IlSole24Ore)
    indicators = repo.get_economic_indicators(country=country)
    for ind in indicators:
        if ind.source == "ilsole_real":
            parts: list[str] = []
            if ind.gdp_growth_qq is not None:
                parts.append(f"GDP QQ: {ind.gdp_growth_qq:.2f}%")
            if ind.unemployment is not None:
                parts.append(f"Unemployment: {ind.unemployment:.1f}%")
            if ind.consumer_prices is not None:
                parts.append(f"CPI: {ind.consumer_prices:.1f}%")
            if ind.industrial_production is not None:
                parts.append(f"Industrial production: {ind.industrial_production:.1f}%")
            if ind.st_rate is not None:
                parts.append(f"Short-term rate: {ind.st_rate:.2f}%")
            if ind.lt_rate is not None:
                parts.append(f"Long-term rate: {ind.lt_rate:.2f}%")
            if parts:
                lines.append("Economic indicators: " + ", ".join(parts))
        elif ind.source == "ilsole_forecast":
            parts = []
            if ind.gdp_growth_6m is not None:
                parts.append(f"GDP 6m forecast: {ind.gdp_growth_6m:.2f}%")
            if ind.inflation_6m is not None:
                parts.append(f"Inflation 6m forecast: {ind.inflation_6m:.1f}%")
            if ind.earnings_12m is not None:
                parts.append(f"Earnings 12m forecast: {ind.earnings_12m:.1f}%")
            if parts:
                lines.append("Forecasts: " + ", ".join(parts))

    # Trading Economics indicators (PMI, unemployment, CPI, etc.)
    te_rows = repo.get_te_indicators(country=country)
    te_parts: list[str] = []
    for row in te_rows:
        if row.indicator_key in _KEY_TE_INDICATORS and row.value is not None:
            unit = f" {row.unit}" if row.unit else ""
            te_parts.append(f"{row.indicator_key}: {row.value:.2f}{unit}")
    if te_parts:
        lines.append("Trading Economics: " + "; ".join(te_parts))

    # Bond yields — compute 10Y-2Y spread if available
    bond_yields = repo.get_bond_yields(country=country)
    yield_map: dict[str, float] = {}
    for bond in bond_yields:
        if bond.yield_value is not None:
            yield_map[bond.maturity] = bond.yield_value
    if yield_map:
        yield_parts = [f"{m}: {v:.2f}%" for m, v in sorted(yield_map.items())]
        lines.append("Bond yields: " + ", ".join(yield_parts))
        if "10Y" in yield_map and "2Y" in yield_map:
            spread = yield_map["10Y"] - yield_map["2Y"]
            lines.append(
                f"10Y-2Y spread: {spread:+.2f}% ({'steepening' if spread > 0 else 'inverted'})"
            )

    return "\n".join(lines) if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# Clamping helpers
# ---------------------------------------------------------------------------


def _clamp_delta(value: float) -> float:
    return max(DELTA_MIN, min(DELTA_MAX, value))


def _clamp_tau(value: float) -> float:
    return max(TAU_MIN, min(TAU_MAX, value))


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Main service entry points
# ---------------------------------------------------------------------------


def classify_macro_regime(
    session: Session,
    country: str = "United States",
    macro_summary_override: str | None = None,
) -> CalibrationResult:
    """Classify business cycle phase and return calibrated (δ, τ) for Black-Litterman.

    Args:
        session: Active SQLAlchemy session.
        country: Country/region name to fetch macro data for.
        macro_summary_override: If provided, skip DB fetch and use this text directly.
            Useful for testing or when passing externally sourced macro context.

    Returns:
        :class:`CalibrationResult` with clamped ``delta``, ``tau``, phase, confidence,
        and rationale.

    Raises:
        ValueError: If no macro data is found in the DB and no override is given.
    """
    repo = MacroRegimeRepository(session)

    if macro_summary_override is not None:
        macro_summary = macro_summary_override
    else:
        macro_summary = _build_macro_summary(repo, country)
        if not macro_summary:
            raise ValueError(
                f"No macro data found in DB for country '{country}'. "
                "Fetch macro data first via POST /api/v1/macro-data/fetch."
            )

    raw: MacroRegimeCalibration = b.ClassifyMacroRegime(macro_summary=macro_summary)

    delta = _clamp_delta(raw.delta)
    tau = _clamp_tau(raw.tau)
    confidence = _clamp_confidence(raw.confidence)

    if delta != raw.delta or tau != raw.tau:
        logger.warning(
            "LLM returned out-of-range values (delta=%.4f, tau=%.4f); clamped to (%.4f, %.4f)",
            raw.delta,
            raw.tau,
            delta,
            tau,
        )

    return CalibrationResult(
        phase=raw.phase,
        delta=delta,
        tau=tau,
        confidence=confidence,
        rationale=raw.rationale,
        macro_summary=macro_summary,
    )


def build_bl_config_from_calibration(
    result: CalibrationResult,
    views: tuple[str, ...] = (),
) -> dict:
    """Return a dict of kwargs that can construct a ``BlackLittermanConfig``.

    This wires the calibrated (delta, tau) back into the optimizer config layer.
    The caller can do::

        from optimizer.views._config import BlackLittermanConfig
        from optimizer.moments._config import MomentEstimationConfig, MuEstimatorType

        prior_cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EQUILIBRIUM,
            risk_aversion=result.delta,   # δ from LLM
        )
        config = BlackLittermanConfig(
            views=views,
            tau=result.tau,               # τ from LLM
            prior_config=prior_cfg,
        )

    Returns a plain dict so the response is JSON-serialisable.
    """
    return {
        "views": list(views),
        "tau": result.tau,
        "prior_config": {
            "mu_estimator": "equilibrium",
            "risk_aversion": result.delta,
            "cov_estimator": "ledoit_wolf",
        },
    }
