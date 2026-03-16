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
from datetime import datetime, timezone
from typing import Any

from baml_client import b
from baml_client.types import BusinessCyclePhase, MacroRegimeCalibration
from sqlalchemy.orm import Session

from app.repositories.macro_regime_repository import MacroRegimeRepository
from app.services._progress import ProgressCallback, _noop

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
    "manufacturing_pmi",
    "services_pmi",
    "composite_pmi",
    "unemployment_rate",
    "inflation_rate",
    "gdp_growth_rate",
    "industrial_production",
    "government_debt_gdp",
    "budget_balance_gdp",
    "leading_economic_index",
    "interest_rate",
    "core_inflation",
    "consumer_confidence",
    "business_confidence",
    "non_farm_payrolls",
    "initial_jobless_claims",
    "ism_manufacturing_new_orders",
}


def _build_macro_summary(
    repo: MacroRegimeRepository,
    country: str,
) -> str:
    """Assemble a markdown-formatted macro summary for the LLM from DB rows."""
    sections: list[str] = [f"## {country}"]

    # IlSole24Ore forecasts (unique consensus data not in TE)
    indicators = repo.get_economic_indicators(country=country)
    forecast_rows: list[tuple[str, str]] = []
    for ind in indicators:
        if ind.gdp_growth_6m is not None:
            forecast_rows.append(("GDP 6m forecast", f"{ind.gdp_growth_6m:.2f}%"))
        if ind.inflation_6m is not None:
            forecast_rows.append(("Inflation 6m forecast", f"{ind.inflation_6m:.1f}%"))
        if ind.earnings_12m is not None:
            forecast_rows.append(("Earnings 12m forecast", f"{ind.earnings_12m:.1f}%"))
    if forecast_rows:
        sections.append("### Consensus Forecasts")
        sections.append("| Indicator | Value |")
        sections.append("|-----------|-------|")
        for label, value in forecast_rows:
            sections.append(f"| {label} | {value} |")

    # Trading Economics indicators (PMI, unemployment, CPI, etc.)
    te_rows = repo.get_te_indicators(country=country)
    te_table: list[tuple[str, str, str]] = []
    for row in te_rows:
        if row.indicator_key in _KEY_TE_INDICATORS and row.value is not None:
            label = (row.raw_name if row.raw_name else row.indicator_key).replace("|", "\\|")
            unit = row.unit if row.unit else ""
            te_table.append((label, f"{row.value:.2f}", unit))
    if te_table:
        sections.append("### Economic Indicators")
        sections.append("| Indicator | Value | Unit |")
        sections.append("|-----------|------:|------|")
        for label, value, unit in te_table:
            sections.append(f"| {label} | {value} | {unit} |")

    # Bond yields — compute 10Y-2Y spread if available
    bond_yields = repo.get_bond_yields(country=country)
    yield_map: dict[str, float] = {}
    for bond in bond_yields:
        if bond.yield_value is not None:
            yield_map[bond.maturity] = bond.yield_value
    if yield_map:
        sections.append("### Bond Yields")
        sections.append("| Maturity | Yield |")
        sections.append("|----------|------:|")
        for m in sorted(yield_map):
            sections.append(f"| {m} | {yield_map[m]:.2f}% |")
        if "10Y" in yield_map and "2Y" in yield_map:
            spread = yield_map["10Y"] - yield_map["2Y"]
            signal = "steepening" if spread > 0 else "inverted"
            sections.append(f"10Y-2Y spread: **{spread:+.2f}%** ({signal})")

    # News summary — latest AI-generated sentiment signal (optional)
    news_summary = repo.get_macro_news_summary(country)
    if news_summary is not None and (
        news_summary.sentiment is not None or news_summary.summary is not None
    ):
        sections.append("### Recent News Summary")
        if news_summary.sentiment is not None:
            score_part = (
                f" (score: {news_summary.sentiment_score:.2f})"
                if news_summary.sentiment_score is not None
                else ""
            )
            sections.append(f"Sentiment: {news_summary.sentiment}{score_part}")
        if news_summary.summary:
            sections.append(news_summary.summary)

    return "\n".join(sections) if len(sections) > 1 else ""


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
    country: str = "USA",
    macro_summary_override: str | None = None,
    force_refresh: bool = False,
) -> CalibrationResult:
    """Classify business cycle phase and return calibrated (δ, τ) for Black-Litterman.

    By default, returns the cached calibration from the ``macro_calibrations``
    table if one exists.  Set ``force_refresh=True`` to invoke the LLM and
    persist the fresh result.

    Args:
        session: Active SQLAlchemy session.
        country: Country/region name to fetch macro data for.
        macro_summary_override: If provided, skip DB fetch and use this text directly.
            Useful for testing or when passing externally sourced macro context.
        force_refresh: When True, bypass the cache and call the LLM.

    Returns:
        :class:`CalibrationResult` with clamped ``delta``, ``tau``, phase, confidence,
        and rationale.

    Raises:
        ValueError: If no macro data is found in the DB and no override is given.
    """
    repo = MacroRegimeRepository(session)

    # ── Return cached result if available ──
    if not force_refresh and macro_summary_override is None:
        cached = repo.get_macro_calibration(country)
        if cached is not None:
            logger.info("Returning cached calibration for country=%s", country)
            return CalibrationResult(
                phase=BusinessCyclePhase(cached.phase),
                delta=cached.delta,
                tau=cached.tau,
                confidence=cached.confidence,
                rationale=cached.rationale or "",
                macro_summary=cached.macro_summary or "",
            )

    # ── Build macro summary ──
    if macro_summary_override is not None:
        macro_summary = macro_summary_override
    else:
        macro_summary = _build_macro_summary(repo, country)
        if not macro_summary:
            raise ValueError(
                f"No macro data found in DB for country '{country}'. "
                "Fetch macro data first via POST /api/v1/macro-data/fetch."
            )

    # ── Call LLM ──
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

    result = CalibrationResult(
        phase=raw.phase,
        delta=delta,
        tau=tau,
        confidence=confidence,
        rationale=raw.rationale,
        macro_summary=macro_summary,
    )

    # ── Persist to DB (only for non-override requests) ──
    if macro_summary_override is None:
        try:
            repo.upsert_macro_calibration(
                country=country,
                data={
                    "phase": raw.phase.value,
                    "delta": delta,
                    "tau": tau,
                    "confidence": confidence,
                    "rationale": raw.rationale,
                    "macro_summary": macro_summary,
                },
            )
            logger.info("Persisted calibration for country=%s phase=%s", country, raw.phase.value)
        except Exception:
            logger.exception("Failed to persist calibration for country=%s (non-fatal)", country)

    return result


# ---------------------------------------------------------------------------
# Standalone bulk calibrate (callable from routes and scheduler)
# ---------------------------------------------------------------------------


def run_bulk_calibrate(
    countries: list[str],
    force_refresh: bool,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Execute batch macro calibration for multiple countries.

    Args:
        countries: List of country names to calibrate.
        force_refresh: Bypass cache and invoke the LLM for each country.
        on_progress: Optional callback for progress updates.

    Returns:
        Result dict with ``countries_processed`` and ``error_count``.
    """
    from app.database import database_manager

    on_progress(total=len(countries))
    all_errors: list[str] = []

    with database_manager.get_session() as session:
        for idx, country in enumerate(countries, 1):
            on_progress(current=idx, current_country=country)
            try:
                classify_macro_regime(
                    session=session,
                    country=country,
                    force_refresh=force_refresh,
                )
                session.commit()
            except Exception as e:
                logger.error("Calibration failed for %s: %s", country, e)
                all_errors.append(f"{country}: {e}")
                session.rollback()

    result_dict = {
        "countries_processed": len(countries),
        "error_count": len(all_errors),
    }
    on_progress(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        errors=all_errors,
        result=result_dict,
    )
    return result_dict


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
