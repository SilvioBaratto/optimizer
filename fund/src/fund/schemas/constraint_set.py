"""MiFID risk-profile schema (deep_agent Fase-5 table) → optimizer knobs.

``ConstraintSet`` is the profiler's structured output: the double-binding
risk-aversion ``a_gamma`` (``min(tolerance, capacity)``), the objective and
downside risk-measure choices, the tail confidence ``beta``, the drawdown-ceiling
tiers ``nu1/nu2/nu3``, plus long-only bounds, ESG policy, universe filters and the
usual regularisation / estimator selectors. It is pure, serialisable, frozen
(hashable) pydantic-v2 data — constructing one imports **no** ``optimizer`` code.

``to_mean_risk_config`` is the bridge: it maps the MiFID vocabulary onto
``optimizer.optimization.MeanRiskConfig``, importing the optimizer **lazily inside
the method** so schema construction stays cheap and optimizer-free. The
``_OBJECTIVE_MAP`` / ``_RISK_MEASURE_MAP`` dicts hold plain enum-value strings (not
optimizer enum objects) for the same reason — they must be total (every fund enum
member maps; an unmapped member is a hard ``KeyError``, never a silent default).

Deliberately **not** mapped (stored/validated only, enforced in Fase 7): the
``nu1/nu2/nu3`` drawdown ceilings (``MeanRiskConfig`` has no drawdown-ceiling
field), ``horizon``, ``esg``, ``universe_filters``, ``moments_estimator`` and
``uncertainty_level``. ``beta`` **is** mapped onto ``cvar_beta`` / ``cdar_beta``
so a CVaR/CDaR profile honours the client's chosen tail confidence (D34).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from fund.schemas.enums import (
    GicsSector,
    Horizon,
    MomentsEstimator,
    ObjectiveChoice,
    RiskMeasureChoice,
    UncertaintyLevel,
)

if TYPE_CHECKING:  # import only for the annotation; runtime import is method-local
    from optimizer.optimization import MeanRiskConfig

__all__ = ["Bounds", "ConstraintSet", "EsgPolicy", "UniverseFilters"]


class Bounds(BaseModel):
    """Per-asset weight bounds — long-only, fully invested, no leverage (D17)."""

    model_config = ConfigDict(frozen=True)

    min_weights: float = Field(default=0.0, ge=0.0)  # long-only: no shorts
    max_weights: float = Field(default=1.0, le=1.0)
    budget: float = Field(default=1.0, gt=0.0)  # fully invested


class EsgPolicy(BaseModel):
    """ESG constraints (D16): GICS-sector exclusions + optional thresholds.

    ``min_taxonomy`` / ``min_sfdr`` are proportions in ``[0, 1]`` (provider data
    lands later); ``pai_flags`` names the Principal Adverse Indicators to screen.
    """

    model_config = ConfigDict(frozen=True)

    exclusions: tuple[GicsSector, ...] = ()  # excluded GICS sectors
    min_taxonomy: float | None = Field(default=None, ge=0.0, le=1.0)
    min_sfdr: float | None = Field(default=None, ge=0.0, le=1.0)
    pai_flags: tuple[str, ...] = ()


class UniverseFilters(BaseModel):
    """Knowledge-&-experience universe restrictions (D32).

    Low K&E → exclude complex/leveraged instruments and cap per-position weight.
    Stored/validated only here; the ``pre_selection`` pipeline consumes them in
    Fase 7 (not mapped into ``MeanRiskConfig``).
    """

    model_config = ConfigDict(frozen=True)

    no_complex: bool = False
    no_leverage: bool = False
    max_position_cap: float | None = Field(default=None, gt=0.0, le=1.0)


class ConstraintSet(BaseModel):
    """MiFID profile → optimizer knobs (deep_agent Fase-5 table).

    ``a_gamma`` is the regulatory double-binding ``min(risk_tolerance,
    loss_capacity)`` risk-aversion. ``beta`` is the tail-measure confidence used
    by CVaR/CDaR. ``nu1/nu2/nu3`` are drawdown-ceiling tiers as fractions of
    capital (validated here, enforced by the Fase-7 ``risk_check`` gate).
    """

    model_config = ConfigDict(frozen=True)

    portfolio_id: str  # D1 namespacing
    base_currency: str = Field(pattern=r"^[A-Z]{3}$")  # D8 ISO-4217
    a_gamma: float = Field(gt=0.0)  # min(tolerance, capacity), double-binding
    objective: ObjectiveChoice
    risk_measure: RiskMeasureChoice
    beta: float = Field(default=0.95, gt=0.0, lt=1.0)  # tail confidence (CVaR/CDaR)
    nu1: float = Field(ge=0.0, le=1.0)  # drawdown ceiling tiers (fractions of
    nu2: float = Field(ge=0.0, le=1.0)  # capital) — stored/validated only, mapped
    nu3: float = Field(ge=0.0, le=1.0)  # to nothing here (Fase-7 risk_check, D20:75)
    horizon: Horizon
    esg: EsgPolicy = EsgPolicy()
    universe_filters: UniverseFilters = UniverseFilters()
    bounds: Bounds = Bounds()
    cardinality: int | None = Field(default=None, gt=0)  # D27 cap per profile
    moments_estimator: MomentsEstimator = MomentsEstimator.LEDOIT_WOLF  # D23
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.NONE  # D35
    l1_coef: float = Field(default=0.0, ge=0.0)  # D26
    l2_coef: float = Field(default=0.0, ge=0.0)  # D26
    risk_free_rate: float | None = None  # D24; None → 0.0 at map time

    def to_mean_risk_config(self) -> MeanRiskConfig:
        """Map onto ``MeanRiskConfig``. fund is the bridge → the import is OK."""
        from optimizer.optimization import (
            MeanRiskConfig,
            ObjectiveFunctionType,
            RiskMeasureType,
        )

        return MeanRiskConfig(
            objective=ObjectiveFunctionType(_OBJECTIVE_MAP[self.objective]),
            risk_measure=RiskMeasureType(_RISK_MEASURE_MAP[self.risk_measure]),
            risk_aversion=self.a_gamma,
            min_weights=self.bounds.min_weights,
            max_weights=self.bounds.max_weights,
            budget=self.bounds.budget,
            cardinality=self.cardinality,
            cvar_beta=self.beta,
            cdar_beta=self.beta,
            l1_coef=self.l1_coef,
            l2_coef=self.l2_coef,
            risk_free_rate=0.0 if self.risk_free_rate is None else self.risk_free_rate,
        )


# ---------------------------------------------------------------------------
# MiFID → optimizer enum maps (total; plain strings keep construction lazy).
# ---------------------------------------------------------------------------
# Objective spectrum (deep_agent.md:300 + :336): protection → MinRisk, the
# income/growth middle rides the risk-aversion slider via MAXIMIZE_UTILITY, max →
# MAXIMIZE_RATIO. Values are optimizer enum *strings*; to_mean_risk_config rebuilds
# the enum so this module imports no optimizer code at construction time.
_OBJECTIVE_MAP: dict[ObjectiveChoice, str] = {
    ObjectiveChoice.PROTECTION: "minimize_risk",
    ObjectiveChoice.INCOME: "maximize_utility",
    ObjectiveChoice.GROWTH: "maximize_utility",
    ObjectiveChoice.MAX: "maximize_ratio",
}

# D34: the downside subset maps 1:1 by name onto RiskMeasureType.
_RISK_MEASURE_MAP: dict[RiskMeasureChoice, str] = {
    RiskMeasureChoice.VARIANCE: "variance",
    RiskMeasureChoice.SEMI_VARIANCE: "semi_variance",
    RiskMeasureChoice.CVAR: "cvar",
    RiskMeasureChoice.CDAR: "cdar",
    RiskMeasureChoice.MAX_DRAWDOWN: "max_drawdown",
}
