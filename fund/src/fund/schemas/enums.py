"""MiFID-facing ``str, Enum`` vocabulary shared by the Phase-4 schemas.

These are the fund-side vocabulary: a client-friendly subset/rename of the
``optimizer.optimization`` enums. The mapping onto the optimizer enums lives in
the schema methods (``ConstraintSet.to_mean_risk_config`` etc., Tasks 3-4), never
here — this module imports nothing from ``optimizer`` / ``deepagents`` / ``app``
so schema construction stays optimizer-free. Values are lowercase snake_case
(repo-wide ``str, Enum`` idiom).
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "GicsSector",
    "Horizon",
    "KnowledgeLevel",
    "LossReaction",
    "MomentsEstimator",
    "ObjectiveChoice",
    "RiskMeasureChoice",
    "RiskToleranceBand",
    "UncertaintyLevel",
]


class ObjectiveChoice(str, Enum):
    """MiFID suitability objective (D13). Maps onto ``ObjectiveFunctionType``."""

    PROTECTION = "protection"
    INCOME = "income"
    GROWTH = "growth"
    MAX = "max"


class RiskMeasureChoice(str, Enum):
    """Downside risk-measure subset (D34). Maps onto ``RiskMeasureType``."""

    VARIANCE = "variance"
    SEMI_VARIANCE = "semi_variance"
    CVAR = "cvar"
    CDAR = "cdar"
    MAX_DRAWDOWN = "max_drawdown"


class Horizon(str, Enum):
    """Investment horizon bucket (D13)."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class GicsSector(str, Enum):
    """The 11 GICS sectors — ESG exclusions / sector caps (D16, D39)."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


class MomentsEstimator(str, Enum):
    """Moments-estimator selector (D23). ``LEDOIT_WOLF`` is the default."""

    LEDOIT_WOLF = "ledoit_wolf"
    EMPIRICAL = "empirical"
    EW = "ew"


class UncertaintyLevel(str, Enum):
    """Robust-optimization uncertainty level (D35)."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


class KnowledgeLevel(str, Enum):
    """MiFID knowledge-&-experience pillar (Fase 5). Low levels (``none`` /
    ``basic``) drive the ``UniverseFilters`` restrictions (no complex / no
    leverage, tighter caps)."""

    NONE = "none"
    BASIC = "basic"
    INFORMED = "informed"
    ADVANCED = "advanced"


class LossReaction(str, Enum):
    """Client's reaction to an extreme drawdown scenario (Fase 5). Drives the
    downside ``risk_measure`` + tail ``beta`` (protection → CVaR/CDaR/MaxDD),
    distinct from the attitudinal risk-tolerance Likert that scores appetite."""

    SELL_ALL = "sell_all"
    SELL_SOME = "sell_some"
    HOLD = "hold"
    BUY_MORE = "buy_more"


class RiskToleranceBand(str, Enum):
    """The 5 named MiFID risk categories (Fase 5, SPEC §8.1). The appetite score
    buckets into one of these; the category name is recorded in the suitability
    assessment alongside the derived ``a_gamma``."""

    DEFENSIVE = "defensive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GROWTH = "growth"
    AGGRESSIVE = "aggressive"
