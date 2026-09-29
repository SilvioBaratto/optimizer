"""MiFID-facing ``str, Enum`` vocabulary shared across fund schemas.

These are the fund-side vocabulary: a client-friendly subset/rename of the
``optimizer.optimization`` enums. The mapping onto the optimizer enums lives in
the schema methods (``ConstraintSet.to_mean_risk_config`` etc.), never here —
this module imports nothing from ``optimizer`` / ``deepagents`` / ``app`` so
schema construction stays optimizer-free. Values are lowercase snake_case
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
    """MiFID suitability objective. Maps onto ``ObjectiveFunctionType``."""

    PROTECTION = "protection"
    INCOME = "income"
    GROWTH = "growth"
    MAX = "max"


class RiskMeasureChoice(str, Enum):
    """Downside risk-measure subset. Maps onto ``RiskMeasureType``."""

    VARIANCE = "variance"
    SEMI_VARIANCE = "semi_variance"
    CVAR = "cvar"
    CDAR = "cdar"
    MAX_DRAWDOWN = "max_drawdown"


class Horizon(str, Enum):
    """Investment horizon bucket."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class GicsSector(str, Enum):
    """The 11 GICS sectors used for ESG exclusions and sector caps."""

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
    """Moments-estimator selector. ``LEDOIT_WOLF`` is the default."""

    LEDOIT_WOLF = "ledoit_wolf"
    EMPIRICAL = "empirical"
    EW = "ew"


class UncertaintyLevel(str, Enum):
    """Robust-optimization uncertainty level."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


class KnowledgeLevel(str, Enum):
    """MiFID knowledge-and-experience pillar.

    Low ``none``/``basic`` levels restrict the investable universe: no complex
    products, no leverage, tighter position caps.
    """

    NONE = "none"
    BASIC = "basic"
    INFORMED = "informed"
    ADVANCED = "advanced"


class LossReaction(str, Enum):
    """Client's reaction to an extreme drawdown scenario.

    Drives the downside ``risk_measure`` and tail ``beta`` selection
    (``sell_all``/protection → CVaR/CDaR/MaxDD), distinct from the attitudinal
    risk-tolerance score that measures appetite.
    """

    SELL_ALL = "sell_all"
    SELL_SOME = "sell_some"
    HOLD = "hold"
    BUY_MORE = "buy_more"


class RiskToleranceBand(str, Enum):
    """The five named MiFID risk categories.

    The appetite score buckets into one of these; the category name is recorded
    in the suitability assessment alongside the derived ``a_gamma``.
    """

    DEFENSIVE = "defensive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GROWTH = "growth"
    AGGRESSIVE = "aggressive"
