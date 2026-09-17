"""Public surface for the Phase-4 fund schemas.

Re-exports the four typed I/O schemas (``PortfolioMandate``, ``ConstraintSet``,
``ViewSet``, ``AllocDecision``) with their nested models, the shared MiFID-facing
enums, and the model-agnostic ``structured_call`` helper (with
``StructuredOutputError`` / the ``SupportsStructuredOutput`` protocol). Import
from ``fund.schemas`` rather than the individual modules.

The load-bearing invariant holds across the surface: schemas carry structured
*inputs*, never weights. ``optimizer`` is imported only inside the mapping
methods (``ConstraintSet.to_mean_risk_config`` / ``ViewSet.to_black_litterman_config``),
so importing this package pulls in no optimizer code.
"""

from __future__ import annotations

from fund.schemas.constraint_set import (
    Bounds,
    ConstraintSet,
    EsgPolicy,
    UniverseFilters,
)
from fund.schemas.decision import AllocDecision, ConstraintSetRef
from fund.schemas.enums import (
    GicsSector,
    Horizon,
    KnowledgeLevel,
    LossReaction,
    MomentsEstimator,
    ObjectiveChoice,
    RiskMeasureChoice,
    RiskToleranceBand,
    UncertaintyLevel,
)
from fund.schemas.mandate import PortfolioMandate, RunTriggers
from fund.schemas.questionnaire import (
    CapacityAnswers,
    EsgAnswers,
    KnowledgeAnswers,
    MiFIDAnswers,
    MiFIDQuestion,
    ObjectivesAnswers,
    SuitabilityAssessment,
)
from fund.schemas.structured import (
    StructuredOutputError,
    SupportsStructuredOutput,
    structured_call,
)
from fund.schemas.views import View, ViewKind, ViewSet

__all__ = [
    "AllocDecision",
    "Bounds",
    "CapacityAnswers",
    "ConstraintSet",
    "ConstraintSetRef",
    "EsgAnswers",
    "EsgPolicy",
    "GicsSector",
    "Horizon",
    "KnowledgeAnswers",
    "KnowledgeLevel",
    "LossReaction",
    "MiFIDAnswers",
    "MiFIDQuestion",
    "MomentsEstimator",
    "ObjectiveChoice",
    "ObjectivesAnswers",
    "PortfolioMandate",
    "RiskMeasureChoice",
    "RiskToleranceBand",
    "RunTriggers",
    "StructuredOutputError",
    "SuitabilityAssessment",
    "SupportsStructuredOutput",
    "UncertaintyLevel",
    "UniverseFilters",
    "View",
    "ViewKind",
    "ViewSet",
    "structured_call",
]
