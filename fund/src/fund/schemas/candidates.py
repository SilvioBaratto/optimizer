"""The candidate portfolios a run computes for the user to choose from.

A run fits four methods over one returns panel and one set of shared constraints:
``hrp``, ``mean_risk``, ``min_risk`` and ``risk_parity``. They are always listed in
that alphabetical id order, which says nothing about how good a candidate is, and the
user, never the model, picks one. A candidate reports only neutral measures
(volatility, CVaR, its largest position, how many names it holds, the turnover from
current holdings) and the constraint check on its weights; mean return and Sharpe
ratio are deliberately absent.

Nothing in these models orders, scores or singles out a candidate, and
``extra="forbid"`` refuses such a field when a persisted set is read back, so a
stored row cannot acquire one. Pure, frozen pydantic-v2 data: importing this module
pulls in no optimizer code.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CANDIDATE_IDS",
    "Candidate",
    "CandidateId",
    "CandidateMetrics",
    "CandidateSet",
    "ConstraintCheck",
]

CandidateId = Literal["hrp", "mean_risk", "min_risk", "risk_parity"]

# Alphabetical by id, so the order is fixed by the names and never by a result.
CANDIDATE_IDS: tuple[CandidateId, ...] = ("hrp", "mean_risk", "min_risk", "risk_parity")

_FROZEN_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class CandidateMetrics(BaseModel):
    """The neutral, in-sample measures shown for one candidate.

    Attributes:
        annualized_volatility: Ex-ante standard deviation of the candidate's returns
            over the fitted panel, annualised by the panel's bars per year.
        cvar_95: Mean loss in the worst 5% of bars, as a positive fraction per bar.
        largest_position: The largest absolute weight.
        n_positions: Names held above ``fund.tools.mandate.HELD_TOL``.
        turnover: ``sum |w - current|`` over the target and the held tickers alike,
            the same L1 distance the drift band measures.
    """

    model_config = _FROZEN_STRICT

    annualized_volatility: float = Field(ge=0.0)
    cvar_95: float
    largest_position: float = Field(ge=0.0)
    n_positions: int = Field(ge=0)
    turnover: float = Field(ge=0.0)


class ConstraintCheck(BaseModel):
    """The constraint verdict on one candidate's weights.

    ``passed`` holds exactly when ``violations`` is empty; ``warnings`` (a soft
    drawdown tier) never fail the check.
    """

    model_config = _FROZEN_STRICT

    passed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def _passed_mirrors_violations(self) -> ConstraintCheck:
        if self.passed == bool(self.violations):
            raise ValueError("a check passes exactly when it records no violations")
        return self


class Candidate(BaseModel):
    """One method's portfolio, or the reason it has none.

    Attributes:
        candidate_id: The method id, one of :data:`CANDIDATE_IDS`.
        label: A description built from the method's settings only.
        settings: The effective optimizer configuration, JSON-safe.
        feasible: Whether its method solved, nothing is breached and
            ``constraint_check`` passed; only as strong as that check. The pure
            ``compute_candidates`` tool checks the shared floor, cap and budget
            only; the bound agent tool replaces that with the risk gate's live
            check (ESG and K&E screens, eligible universe, loss ceilings), and only
            a set that has been through it says whether the user may choose a
            candidate.
        weights: Ticker to weight; empty when the method produced no weights, kept
            when it solved but breached so the user can see why.
        metrics: The neutral measures, absent when there are no weights.
        constraint_check: The verdict on ``weights`` (bounds only, or the live
            check; see ``feasible``), absent when there are none.
        breaches: Why the candidate is not feasible (a solver failure, a cardinality
            limit, the check's violations); empty for a feasible one.
    """

    model_config = _FROZEN_STRICT

    candidate_id: CandidateId
    label: str
    settings: dict[str, Any]
    feasible: bool
    weights: dict[str, float] = Field(default_factory=dict)
    metrics: CandidateMetrics | None = None
    constraint_check: ConstraintCheck | None = None
    breaches: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def _feasibility_is_consistent(self) -> Candidate:
        if not self.feasible:
            if not self.breaches:
                raise ValueError("an infeasible candidate must name a breach")
            return self
        check = self.constraint_check
        complete = bool(self.weights) and self.metrics is not None
        if not complete or check is None or not check.passed or self.breaches:
            raise ValueError(
                "a feasible candidate needs weights, metrics and a passed check, "
                "and no breach"
            )
        return self


class CandidateSet(BaseModel):
    """The four candidates one computation produced, over one panel.

    Attributes:
        asof: The decision bar the panel ends on.
        universe: The tickers every method was fitted over.
        missing: Requested tickers dropped for lack of usable prices.
        n_observations: Return rows in the shared panel.
        periods_per_year: Bars a year of the panel holds (annualises the metrics).
        candidates: Exactly one candidate per id, in :data:`CANDIDATE_IDS` order.
    """

    model_config = _FROZEN_STRICT

    asof: dt.date
    universe: tuple[str, ...]
    missing: tuple[str, ...] = ()
    n_observations: int = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    candidates: tuple[Candidate, ...]

    @model_validator(mode="after")
    def _one_candidate_per_id_in_id_order(self) -> CandidateSet:
        ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if ids != CANDIDATE_IDS:
            raise ValueError(
                f"candidates must be exactly {', '.join(CANDIDATE_IDS)} in that "
                f"order (got {', '.join(ids) or 'none'})"
            )
        return self

    def selectable_ids(self) -> tuple[CandidateId, ...]:
        """Return the ids the user may choose, in :data:`CANDIDATE_IDS` order."""
        return tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.feasible
        )

    def candidate(self, candidate_id: str) -> Candidate | None:
        """Return the candidate with ``candidate_id``, or ``None`` for an unknown id."""
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def digest(self) -> str:
        """Return the sha256 hex digest of the set's sorted-key JSON.

        The same canonical text the audit row stores, so the digest equals that row's
        ``llm_response_hash`` and does not depend on the order dict keys arrived in.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
