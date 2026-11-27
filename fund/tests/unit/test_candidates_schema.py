"""The candidate-set schema the adviser chooses from (``fund.schemas.candidates``).

Pins the structure every reader relies on: exactly the four method ids in their fixed
alphabetical order, a feasible candidate that carries weights, metrics and a passed
check, an infeasible one that names at least one breach, and no field that could
order or single out one candidate. Pure pydantic, no database.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any

import pytest
from pydantic import ValidationError

from fund.schemas.candidates import (
    CANDIDATE_IDS,
    Candidate,
    CandidateMetrics,
    CandidateSet,
    ConstraintCheck,
)

_ASOF = dt.date(2024, 1, 30)
_UNIVERSE = ("AAA", "BBB")
_METRICS = CandidateMetrics(
    annualized_volatility=0.12,
    cvar_95=0.02,
    largest_position=0.6,
    n_positions=2,
    turnover=0.4,
)
_PASSED = ConstraintCheck(passed=True)
_SOLVER_FAILED = {"type": "solver_failed", "reason": "SolverError: no solution"}


def _feasible(candidate_id: str, weights: dict[str, float] | None = None) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,  # type: ignore[arg-type]
        label=f"method {candidate_id}",
        settings={"risk_measure": "variance", "max_weights": 1.0},
        feasible=True,
        weights=weights if weights is not None else {"AAA": 0.6, "BBB": 0.4},
        metrics=_METRICS,
        constraint_check=_PASSED,
    )


def _infeasible(candidate_id: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,  # type: ignore[arg-type]
        label=f"method {candidate_id}",
        settings={},
        feasible=False,
        breaches=(_SOLVER_FAILED,),
    )


def _candidate_set(candidates: tuple[Candidate, ...] | None = None) -> CandidateSet:
    return CandidateSet(
        asof=_ASOF,
        universe=_UNIVERSE,
        n_observations=90,
        periods_per_year=365,
        candidates=(
            candidates
            if candidates is not None
            else tuple(_feasible(candidate_id) for candidate_id in CANDIDATE_IDS)
        ),
    )


@pytest.mark.criterion("REQ-FUND-043")
def test_candidate_set_requires_the_four_ids_in_alphabetical_order() -> None:
    """Validates fund.schemas.candidates.CandidateSet: the four ids, in id order."""
    assert CANDIDATE_IDS == ("hrp", "mean_risk", "min_risk", "risk_parity")
    assert list(CANDIDATE_IDS) == sorted(CANDIDATE_IDS)
    accepted = _candidate_set()
    assert [c.candidate_id for c in accepted.candidates] == list(CANDIDATE_IDS)

    reordered = tuple(
        _feasible(candidate_id) for candidate_id in reversed(CANDIDATE_IDS)
    )
    missing_one = tuple(_feasible(candidate_id) for candidate_id in CANDIDATE_IDS[:3])
    repeated = (*missing_one, _feasible("hrp"))
    for candidates in (reordered, missing_one, repeated, ()):
        with pytest.raises(ValidationError, match="hrp, mean_risk, min_risk"):
            _candidate_set(candidates)
    with pytest.raises(ValidationError):
        _feasible("equal_weight")


@pytest.mark.criterion("REQ-FUND-045")
@pytest.mark.parametrize(
    "field", ["rank", "score", "recommended", "best", "preferred", "matches_profile"]
)
def test_candidate_set_rejects_a_ranking_or_recommendation_field(field: str) -> None:
    """Validates fund.schemas.candidates.CandidateSet and Candidate: extra fields that
    could single out a candidate are refused on read-back, at either level."""
    payload = _candidate_set().model_dump(mode="json")
    on_a_candidate = json.loads(json.dumps(payload))
    on_a_candidate["candidates"][1][field] = True
    on_the_set = {**payload, field: "mean_risk"}

    for tampered in (on_a_candidate, on_the_set):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CandidateSet.model_validate(tampered)
    ordering = re.compile(r"rank|score|recommend|best|prefer|order|profile")
    fields = [
        *CandidateSet.model_fields,
        *Candidate.model_fields,
        *CandidateMetrics.model_fields,
        *ConstraintCheck.model_fields,
    ]
    assert [name for name in fields if ordering.search(name)] == []


@pytest.mark.criterion("REQ-FUND-045")
def test_candidate_metrics_are_exactly_the_five_neutral_measures() -> None:
    """Validates fund.schemas.candidates.CandidateMetrics: the five measures only,
    finite and never a return or ratio figure."""
    assert set(CandidateMetrics.model_fields) == {
        "annualized_volatility",
        "cvar_95",
        "largest_position",
        "n_positions",
        "turnover",
    }
    with pytest.raises(ValidationError):
        CandidateMetrics(**{**_METRICS.model_dump(), "cvar_95": float("nan")})
    with pytest.raises(ValidationError):
        CandidateMetrics(**{**_METRICS.model_dump(), "sharpe_ratio": 1.2})


@pytest.mark.criterion("REQ-FUND-045")
def test_constraint_check_passes_only_without_violations() -> None:
    """Validates fund.schemas.candidates.ConstraintCheck: ``passed`` mirrors the
    violations, while a warning never fails the check."""
    breach = {"type": "max_weight", "asset": "AAA", "value": 0.7, "limit": 0.5}
    warning = {"type": "drawdown_nu1", "value": 0.06, "limit": 0.05}

    assert ConstraintCheck(passed=True, warnings=(warning,)).passed is True
    assert ConstraintCheck(passed=False, violations=(breach,)).violations == (breach,)
    with pytest.raises(ValidationError, match="violations"):
        ConstraintCheck(passed=True, violations=(breach,))
    with pytest.raises(ValidationError, match="violations"):
        ConstraintCheck(passed=False)


_FEASIBLE_GAPS: dict[str, dict[str, Any]] = {
    "no weights": {"weights": {}},
    "no metrics": {"metrics": None},
    "no check": {"constraint_check": None},
    "failed check": {
        "constraint_check": ConstraintCheck(
            passed=False, violations=({"type": "budget", "value": 0.9},)
        )
    },
    "a breach": {"breaches": (_SOLVER_FAILED,)},
}


@pytest.mark.criterion("REQ-FUND-044")
@pytest.mark.parametrize("gap", sorted(_FEASIBLE_GAPS))
def test_feasible_candidate_needs_weights_metrics_and_a_passed_check(gap: str) -> None:
    """Validates fund.schemas.candidates.Candidate: ``feasible`` is refused unless
    weights, metrics and a passed check are all present and nothing is breached."""
    complete = dict(_feasible("hrp"))

    with pytest.raises(ValidationError, match="feasible"):
        Candidate(**{**complete, **_FEASIBLE_GAPS[gap]})


@pytest.mark.criterion("REQ-FUND-044")
def test_infeasible_candidate_needs_a_breach() -> None:
    """Validates fund.schemas.candidates.Candidate: an infeasible candidate names a
    breach, and one that solved but breached keeps its weights for the adviser."""
    with pytest.raises(ValidationError, match="breach"):
        Candidate(candidate_id="hrp", label="x", settings={}, feasible=False)

    over_cap = {"type": "max_weight", "asset": "AAA", "value": 0.7, "limit": 0.5}
    solved_but_breached = Candidate(
        candidate_id="hrp",
        label="x",
        settings={},
        feasible=False,
        weights={"AAA": 0.7, "BBB": 0.3},
        metrics=_METRICS,
        constraint_check=ConstraintCheck(passed=False, violations=(over_cap,)),
        breaches=(over_cap,),
    )
    assert solved_but_breached.weights == {"AAA": 0.7, "BBB": 0.3}
    assert _infeasible("hrp").breaches == (_SOLVER_FAILED,)


@pytest.mark.criterion("REQ-FUND-044")
def test_selectable_ids_are_the_feasible_ones_in_id_order() -> None:
    """Validates fund.schemas.candidates.CandidateSet.selectable_ids and .candidate:
    infeasible candidates stay listed but are not selectable."""
    mixed = _candidate_set(
        (
            _infeasible("hrp"),
            _feasible("mean_risk"),
            _feasible("min_risk"),
            _infeasible("risk_parity"),
        )
    )
    nothing_solved = _candidate_set(
        tuple(_infeasible(candidate_id) for candidate_id in CANDIDATE_IDS)
    )

    assert mixed.selectable_ids() == ("mean_risk", "min_risk")
    assert [c.candidate_id for c in mixed.candidates] == list(CANDIDATE_IDS)
    assert mixed.candidate("risk_parity") is mixed.candidates[3]
    assert mixed.candidate("equal_weight") is None
    assert nothing_solved.selectable_ids() == ()
    assert _candidate_set().selectable_ids() == CANDIDATE_IDS


@pytest.mark.criterion("REQ-FUND-046")
def test_digest_ignores_key_order() -> None:
    """Validates fund.schemas.candidates.CandidateSet.digest: the sha256 of the
    sorted-key JSON, the same whatever order the weight and setting keys arrived in."""
    forward = _candidate_set(
        tuple(
            _feasible(candidate_id, {"AAA": 0.6, "BBB": 0.4})
            for candidate_id in CANDIDATE_IDS
        )
    )
    backward = _candidate_set(
        tuple(
            _feasible(candidate_id, {"BBB": 0.4, "AAA": 0.6})
            for candidate_id in CANDIDATE_IDS
        )
    )
    changed = _candidate_set(
        tuple(
            _feasible(candidate_id, {"AAA": 0.5, "BBB": 0.5})
            for candidate_id in CANDIDATE_IDS
        )
    )
    canonical = json.dumps(forward.model_dump(mode="json"), sort_keys=True)

    assert forward.digest() == backward.digest()
    assert forward.digest() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert changed.digest() != forward.digest()
