"""``compute_candidates``: four neutral candidate portfolios over one returns panel.

Drives the pure tool over the shared seeded SQLite panel (``_fund_fakes``): the four
methods in their fixed id order, fitted on one universe with one set of bounds, one
budget and one moment estimator; a method that fails or breaches is reported
infeasible beside the others; universe-level failures stay tool errors.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest
from _fund_fakes import (
    ASOF,
    LEAD_DAYS,
    START,
    UNIVERSE,
    make_constraint_set,
    seed_panel,
)
from optimizer.moments import MomentEstimationConfig
from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    build_mean_risk,
)
from optimizer.preprocessing import prices_to_returns

from fund.agents.toolsets import _optimizer_constraints
from fund.schemas.candidates import CANDIDATE_IDS, CandidateSet
from fund.schemas.constraint_set import Bounds
from fund.tools import candidates as candidates_module
from fund.tools import optimize as optimize_module
from fund.tools.candidates import compute_candidates
from fund.tools.optimize import optimize_portfolio
from fund.tools.prices import load_price_frame

_TOLERANCE = 1e-9
_SHARED_SETTINGS = ("min_weights", "max_weights", "budget", "prior_config")


def _profile(**overrides: Any) -> dict[str, Any]:
    """The mapped optimizer constraints of the shared growth/variance profile."""
    return _optimizer_constraints(make_constraint_set().model_copy(update=overrides))


def _candidates(session: Any, **kwargs: Any) -> CandidateSet:
    result = compute_candidates(session, ASOF, UNIVERSE, **kwargs)
    assert result["ok"] is True, result
    return CandidateSet.model_validate(result["data"])


@pytest.mark.criterion("REQ-FUND-043")
@pytest.mark.criterion("REQ-FUND-045")
def test_compute_candidates_returns_four_candidates_over_one_panel(
    db_session, monkeypatch
) -> None:
    """Validates fund.tools.candidates.compute_candidates: the four ids in order, one
    price load, one universe and the same bounds, budget and prior for every method."""
    seed_panel(db_session)
    loads: list[tuple[str, ...]] = []

    def counting_load(*args: Any, **kwargs: Any) -> Any:
        """Record the tickers of each price load, then load them for real."""
        loads.append(tuple(args[2]))
        return load_price_frame(*args, **kwargs)

    monkeypatch.setattr(optimize_module, "load_price_frame", counting_load)

    candidate_set = _candidates(db_session, constraints=_profile())

    assert loads == [tuple(UNIVERSE)]
    assert [c.candidate_id for c in candidate_set.candidates] == list(CANDIDATE_IDS)
    assert candidate_set.universe == tuple(UNIVERSE)
    assert candidate_set.selectable_ids() == CANDIDATE_IDS
    for candidate in candidate_set.candidates:
        assert set(candidate.weights) == set(UNIVERSE)
        assert sum(candidate.weights.values()) == pytest.approx(1.0, abs=_TOLERANCE)
        assert candidate.metrics is not None
        assert set(candidate.metrics.model_dump()) == {
            "annualized_volatility",
            "cvar_95",
            "largest_position",
            "n_positions",
            "turnover",
        }
        assert candidate.constraint_check is not None
        assert candidate.constraint_check.passed is True
    for name in _SHARED_SETTINGS:
        assert len({repr(c.settings[name]) for c in candidate_set.candidates}) == 1


@pytest.mark.criterion("REQ-FUND-043")
def test_mean_risk_candidate_equals_optimize_portfolio_under_the_profile(
    db_session,
) -> None:
    """Validates fund.tools.candidates.compute_candidates: ``mean_risk`` is exactly the
    weights ``optimize_portfolio`` computes under the client's profile."""
    seed_panel(db_session)
    constraints = _profile()
    single = optimize_portfolio(db_session, ASOF, UNIVERSE, constraints=constraints)

    mean_risk = _candidates(db_session, constraints=constraints).candidate("mean_risk")

    assert single["ok"] is True
    assert mean_risk is not None
    assert mean_risk.weights == single["data"]["weights"]
    assert mean_risk.label == "Mean-risk (maximize_utility, variance)"
    assert mean_risk.settings["risk_aversion"] == single["data"]["risk_aversion"]
    assert mean_risk.settings["risk_free_rate"] == single["data"]["risk_free_rate"]


@pytest.mark.criterion("REQ-FUND-043")
def test_min_risk_candidate_is_minimum_variance_under_the_shared_bounds(
    db_session,
) -> None:
    """Validates fund.tools.candidates.compute_candidates: ``min_risk`` is the
    minimum-variance portfolio under the profile's bounds, without its penalties."""
    seed_panel(db_session)
    cap = 0.45
    constraints = _profile(bounds=Bounds(max_weights=cap), l2_coef=0.2)
    frame, _ = load_price_frame(db_session, ASOF, UNIVERSE)
    expected = (
        build_mean_risk(
            MeanRiskConfig(
                objective=ObjectiveFunctionType.MINIMIZE_RISK,
                risk_measure=RiskMeasureType.VARIANCE,
                max_weights=cap,
                prior_config=MomentEstimationConfig(),
            )
        )
        .fit(prices_to_returns(frame))
        .weights_
    )

    candidate_set = _candidates(db_session, constraints=constraints)
    min_risk = candidate_set.candidate("min_risk")

    assert min_risk is not None
    assert min_risk.label == "Minimum variance"
    assert min_risk.settings["objective"] == "minimize_risk"
    assert min_risk.settings["risk_measure"] == "variance"
    assert min_risk.settings["l2_coef"] == 0.0
    assert [min_risk.weights[t] for t in UNIVERSE] == pytest.approx(
        list(expected), abs=1e-6
    )
    assert min_risk.metrics is not None
    for other in candidate_set.candidates:
        assert other.metrics is not None
        assert (
            min_risk.metrics.annualized_volatility
            <= other.metrics.annualized_volatility + _TOLERANCE
        )


@pytest.mark.criterion("REQ-FUND-043")
def test_every_candidate_respects_the_position_cap(db_session) -> None:
    """Validates fund.tools.candidates.compute_candidates: a cap that binds every
    method holds for every candidate, and its check passes."""
    seed_panel(db_session)
    cap = 0.36

    candidate_set = _candidates(
        db_session, constraints=_profile(bounds=Bounds(max_weights=cap))
    )

    assert candidate_set.selectable_ids() == CANDIDATE_IDS
    for candidate in candidate_set.candidates:
        assert candidate.metrics is not None
        assert max(candidate.weights.values()) <= cap + _TOLERANCE
        assert candidate.metrics.largest_position <= cap + _TOLERANCE
        assert candidate.settings["max_weights"] == cap


class _FixedWeights:
    """Stands in for a skfolio estimator whose fit lands on fixed weights."""

    def __init__(self, weights: list[float]) -> None:
        self.weights_ = np.asarray(weights, dtype=float)

    def fit(self, returns: Any) -> _FixedWeights:
        """Ignore ``returns`` and return the estimator, as skfolio's ``fit`` does."""
        return self


def _raise_solver_error(config: Any) -> Any:
    raise RuntimeError("solver blew up")


@pytest.mark.criterion("REQ-FUND-044")
def test_a_failing_method_is_infeasible_and_the_others_still_return(
    db_session, monkeypatch
) -> None:
    """Validates fund.tools.candidates.compute_candidates: a raising, a non-finite and
    a breaching method are each infeasible with a breach; the rest stay selectable."""
    seed_panel(db_session)
    monkeypatch.setattr(candidates_module, "build_hrp", _raise_solver_error)
    monkeypatch.setattr(
        candidates_module,
        "build_risk_budgeting",
        lambda config: _FixedWeights([np.nan, 0.5, 0.5]),
    )
    # ``fit_mean_risk`` reads the module's builder, so a min-variance config sees a
    # concentrated vector the 0.5 cap can only clip, leaving the budget short.
    real_build = optimize_module.build_mean_risk

    def concentrated_min_risk(config: MeanRiskConfig) -> Any:
        """Return the concentrated stand-in for a min-variance config only."""
        if config.objective is ObjectiveFunctionType.MINIMIZE_RISK:
            return _FixedWeights([0.98, 0.01, 0.01])
        return real_build(config)

    monkeypatch.setattr(optimize_module, "build_mean_risk", concentrated_min_risk)

    candidate_set = _candidates(
        db_session, constraints=_profile(bounds=Bounds(max_weights=0.5))
    )
    hrp, mean_risk, min_risk, risk_parity = candidate_set.candidates

    assert candidate_set.selectable_ids() == ("mean_risk",)
    assert mean_risk.feasible is True
    assert hrp.feasible is False
    assert hrp.weights == {}
    assert hrp.metrics is None
    assert hrp.breaches == (
        {"type": "solver_failed", "reason": "RuntimeError: solver blew up"},
    )
    assert risk_parity.feasible is False
    assert [b["type"] for b in risk_parity.breaches] == ["solver_failed"]
    assert "non-finite" in risk_parity.breaches[0]["reason"]
    assert min_risk.feasible is False
    assert min_risk.weights["AAA"] == pytest.approx(0.5)
    assert [b["type"] for b in min_risk.breaches] == ["budget"]
    assert min_risk.constraint_check is not None
    assert min_risk.constraint_check.passed is False


@pytest.mark.criterion("REQ-FUND-044")
def test_cardinality_is_a_breach_for_methods_that_cannot_enforce_it(
    db_session,
) -> None:
    """Validates fund.tools.candidates.compute_candidates: HRP and risk parity hold
    every asset, so a cardinality limit below the universe breaches them."""
    seed_panel(db_session)
    limit = 2

    candidate_set = _candidates(db_session, constraints=_profile(cardinality=limit))

    for candidate_id in ("hrp", "risk_parity"):
        candidate = candidate_set.candidate(candidate_id)
        assert candidate is not None
        assert candidate.feasible is False
        assert {"type": "cardinality", "value": 3, "limit": limit} in candidate.breaches
        assert set(candidate.weights) == set(UNIVERSE)
    for candidate in candidate_set.candidates:
        if candidate.feasible:
            assert candidate.metrics is not None
            assert candidate.metrics.n_positions <= limit
        else:
            assert candidate.breaches


@pytest.mark.criterion("REQ-FUND-045")
def test_turnover_is_the_l1_distance_from_current_holdings(db_session) -> None:
    """Validates fund.tools.candidates.compute_candidates: turnover sums the absolute
    weight change over held and target tickers alike; no holdings means the budget."""
    seed_panel(db_session)
    current = {"AAA": 0.5, "ZZZ": 0.5}

    from_holdings = _candidates(
        db_session, constraints=_profile(), current_weights=current
    )
    from_cash = _candidates(db_session, constraints=_profile())

    for candidate in from_holdings.candidates:
        tickers = set(candidate.weights) | set(current)
        expected = sum(
            abs(candidate.weights.get(t, 0.0) - current.get(t, 0.0)) for t in tickers
        )
        assert candidate.metrics is not None
        assert candidate.metrics.turnover == pytest.approx(expected)
    for candidate in from_cash.candidates:
        assert candidate.metrics is not None
        assert candidate.metrics.turnover == pytest.approx(1.0)


@pytest.mark.criterion("REQ-FUND-043")
def test_compute_candidates_is_deterministic(db_session) -> None:
    """Validates fund.tools.candidates.compute_candidates: the same data gives the
    same candidate set, down to its digest."""
    seed_panel(db_session)
    constraints = _profile()

    first = compute_candidates(db_session, ASOF, UNIVERSE, constraints=constraints)
    second = compute_candidates(db_session, ASOF, UNIVERSE, constraints=constraints)

    assert first == second
    assert (
        CandidateSet.model_validate(first["data"]).digest()
        == CandidateSet.model_validate(second["data"]).digest()
    )


@pytest.mark.criterion("REQ-FUND-044")
def test_universe_level_errors_stay_tool_errors(db_session) -> None:
    """Validates fund.tools.candidates.compute_candidates: an empty, unpriced, too
    young or too narrow universe is an error envelope, not four infeasible methods."""
    seed_panel(db_session)
    young_bar = START - dt.timedelta(days=LEAD_DAYS - 10)

    empty = compute_candidates(db_session, ASOF, [])
    unpriced = compute_candidates(db_session, ASOF, ["ZZZ"])
    thin = compute_candidates(db_session, young_bar, UNIVERSE, constraints=_profile())
    too_narrow = compute_candidates(
        db_session, ASOF, UNIVERSE, constraints={**_profile(), "max_weights": 0.2}
    )

    assert empty == {"ok": False, "error": "empty universe"}
    assert unpriced == {"ok": False, "error": "no priced assets in universe"}
    assert too_narrow["ok"] is False
    assert "cannot reach budget" in too_narrow["error"]
    assert thin["ok"] is False
    assert "insufficient common history to compute candidates" in thin["error"]
