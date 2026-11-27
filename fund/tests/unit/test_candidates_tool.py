"""``compute_candidates``: four neutral candidate portfolios over one returns panel.

Drives the pure tool over the shared seeded SQLite panel (``_fund_fakes``): the four
methods in their fixed id order, fitted on one universe with one set of bounds, one
budget and one moment estimator; a method that fails or breaches is reported
infeasible beside the others; universe-level failures stay tool errors.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
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
from fund.schemas.enums import RiskMeasureChoice
from fund.tools import candidates as candidates_module
from fund.tools import optimize as optimize_module
from fund.tools.candidates import compute_candidates
from fund.tools.optimize import optimize_portfolio
from fund.tools.prices import load_price_frame

_TOLERANCE = 1e-9
# A QP solver reaches its optimum only to about this precision.
_SOLVER_TOLERANCE = 1e-6
_SHARED_SETTINGS = ("min_weights", "max_weights", "budget", "prior_config")
# A budget below full investment, so methods that solve at a unit budget must scale.
_PARTIAL_BUDGET = 0.95
# Above the smallest HRP and risk-parity weight on the seeded panel, so it binds.
_BINDING_FLOOR = 0.3
# Below every method's largest unconstrained weight even divided by the budget.
_TIGHT_CAP = 0.33
# A cap the concentrated stand-in can only be clipped to, leaving the budget short.
_CLIPPING_CAP = 0.5
# Bars of history at the young decision bar, under the common-sample floor.
_YOUNG_BARS = 10
_MARKED_WEIGHTS = [0.2, 0.3, 0.5]


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
    assert [candidate.candidate_id for candidate in candidate_set.candidates] == list(
        CANDIDATE_IDS
    )
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
        shared_values = {
            repr(candidate.settings[name]) for candidate in candidate_set.candidates
        }
        assert len(shared_values) == 1


@pytest.mark.criterion("REQ-FUND-043")
@pytest.mark.parametrize(
    ("risk_measure", "label", "calibrated"),
    [
        (RiskMeasureChoice.VARIANCE, "Mean-risk (maximize_utility, variance)", False),
        (RiskMeasureChoice.CVAR, "Mean-risk (maximize_utility, cvar)", True),
    ],
    ids=["variance", "cvar"],
)
def test_mean_risk_candidate_equals_optimize_portfolio_under_the_profile(
    db_session, risk_measure: RiskMeasureChoice, label: str, calibrated: bool
) -> None:
    """Validates fund.tools.candidates.compute_candidates: ``mean_risk`` is exactly the
    weights ``optimize_portfolio`` computes under the client's profile, including the
    risk-aversion rescaling a first-order risk measure gets."""
    seed_panel(db_session)
    constraints = _profile(risk_measure=risk_measure)
    single = optimize_portfolio(db_session, ASOF, UNIVERSE, constraints=constraints)

    mean_risk = _candidates(db_session, constraints=constraints).candidate("mean_risk")

    assert single["ok"] is True
    assert mean_risk is not None
    assert mean_risk.weights == single["data"]["weights"]
    assert mean_risk.label == label
    assert mean_risk.settings["risk_aversion"] == single["data"]["risk_aversion"]
    assert mean_risk.settings["risk_free_rate"] == single["data"]["risk_free_rate"]
    rescaled = mean_risk.settings["risk_aversion"] != constraints["risk_aversion"]
    assert rescaled is calibrated


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
    assert [min_risk.weights[ticker] for ticker in UNIVERSE] == pytest.approx(
        list(expected), abs=_SOLVER_TOLERANCE
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


@pytest.mark.criterion("REQ-FUND-043")
def test_every_candidate_meets_a_binding_floor_under_a_partial_budget(
    db_session,
) -> None:
    """Validates fund.tools.candidates.compute_candidates: HRP and risk parity, which
    solve fully invested, meet the same floor and budget as the mean-risk methods."""
    seed_panel(db_session)
    bounds = Bounds(min_weights=_BINDING_FLOOR, budget=_PARTIAL_BUDGET)

    candidate_set = _candidates(db_session, constraints=_profile(bounds=bounds))

    assert candidate_set.selectable_ids() == CANDIDATE_IDS
    for candidate in candidate_set.candidates:
        weights = candidate.weights.values()
        assert sum(weights) == pytest.approx(_PARTIAL_BUDGET, abs=_TOLERANCE)
        assert min(weights) >= _BINDING_FLOOR - _TOLERANCE
        assert candidate.settings["min_weights"] == _BINDING_FLOOR
        assert candidate.settings["budget"] == _PARTIAL_BUDGET
    for candidate_id in ("hrp", "risk_parity"):
        candidate = candidate_set.candidate(candidate_id)
        assert candidate is not None
        assert min(candidate.weights.values()) == pytest.approx(
            _BINDING_FLOOR, abs=_SOLVER_TOLERANCE
        )


@pytest.mark.criterion("REQ-FUND-043")
def test_every_candidate_meets_the_cap_in_absolute_weight_under_a_partial_budget(
    db_session,
) -> None:
    """Validates fund.tools.candidates.compute_candidates: under a partial budget the
    cap binds every method at the same absolute weight, never at cap times budget."""
    seed_panel(db_session)
    bounds = Bounds(max_weights=_TIGHT_CAP, budget=_PARTIAL_BUDGET)

    candidate_set = _candidates(db_session, constraints=_profile(bounds=bounds))

    assert candidate_set.selectable_ids() == CANDIDATE_IDS
    for candidate in candidate_set.candidates:
        assert candidate.metrics is not None
        assert sum(candidate.weights.values()) == pytest.approx(
            _PARTIAL_BUDGET, abs=_TOLERANCE
        )
        assert max(candidate.weights.values()) <= _TIGHT_CAP + _TOLERANCE
        assert candidate.metrics.largest_position == pytest.approx(
            _TIGHT_CAP, abs=_SOLVER_TOLERANCE
        )


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
    # concentrated vector the cap can only clip, leaving the budget short.
    real_build = optimize_module.build_mean_risk

    def concentrated_min_risk(config: MeanRiskConfig) -> Any:
        """Return the concentrated stand-in for a min-variance config only."""
        if config.objective is ObjectiveFunctionType.MINIMIZE_RISK:
            return _FixedWeights([0.98, 0.01, 0.01])
        return real_build(config)

    monkeypatch.setattr(optimize_module, "build_mean_risk", concentrated_min_risk)

    candidate_set = _candidates(
        db_session, constraints=_profile(bounds=Bounds(max_weights=_CLIPPING_CAP))
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
    assert [breach["type"] for breach in risk_parity.breaches] == ["solver_failed"]
    assert "non-finite" in risk_parity.breaches[0]["reason"]
    assert min_risk.feasible is False
    assert min_risk.weights["AAA"] == pytest.approx(_CLIPPING_CAP)
    assert [breach["type"] for breach in min_risk.breaches] == ["budget"]
    assert min_risk.constraint_check is not None
    assert min_risk.constraint_check.passed is False


def _short_risk_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk parity returns one weight fewer than the universe has tickers."""
    monkeypatch.setattr(
        candidates_module, "build_risk_budgeting", lambda config: _FixedWeights([1.0])
    )


def _unmeasurable_risk_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measuring risk parity's (marked) weights raises; every other method's works."""
    monkeypatch.setattr(
        candidates_module,
        "build_risk_budgeting",
        lambda config: _FixedWeights(_MARKED_WEIGHTS),
    )
    real_metrics = candidates_module._metrics

    def failing_metrics(
        panel: Any, weights: Any, by_ticker: dict[str, float], current: Any
    ) -> Any:
        """Raise for the marked weights only, else measure them for real."""
        if list(by_ticker.values()) == _MARKED_WEIGHTS:
            raise RuntimeError("metrics failed")
        return real_metrics(panel, weights, by_ticker, current)

    monkeypatch.setattr(candidates_module, "_metrics", failing_metrics)


@pytest.mark.criterion("REQ-FUND-044")
@pytest.mark.parametrize(
    "break_risk_parity",
    [_short_risk_parity, _unmeasurable_risk_parity],
    ids=["pairing", "metrics"],
)
def test_a_method_whose_cleaning_or_metrics_raise_is_infeasible_and_the_others_return(
    db_session,
    monkeypatch,
    break_risk_parity: Callable[[pytest.MonkeyPatch], None],
) -> None:
    """Validates fund.tools.candidates.compute_candidates: a failure after the fit,
    while pairing the weights with tickers or measuring them, makes only that method
    infeasible."""
    seed_panel(db_session)
    break_risk_parity(monkeypatch)

    candidate_set = _candidates(db_session, constraints=_profile())
    risk_parity = candidate_set.candidate("risk_parity")

    assert candidate_set.selectable_ids() == ("hrp", "mean_risk", "min_risk")
    assert risk_parity is not None
    assert risk_parity.feasible is False
    assert risk_parity.weights == {}
    assert [breach["type"] for breach in risk_parity.breaches] == ["solver_failed"]


@pytest.mark.criterion("REQ-FUND-044")
def test_cardinality_is_a_breach_for_methods_that_cannot_enforce_it(
    db_session,
) -> None:
    """Validates fund.tools.candidates.compute_candidates: HRP and risk parity hold
    every asset, so a cardinality limit below the universe breaches them."""
    seed_panel(db_session)
    limit = len(UNIVERSE) - 1
    over_limit = {"type": "cardinality", "value": len(UNIVERSE), "limit": limit}

    candidate_set = _candidates(db_session, constraints=_profile(cardinality=limit))

    for candidate_id in ("hrp", "risk_parity"):
        candidate = candidate_set.candidate(candidate_id)
        assert candidate is not None
        assert candidate.feasible is False
        assert over_limit in candidate.breaches
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
            abs(candidate.weights.get(ticker, 0.0) - current.get(ticker, 0.0))
            for ticker in tickers
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
    young_bar = START - dt.timedelta(days=LEAD_DAYS - _YOUNG_BARS)
    narrow_cap = 0.2

    empty = compute_candidates(db_session, ASOF, [])
    unpriced = compute_candidates(db_session, ASOF, ["ZZZ"])
    thin = compute_candidates(db_session, young_bar, UNIVERSE, constraints=_profile())
    too_narrow = compute_candidates(
        db_session,
        ASOF,
        UNIVERSE,
        constraints={**_profile(), "max_weights": narrow_cap},
    )

    assert empty == {"ok": False, "error": "empty universe"}
    assert unpriced == {"ok": False, "error": "no priced assets in universe"}
    assert too_narrow["ok"] is False
    assert "cannot reach budget" in too_narrow["error"]
    assert thin["ok"] is False
    assert "insufficient common history to compute candidates" in thin["error"]
