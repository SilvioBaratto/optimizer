"""``compute_candidates``: four neutral candidate portfolios over one returns panel.

The optimizer computes every weight here, never the model, and no method is placed
above another: ``hrp``, ``mean_risk``, ``min_risk`` and ``risk_parity`` always come
back in that alphabetical id order, whatever their figures. The user picks
one later; nothing in this module orders or scores them.

Every method is fitted on one returns panel (``load_returns_panel``, the panel
``optimize_portfolio`` fits) under one shared configuration: the weight floor, the
cap (already tightened by the position limit), the budget and the moment estimators
of the prior.

* ``hrp``: hierarchical risk parity on variance, default distance and linkage;
* ``mean_risk``: the client's objective and risk measure, ``optimize_portfolio``'s
  own solve;
* ``min_risk``: minimum variance, without the L1/L2 penalties;
* ``risk_parity``: equal risk contribution on variance.

A method that raises, returns non-finite weights, holds more names than the
cardinality limit (HRP and risk parity cannot enforce one) or fails the bounds check
comes back with ``feasible=False`` and its breaches, beside the others. A failure of
the universe itself (empty, unpriced, too little history, bounds the priced assets
cannot meet) is a tool error, since no method could run. Deterministic: no random
state, one panel, and a plain-JSON result.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from optimizer.optimization import (
    HRPConfig,
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskBudgetingConfig,
    RiskMeasureType,
    build_hrp,
    build_risk_budgeting,
)
from skfolio import Portfolio

from fund.observe import drift_l1
from fund.schemas.candidates import (
    CANDIDATE_IDS,
    Candidate,
    CandidateId,
    CandidateMetrics,
    CandidateSet,
    ConstraintCheck,
)
from fund.tools._base import ToolResult, coerce_date, err, ok, tool_envelope
from fund.tools.mandate import HELD_TOL
from fund.tools.optimize import (
    ReturnsPanel,
    _calibrate_risk_aversion,
    _clean_weights,
    fit_mean_risk,
    load_returns_panel,
)
from fund.tools.risk import risk_check

if TYPE_CHECKING:
    import datetime as dt

    import pandas as pd
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# The tail the CVaR metric averages: the worst 5% of bars.
CVAR_BETA = 0.95

# Breach types a method itself produces; a constraint check never emits them, so a
# re-check can replace the check's violations and keep these.
SOLVER_FAILED = "solver_failed"
CARDINALITY_BREACH = "cardinality"
_METHOD_BREACHES = frozenset({SOLVER_FAILED, CARDINALITY_BREACH})

# A check that could not run is a failed check: the weights have not been shown to
# respect the mandate.
CHECK_FAILED = "constraint_check_failed"

_ACTION = "compute candidates"
# A breach reason is the exception type and the first line of its message, cut to
# this length: solver messages run to many lines, and the traceback is in the log.
_MAX_REASON_CHARS = 200
_NON_FINITE = "the method returned non-finite weights"
_NO_WEIGHTS = "the method produced no weights"

# What HRP and risk budgeting assume when the shared config leaves a bound open.
_NO_FLOOR = 0.0
_NO_CAP = 1.0
# HRP and risk budgeting always solve a fully invested book: weights summing to one.
_UNIT_BUDGET = 1.0

_HRP_LABEL = "Hierarchical risk parity (variance)"
_MIN_RISK_LABEL = "Minimum variance"
_RISK_PARITY_LABEL = "Equal risk contribution (variance)"
_MEAN_RISK_LABEL = "Mean-risk ({objective}, {risk_measure})"


def _floor(shared: MeanRiskConfig) -> float:
    return _NO_FLOOR if shared.min_weights is None else float(shared.min_weights)


def _cap(shared: MeanRiskConfig) -> float:
    return _NO_CAP if shared.max_weights is None else float(shared.max_weights)


def _budget(shared: MeanRiskConfig) -> float:
    return _UNIT_BUDGET if shared.budget is None else float(shared.budget)


# HRP and risk budgeting solve on the shared bounds divided by the budget, and
# ``_clean_weights`` then scales their unit-sum book to the budget, so the final
# weights meet the same absolute floor, cap and budget as the mean-risk methods. A
# unit-sum book cannot hold more than one in a name, so a cap above that binds nothing.
def _unit_floor(shared: MeanRiskConfig) -> float:
    return _floor(shared) / _budget(shared)


def _unit_cap(shared: MeanRiskConfig) -> float:
    return min(_cap(shared) / _budget(shared), _UNIT_BUDGET)


def _hrp_config(_returns: pd.DataFrame, shared: MeanRiskConfig) -> HRPConfig:
    return HRPConfig(
        risk_measure=RiskMeasureType.VARIANCE,
        prior_config=shared.prior_config,
        min_weights=_unit_floor(shared),
        max_weights=_unit_cap(shared),
    )


def _mean_risk_config(returns: pd.DataFrame, shared: MeanRiskConfig) -> MeanRiskConfig:
    return _calibrate_risk_aversion(shared, returns)


def _min_risk_config(_returns: pd.DataFrame, shared: MeanRiskConfig) -> MeanRiskConfig:
    # The penalties tilt the client's own utility; minimum variance is the plain
    # least-variance book under the same bounds.
    return replace(
        shared,
        objective=ObjectiveFunctionType.MINIMIZE_RISK,
        risk_measure=RiskMeasureType.VARIANCE,
        l1_coef=0.0,
        l2_coef=0.0,
    )


def _risk_parity_config(
    _returns: pd.DataFrame, shared: MeanRiskConfig
) -> RiskBudgetingConfig:
    return RiskBudgetingConfig(
        prior_config=shared.prior_config,
        min_weights=_unit_floor(shared),
        max_weights=_unit_cap(shared),
    )


def _fitted_weights(model: Any, returns: pd.DataFrame) -> np.ndarray:
    model.fit(returns)
    if model.weights_ is None:
        raise ValueError(_NO_WEIGHTS)
    return np.asarray(model.weights_, dtype=float)


def _fit_hrp(returns: pd.DataFrame, config: HRPConfig) -> np.ndarray:
    return _fitted_weights(build_hrp(config), returns)


def _fit_risk_parity(returns: pd.DataFrame, config: RiskBudgetingConfig) -> np.ndarray:
    return _fitted_weights(build_risk_budgeting(config), returns)


def _mean_risk_label(shared: MeanRiskConfig) -> str:
    return _MEAN_RISK_LABEL.format(
        objective=shared.objective.value, risk_measure=shared.risk_measure.value
    )


@dataclass(frozen=True)
class _Fitter:
    """How one method is described, configured from the shared settings and fitted.

    Attributes:
        label: The method's description, built from the shared settings only.
        configure: The method's own config, from the returns and the shared config.
        fit: The weights that config solves to on the returns, one per column.
    """

    label: Callable[[MeanRiskConfig], str]
    configure: Callable[[pd.DataFrame, MeanRiskConfig], Any]
    fit: Callable[[pd.DataFrame, Any], np.ndarray]


def _constant(label: str) -> Callable[[MeanRiskConfig], str]:
    return lambda _shared: label


_FITTERS: dict[CandidateId, _Fitter] = {
    "hrp": _Fitter(_constant(_HRP_LABEL), _hrp_config, _fit_hrp),
    "mean_risk": _Fitter(_mean_risk_label, _mean_risk_config, fit_mean_risk),
    "min_risk": _Fitter(_constant(_MIN_RISK_LABEL), _min_risk_config, fit_mean_risk),
    "risk_parity": _Fitter(
        _constant(_RISK_PARITY_LABEL), _risk_parity_config, _fit_risk_parity
    ),
}


@dataclass(frozen=True)
class _Solved:
    """A method's cleaned weights and measures, before a constraint check judges them.

    Attributes:
        candidate_id: The method id.
        label: The method's description.
        settings: The effective configuration, JSON-safe.
        weights: Ticker to cleaned weight.
        metrics: The neutral measures of ``weights``.
        method_breaches: Breaches the method itself caused (a cardinality limit).
    """

    candidate_id: CandidateId
    label: str
    settings: dict[str, Any]
    weights: dict[str, float]
    metrics: CandidateMetrics
    method_breaches: tuple[dict[str, Any], ...]

    def judged(self, check: ConstraintCheck) -> Candidate:
        """Return the candidate ``check`` makes of these weights."""
        breaches = (*self.method_breaches, *check.violations)
        return Candidate(
            candidate_id=self.candidate_id,
            label=self.label,
            settings=self.settings,
            feasible=not breaches,
            weights=self.weights,
            metrics=self.metrics,
            constraint_check=check,
            breaches=breaches,
        )


def constraint_check_from(result: ToolResult) -> ConstraintCheck:
    """Read a ``risk_check``-shaped verdict envelope as a ``ConstraintCheck``.

    Args:
        result: The envelope of ``risk_check`` or of the bound live check, whose
            ``data`` carries ``passed``, ``violations`` and ``warnings``.

    Returns:
        The verdict; an error envelope fails closed as a ``constraint_check_failed``
        violation carrying the error.
    """
    if not result.get("ok"):
        failure = {"type": CHECK_FAILED, "reason": str(result.get("error"))}
        return ConstraintCheck(passed=False, violations=(failure,))
    data = result["data"]
    return ConstraintCheck(
        passed=bool(data["passed"]),
        violations=tuple(data["violations"]),
        warnings=tuple(data.get("warnings") or ()),
    )


def apply_constraint_check(candidate: Candidate, check: ConstraintCheck) -> Candidate:
    """Return ``candidate`` judged by ``check`` in place of its current check.

    The method's own breaches (a solver failure, a cardinality limit) stay; the
    previous check's violations give way to ``check``'s, and the candidate is
    feasible only when no breach remains.

    Args:
        candidate: A computed candidate.
        check: The verdict on its weights.

    Returns:
        The re-judged candidate; one without weights has nothing to judge and is
        returned unchanged.
    """
    if candidate.metrics is None or not candidate.weights:
        return candidate
    solved = _Solved(
        candidate_id=candidate.candidate_id,
        label=candidate.label,
        settings=candidate.settings,
        weights=candidate.weights,
        metrics=candidate.metrics,
        method_breaches=tuple(
            breach
            for breach in candidate.breaches
            if breach.get("type") in _METHOD_BREACHES
        ),
    )
    return solved.judged(check)


def recheck_candidates(
    candidate_set: CandidateSet,
    check: Callable[[dict[str, float]], ConstraintCheck],
) -> CandidateSet:
    """Return ``candidate_set`` with every solved candidate judged by ``check``.

    Args:
        candidate_set: The computed set.
        check: The verdict on one candidate's weights.

    Returns:
        The same set, in the same id order, each candidate with weights carrying
        ``check``'s verdict on them.
    """
    judged = tuple(
        apply_constraint_check(candidate, check(candidate.weights))
        if candidate.weights
        else candidate
        for candidate in candidate_set.candidates
    )
    return CandidateSet.model_validate({**dict(candidate_set), "candidates": judged})


def _settings(config: Any, shared: MeanRiskConfig) -> dict[str, Any]:
    # HRP and risk budgeting solve on bounds divided by the budget and have no budget
    # field; the bounds their final weights meet are the shared absolute ones, so
    # those are what every method records.
    effective = {
        **dataclasses.asdict(config),
        "min_weights": _floor(shared),
        "max_weights": _cap(shared),
        "budget": shared.budget,
    }
    return json.loads(json.dumps(effective, default=str))


def _metrics(
    panel: ReturnsPanel,
    weights: np.ndarray,
    by_ticker: dict[str, float],
    current: dict[str, float],
) -> CandidateMetrics:
    portfolio = Portfolio(X=panel.returns, weights=weights, cvar_beta=CVAR_BETA)
    sizes = np.abs(weights)
    return CandidateMetrics(
        annualized_volatility=float(portfolio.standard_deviation)
        * math.sqrt(panel.periods),
        cvar_95=float(portfolio.cvar),
        largest_position=float(sizes.max(initial=0.0)),
        n_positions=int(np.count_nonzero(sizes > HELD_TOL)),
        turnover=drift_l1(current, by_ticker),
    )


def _cardinality_breaches(
    n_positions: int, shared: MeanRiskConfig
) -> tuple[dict[str, Any], ...]:
    limit = shared.cardinality
    if limit is None or n_positions <= limit:
        return ()
    return ({"type": CARDINALITY_BREACH, "value": n_positions, "limit": limit},)


def _bounds_check(weights: dict[str, float], shared: MeanRiskConfig) -> ConstraintCheck:
    bounds = {
        "min_weights": shared.min_weights,
        "max_weights": shared.max_weights,
        "budget": shared.budget,
    }
    present = {name: value for name, value in bounds.items() if value is not None}
    return constraint_check_from(risk_check(weights, constraints=present))


def _short_reason(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    reason = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    return reason[:_MAX_REASON_CHARS]


def _unsolved(
    candidate_id: CandidateId, label: str, settings: dict[str, Any], reason: str
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        label=label,
        settings=settings,
        feasible=False,
        breaches=({"type": SOLVER_FAILED, "reason": reason},),
    )


def _solved_weights(fitter: _Fitter, config: Any, panel: ReturnsPanel) -> np.ndarray:
    solved = fitter.fit(panel.returns, config)
    if not np.all(np.isfinite(solved)):
        raise ValueError(_NON_FINITE)
    return _clean_weights(solved, panel.config)


def _by_ticker(panel: ReturnsPanel, weights: np.ndarray) -> dict[str, float]:
    assets = [str(column) for column in panel.returns.columns]
    return {asset: float(weight) for asset, weight in zip(assets, weights, strict=True)}


def _candidate(
    candidate_id: CandidateId, panel: ReturnsPanel, current: dict[str, float]
) -> Candidate:
    fitter = _FITTERS[candidate_id]
    label = fitter.label(panel.config)
    settings: dict[str, Any] = {}
    # Everything from configuring the method to measuring its weights sits in here:
    # one method failing must not cost the user the other three.
    try:
        config = fitter.configure(panel.returns, panel.config)
        settings = _settings(config, panel.config)
        weights = _solved_weights(fitter, config, panel)
        by_ticker = _by_ticker(panel, weights)
        metrics = _metrics(panel, weights, by_ticker, current)
    except Exception as exc:
        logger.warning("candidate %s could not be fitted", candidate_id, exc_info=True)
        return _unsolved(candidate_id, label, settings, _short_reason(exc))
    draft = _Solved(
        candidate_id=candidate_id,
        label=label,
        settings=settings,
        weights=by_ticker,
        metrics=metrics,
        method_breaches=_cardinality_breaches(metrics.n_positions, panel.config),
    )
    return draft.judged(_bounds_check(by_ticker, panel.config))


@tool_envelope
def compute_candidates(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    constraints: dict[str, Any] | None = None,
    current_weights: dict[str, float] | None = None,
    base_currency: str | None = None,
) -> ToolResult:
    """Compute the four candidate portfolios for ``universe`` as of ``asof``.

    A candidate's ``feasible`` here is the bounds-only verdict: its method solved,
    it holds no more names than the cardinality limit, and its weights meet the
    shared floor, cap and budget. It knows nothing of the ESG and K&E screens, the
    eligible universe or the loss ceilings; the bound agent tool re-judges every
    candidate with the risk gate's live check (``recheck_candidates``) and its
    ``feasible`` is that verdict.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        universe: yfinance tickers every method allocates across.
        constraints: The mapped ``MeanRiskConfig`` overrides ``optimize_portfolio``
            takes (objective, risk measure, risk aversion, bounds, budget,
            cardinality, tail beta, penalties, prior, ``annual_risk_free_rate``).
            Only ``mean_risk`` reads the objective, risk measure, aversion and
            penalties; every method shares the bounds, budget and prior.
        current_weights: The book's current weights, which turnover is measured
            from; ``None`` measures it from cash.
        base_currency: Currency prices are FX-converted to before returns are
            derived (``None`` keeps listing currency).

    Returns:
        ``ok`` with a ``CandidateSet`` as JSON: ``asof``, the fitted ``universe``,
        ``missing`` tickers, ``n_observations``, ``periods_per_year`` and the four
        ``candidates`` in id order, each with its label, settings, weights, metrics,
        bounds check and breaches. ``err`` on an empty universe, no priced assets,
        too little common history, or a weight cap or floor the priced assets cannot
        satisfy.
    """
    panel = load_returns_panel(
        session,
        asof,
        universe,
        constraints=constraints,
        base_currency=base_currency,
        action=_ACTION,
    )
    if isinstance(panel, str):
        return err(panel)
    current = dict(current_weights or {})
    candidate_set = CandidateSet(
        asof=coerce_date(asof),
        universe=tuple(str(column) for column in panel.returns.columns),
        missing=tuple(panel.missing),
        n_observations=int(panel.returns.shape[0]),
        periods_per_year=panel.periods,
        candidates=tuple(
            _candidate(candidate_id, panel, current) for candidate_id in CANDIDATE_IDS
        ),
    )
    return ok(candidate_set.model_dump(mode="json"))


__all__ = [
    "CARDINALITY_BREACH",
    "CHECK_FAILED",
    "CVAR_BETA",
    "SOLVER_FAILED",
    "apply_constraint_check",
    "compute_candidates",
    "constraint_check_from",
    "recheck_candidates",
]
