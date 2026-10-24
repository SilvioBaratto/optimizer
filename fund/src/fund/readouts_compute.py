"""Worker-thread recompute service for the Deliberation drill-downs.

Re-derives the read-outs that are *not* persisted — the frontier point and the
backtest (later, the structured risk verdict) — by replaying the fund's frozen
compute tools at the run's ``asof``, under the recorded ``ConstraintSet`` translated
by the same helpers the live toolsets use (so the replay cannot drift from the run's
own allocator and risk gate). It lives **outside** ``fund.tui`` on purpose: it
imports ``fund.tools`` (which import ``optimizer`` at their module top), so the
Deliberation view imports *this* module lazily, inside its worker body only — keeping
``import fund.tui.app`` agent-stack-light. It stays agent-stack-free itself (no
``deepagents`` / ``langgraph``). Every result is plain numbers in a frozen dataclass;
the tools never raise across their envelope, so a failure degrades to an error string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from fund.agents.toolsets import _bounds_args, _optimizer_constraints
from fund.schemas import ConstraintSet
from fund.tools.mandate import constraint_set_breaches, held_tickers
from fund.tools.optimize import optimize_portfolio
from fund.tools.risk import backtest, risk_check

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping, Sequence

    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class FrontierPoint:
    """The allocator's solve under the recorded ConstraintSet, as a single point.

    Never a curve: the objective, risk measure and position cap are the ones the live
    ``optimize_portfolio`` applied to the recorded weights.

    Attributes:
        mean: Expected return of the solved portfolio.
        standard_deviation: Its volatility.
        sharpe_ratio: Its reward-to-risk ratio.
        n_observations: Return rows the solve consumed.
    """

    mean: float
    standard_deviation: float
    sharpe_ratio: float
    n_observations: int


@dataclass(frozen=True)
class BacktestResult:
    """Walk-forward (or degraded in-sample) backtest of the held weights.

    Attributes:
        mean: Mean return over the evaluated window.
        standard_deviation: Its volatility.
        sharpe_ratio: Its reward-to-risk ratio.
        max_drawdown: Worst peak-to-trough loss over the window.
        n_observations: Return rows evaluated.
        n_folds: Walk-forward folds; ``0`` when the panel was too short for one.
        out_of_sample: Whether folds were out-of-sample (``False`` = in-sample).
        window_start: ISO date of the evaluated window's first row.
        window_end: ISO date of its last row.
    """

    mean: float
    standard_deviation: float
    sharpe_ratio: float
    max_drawdown: float
    n_observations: int
    n_folds: int
    out_of_sample: bool
    window_start: str
    window_end: str


@dataclass(frozen=True)
class ResultsCompute:
    """The recomputed *results* read-out: a frontier point and a backtest.

    Each slot is either its value or ``None`` with a sibling ``*_error`` string when the
    underlying tool returned ``ok=False`` (empty universe, no priced assets, …).

    Attributes:
        frontier: The frontier point, or ``None`` on tool failure.
        frontier_error: The failure message when ``frontier`` is ``None``.
        backtest: The backtest result, or ``None`` on tool failure.
        backtest_error: The failure message when ``backtest`` is ``None``.
    """

    frontier: FrontierPoint | None
    frontier_error: str | None
    backtest: BacktestResult | None
    backtest_error: str | None


@dataclass(frozen=True)
class RiskCompute:
    """The recomputed *risk* read-out: the live ``risk_check`` verdict + exposures.

    The dot/label tier is not derived here — the pure severity map in
    :mod:`fund.tui.readouts` maps :attr:`violations` to ``ok`` / ``warn`` / ``breach``
    at render time. This module only runs the check and rolls weights up per sector.

    Attributes:
        passed: ``risk_check``'s pass flag, or ``None`` when the check itself failed.
        violations: The raw violation dicts (``type``/``asset``/``value``), maybe empty.
        sector_exposures: ``(sector, total weight)`` pairs sorted by weight, or ``None``
            when no sector mapping was available (``unavailable``, not a crash).
        error: The failure message when the check could not run, else ``None``.
    """

    passed: bool | None
    violations: tuple[dict[str, Any], ...]
    sector_exposures: tuple[tuple[str, float], ...] | None
    error: str | None


def compute_results(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    constraint_set: Mapping[str, Any] | None,
    weights: Mapping[str, float],
    base_currency: str | None = None,
) -> ResultsCompute:
    """Recompute the results read-out by replaying the compute tools at ``asof``.

    Args:
        session: An open read-only DB session for the price reads.
        asof: The run's as-of date (the recompute point).
        universe: Names to optimise over — the caller passes the selected names (the
            recorded weights keys) so the frontier reflects what was held.
        constraint_set: The recorded ``ConstraintSet.model_dump``; its mapped objective,
            risk measure, beta, bounds and position cap reach ``optimize_portfolio``
            exactly as in a live run. ``None`` solves under the tool defaults.
        weights: The held allocation whose forward performance the backtest evaluates.
        base_currency: The mandate's reporting currency. Prices are FX-converted to it
            exactly as the agent's allocator tools did, so the replay reproduces the
            numbers the run was decided on; ``None`` keeps listing currency.

    Returns:
        A :class:`ResultsCompute`. A failing tool yields a ``None`` slot plus its
        ``*_error`` message rather than raising; a recorded constraint set that no
        longer validates leaves the frontier unavailable (the backtest does not use
        it and still runs).
    """
    cs, cs_error = _rehydrate(constraint_set)
    if cs_error is not None:
        frontier, frontier_error = None, cs_error
    else:
        constraints = _optimizer_constraints(cs) if cs is not None else None
        frontier, frontier_error = _frontier(
            optimize_portfolio(
                session,
                asof,
                universe,
                constraints=constraints,
                base_currency=base_currency,
            )
        )
    result, backtest_error = _backtest(
        backtest(session, asof, dict(weights), base_currency=base_currency)
    )
    return ResultsCompute(
        frontier=frontier,
        frontier_error=frontier_error,
        backtest=result,
        backtest_error=backtest_error,
    )


def compute_risk(
    weights: Mapping[str, float],
    constraint_set: Mapping[str, Any] | None,
    sector_mapping: Mapping[str, str] | None = None,
    *,
    session: Session,
) -> RiskCompute:
    """Recompute the risk read-out: the live ``risk_check`` verdict + sector exposures.

    Applies what the live gate applies: the bounds tightened to the K&E position cap
    and, on every held name, the ESG sector exclusions and K&E instrument restrictions
    (an instrument it cannot classify fails closed).

    Args:
        weights: The held allocation to check and roll up.
        constraint_set: The recorded ``ConstraintSet.model_dump``.
        sector_mapping: ``ticker -> sector``; ``None``/empty yields ``unavailable``.
        session: An open read-only DB session for the instrument screen; untouched
            when the constraint set carries no ESG or K&E restriction.

    Returns:
        A :class:`RiskCompute`. When ``risk_check`` returns ``ok=False``, the recorded
        constraint set no longer validates, or the instrument screen cannot read the
        database, the verdict is ``None`` with an ``error`` string (never a pass),
        but the sector roll-up is still attempted.
    """
    exposures = _sector_exposures(weights, sector_mapping)
    cs, cs_error = _rehydrate(constraint_set)
    if cs_error is not None:
        return RiskCompute(
            passed=None, violations=(), sector_exposures=exposures, error=cs_error
        )
    bounds = _bounds_args(cs) if cs is not None else None
    envelope = risk_check(dict(weights), constraints=bounds)
    if not envelope.get("ok"):
        return RiskCompute(
            passed=None,
            violations=(),
            sector_exposures=exposures,
            error=str(envelope.get("error") or "risk check failed"),
        )
    data = envelope.get("data") or {}
    passed = bool(data.get("passed", False))
    violations: list[dict[str, Any]] = list(data.get("violations") or ())
    if cs is not None:
        try:
            breaches = constraint_set_breaches(session, cs, held_tickers(weights))
        except SQLAlchemyError as exc:
            return RiskCompute(
                passed=None,
                violations=(),
                sector_exposures=exposures,
                error=f"mandate screen failed: {exc.__class__.__name__}",
            )
        if breaches:
            passed = False
            violations.extend(breaches)
    return RiskCompute(
        passed=passed,
        violations=tuple(violations),
        sector_exposures=exposures,
        error=None,
    )


def _sector_exposures(
    weights: Mapping[str, float], sector_mapping: Mapping[str, str] | None
) -> tuple[tuple[str, float], ...] | None:
    """Roll weights up per sector, sorted by weight; ``None`` when nothing rolls up.

    ``None`` both without a mapping and when every weight is empty/non-numeric, so the
    renderer shows one ``unavailable`` line rather than a bare header with no rows.
    """
    if not sector_mapping:
        return None
    totals: dict[str, float] = {}
    for name, weight in weights.items():
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        sector = sector_mapping.get(name, "unmapped")
        totals[sector] = totals.get(sector, 0.0) + value
    return tuple(sorted(totals.items(), key=lambda item: item[1], reverse=True)) or None


def _rehydrate(
    constraint_set: Mapping[str, Any] | None,
) -> tuple[ConstraintSet | None, str | None]:
    """The recorded ``ConstraintSet`` and no error, ``(None, None)`` if none was
    recorded, or ``(None, message)`` when the dump no longer validates.

    An unreadable dump must surface as an error: falling back to default bounds would
    show a verdict for a mandate the client never had.
    """
    if not constraint_set:
        return None, None
    try:
        return ConstraintSet.model_validate(dict(constraint_set)), None
    except ValidationError as exc:
        return None, (
            f"recorded constraint set is unreadable ({exc.error_count()} invalid "
            f"field(s))"
        )


def _frontier(envelope: Mapping[str, Any]) -> tuple[FrontierPoint | None, str | None]:
    """Map an ``optimize_portfolio`` envelope to a frontier point or error."""
    if not envelope.get("ok"):
        return None, str(envelope.get("error") or "optimize failed")
    data = envelope.get("data") or {}
    metrics = data.get("metrics") or {}
    point = FrontierPoint(
        mean=float(metrics.get("mean", 0.0)),
        standard_deviation=float(metrics.get("standard_deviation", 0.0)),
        sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
        n_observations=int(data.get("n_observations", 0)),
    )
    return point, None


def _backtest(envelope: Mapping[str, Any]) -> tuple[BacktestResult | None, str | None]:
    """Map a ``backtest`` envelope to a backtest result or error."""
    if not envelope.get("ok"):
        return None, str(envelope.get("error") or "backtest failed")
    data = envelope.get("data") or {}
    metrics = data.get("metrics") or {}
    result = BacktestResult(
        mean=float(metrics.get("mean", 0.0)),
        standard_deviation=float(metrics.get("standard_deviation", 0.0)),
        sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        n_observations=int(data.get("n_observations", 0)),
        n_folds=int(data.get("n_folds", 0)),
        out_of_sample=bool(data.get("out_of_sample", False)),
        window_start=str(data.get("window_start", "")),
        window_end=str(data.get("window_end", "")),
    )
    return result, None
