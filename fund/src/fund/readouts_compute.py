"""Worker-thread recompute service for the Deliberation drill-downs.

Re-derives the read-outs that are *not* persisted — the frontier point and the
backtest (later, the structured risk verdict) — by replaying the fund's frozen
compute tools at the run's ``asof``. It lives **outside** ``fund.tui`` on purpose: it
imports ``fund.tools`` (which import ``optimizer`` at their module top), so the
Deliberation view imports *this* module lazily, inside its worker body only — keeping
``import fund.tui.app`` agent-stack-light. It stays agent-stack-free itself (no
``deepagents`` / ``langgraph``). Every result is plain numbers in a frozen dataclass;
the tools never raise across their envelope, so a failure degrades to an error string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fund.tools.optimize import optimize_portfolio
from fund.tools.risk import backtest, risk_check

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping, Sequence

    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class FrontierPoint:
    """The allocator's min-variance QP solution as a single point (never a curve).

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
) -> ResultsCompute:
    """Recompute the results read-out by replaying the compute tools at ``asof``.

    Args:
        session: An open read-only DB session for the price reads.
        asof: The run's as-of date (the recompute point).
        universe: Names to optimise over — the caller passes the selected names (the
            recorded weights keys) so the frontier reflects what was held.
        constraint_set: The recorded ``ConstraintSet.model_dump``; only its ``bounds``
            (min/max weights, budget) reach ``optimize_portfolio``.
        weights: The held allocation whose forward performance the backtest evaluates.

    Returns:
        A :class:`ResultsCompute`. A failing tool yields a ``None`` slot plus its
        ``*_error`` message rather than raising.
    """
    bounds = _bounds(constraint_set)
    frontier, frontier_error = _frontier(
        optimize_portfolio(session, asof, universe, constraints=bounds)
    )
    result, backtest_error = _backtest(backtest(session, asof, dict(weights)))
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
) -> RiskCompute:
    """Recompute the risk read-out: the live ``risk_check`` verdict + sector exposures.

    Args:
        weights: The held allocation to check and roll up.
        constraint_set: The recorded ``ConstraintSet.model_dump``; only its ``bounds``
            reach ``risk_check``.
        sector_mapping: ``ticker -> sector``; ``None``/empty yields ``unavailable``.

    Returns:
        A :class:`RiskCompute`. When ``risk_check`` returns ``ok=False`` the verdict is
        ``None`` with an ``error`` string, but the sector roll-up is still attempted.
    """
    bounds = _bounds(constraint_set)
    exposures = _sector_exposures(weights, sector_mapping)
    envelope = risk_check(dict(weights), constraints=bounds)
    if not envelope.get("ok"):
        return RiskCompute(
            passed=None,
            violations=(),
            sector_exposures=exposures,
            error=str(envelope.get("error") or "risk check failed"),
        )
    data = envelope.get("data") or {}
    return RiskCompute(
        passed=bool(data.get("passed", False)),
        violations=tuple(data.get("violations") or ()),
        sector_exposures=exposures,
        error=None,
    )


def _sector_exposures(
    weights: Mapping[str, float], sector_mapping: Mapping[str, str] | None
) -> tuple[tuple[str, float], ...] | None:
    """Roll weights up per sector, sorted by weight; ``None`` without a mapping."""
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
    return tuple(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def _bounds(constraint_set: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Extract the solver bounds the frozen tools consume, or ``None`` for defaults."""
    if not constraint_set:
        return None
    bounds = constraint_set.get("bounds")
    return dict(bounds) if isinstance(bounds, dict) else None


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
