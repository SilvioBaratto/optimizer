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
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from fund.agents.toolsets import _bounds_args, _optimizer_constraints
from fund.config import DEFAULT_LOOKBACK_DAYS
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
        mean: Annualised expected return of the solved portfolio (fraction per year).
        standard_deviation: Its annualised volatility (fraction per year).
        sharpe_ratio: Its annualised reward-to-risk ratio.
        n_observations: Return rows the solve consumed.
    """

    mean: float
    standard_deviation: float
    sharpe_ratio: float
    n_observations: int


@dataclass(frozen=True)
class BacktestResult:
    """Backtest of the held weights over walk-forward test windows (or the full sample).

    The weights were fitted on every return up to ``asof``, so the evaluated windows
    always lie inside the optimizer's own training sample: this is a historical
    read-out of the held book, never a clean holdout test.

    Attributes:
        mean: Annualised mean return over the evaluated window (fraction per year).
        standard_deviation: Its annualised volatility (fraction per year).
        sharpe_ratio: Its annualised reward-to-risk ratio.
        max_drawdown: Worst peak-to-trough loss over the window (not annualised).
        n_observations: Return rows evaluated.
        n_folds: Walk-forward folds; ``0`` when the panel was too short for one.
        out_of_sample: Whether the evaluated windows were unseen by the weight fit;
            always ``False`` for a replay of the allocator's weights.
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
        base_currency: The currency the figures are scored in (the mandate's reporting
            currency), or ``None`` when prices were left in listing currency.
    """

    frontier: FrontierPoint | None
    frontier_error: str | None
    backtest: BacktestResult | None
    backtest_error: str | None
    base_currency: str | None = None


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
        unchecked_limits: Hard limits on the recorded constraint set that this verdict
            does not evaluate, so an ``ok`` is never read as "inside every limit".
    """

    passed: bool | None
    violations: tuple[dict[str, Any], ...]
    sector_exposures: tuple[tuple[str, float], ...] | None
    error: str | None
    unchecked_limits: tuple[str, ...] = ()


def compute_results(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    constraint_set: Mapping[str, Any] | None,
    weights: Mapping[str, float],
    base_currency: str | None = None,
    *,
    lookback_days: int | None = None,
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
        weights: The held allocation whose historical performance the backtest shows.
        base_currency: The mandate's reporting currency. Prices are FX-converted to it
            exactly as the agent's allocator tools did, so the replay reproduces the
            numbers the run was decided on; ``None`` keeps listing currency.
        lookback_days: The run's recorded walk-forward training window; ``None`` uses
            ``DEFAULT_LOOKBACK_DAYS``. The risk agent's backtest trains on this window,
            so the replay must too or it evaluates folds the risk gate never saw.

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
    # The held weights were optimised on every bar up to ``asof``, so the replayed
    # test blocks are inside their estimation sample (reported in-sample).
    train_size = DEFAULT_LOOKBACK_DAYS if lookback_days is None else lookback_days
    result, backtest_error = _backtest(
        backtest(
            session,
            asof,
            dict(weights),
            window={"train_size": train_size},
            base_currency=base_currency,
            fit_end=asof,
        )
    )
    return ResultsCompute(
        frontier=frontier,
        frontier_error=frontier_error,
        backtest=result,
        backtest_error=backtest_error,
        base_currency=base_currency,
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
        unchecked_limits=_unchecked_limits(constraint_set),
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


def _unchecked_limits(constraint_set: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Name the recorded hard limits the risk verdict does not evaluate.

    The verdict covers the weight bounds and budget, the K&E position cap, the ESG
    sector exclusions and the K&E instrument restrictions. Every other limit the
    profiler records is stored-only today, so each one that is actually set is listed
    here and an ``ok`` verdict is never read as "inside every limit":

    * the ``nu1/nu2/nu3`` drawdown ceilings, which need a backtested drawdown;
    * the ESG taxonomy / SFDR minimums and PAI flags;
    * the ``cardinality`` cap on the number of names.
    """
    if not constraint_set:
        return ()
    candidates = (
        _drawdown_ceilings(constraint_set),
        _esg_thresholds(_section(constraint_set, "esg")),
        _cardinality_cap(constraint_set),
    )
    return tuple(item for item in candidates if item is not None)


def _section(constraint_set: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """A nested constraint-set section, or an empty mapping when absent/malformed."""
    value = constraint_set.get(key)
    return value if isinstance(value, dict) else {}


def _is_number(value: object) -> TypeGuard[int | float]:
    """A real number; ``bool`` is excluded because it subclasses ``int``."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _drawdown_ceilings(constraint_set: Mapping[str, Any]) -> str | None:
    """The ``nu1/nu2/nu3`` ceilings as one line, or ``None`` when none is recorded."""
    tiers = [
        f"{name} {float(value):.0%}"
        for name in ("nu1", "nu2", "nu3")
        if _is_number(value := constraint_set.get(name))
    ]
    return f"drawdown ceilings ({' · '.join(tiers)})" if tiers else None


def _esg_thresholds(esg: Mapping[str, Any]) -> str | None:
    """The taxonomy / SFDR minimums and PAI flags that are set, else ``None``."""
    parts: list[str] = []
    for key, label in (("min_taxonomy", "taxonomy"), ("min_sfdr", "SFDR")):
        value = esg.get(key)
        if _is_number(value) and value > 0:
            parts.append(f"{label} ≥ {float(value):.0%}")
    flags = esg.get("pai_flags")
    if isinstance(flags, list | tuple) and flags:
        parts.append(f"PAI flags: {', '.join(str(flag) for flag in flags)}")
    return f"ESG thresholds ({' · '.join(parts)})" if parts else None


def _cardinality_cap(constraint_set: Mapping[str, Any]) -> str | None:
    """The recorded name-count cap, or ``None`` when unset."""
    cap = constraint_set.get("cardinality")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        return None
    return f"cardinality (at most {cap} names)"


def _annualised(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """A tool's annualised mean, volatility and Sharpe, as it reports them.

    The tools annualise their own metrics (252 daily bars), so the Deliberation and
    Performance views and the agent read the same per-year figures.
    """
    return (
        float(metrics.get("annualized_mean", 0.0)),
        float(metrics.get("annualized_standard_deviation", 0.0)),
        float(metrics.get("annualized_sharpe_ratio", 0.0)),
    )


def _frontier(envelope: Mapping[str, Any]) -> tuple[FrontierPoint | None, str | None]:
    """Map an ``optimize_portfolio`` envelope to a frontier point or error."""
    if not envelope.get("ok"):
        return None, str(envelope.get("error") or "optimize failed")
    data = envelope.get("data") or {}
    metrics = data.get("metrics") or {}
    mean, volatility, sharpe = _annualised(metrics)
    point = FrontierPoint(
        mean=mean,
        standard_deviation=volatility,
        sharpe_ratio=sharpe,
        n_observations=int(data.get("n_observations", 0)),
    )
    return point, None


def _backtest(envelope: Mapping[str, Any]) -> tuple[BacktestResult | None, str | None]:
    """Map a ``backtest`` envelope to a backtest result or error."""
    if not envelope.get("ok"):
        return None, str(envelope.get("error") or "backtest failed")
    data = envelope.get("data") or {}
    metrics = data.get("metrics") or {}
    mean, volatility, sharpe = _annualised(metrics)
    result = BacktestResult(
        mean=mean,
        standard_deviation=volatility,
        sharpe_ratio=sharpe,
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        n_observations=int(data.get("n_observations", 0)),
        n_folds=int(data.get("n_folds", 0)),
        # The tool's own flag only reports fold structure. The weights it evaluates
        # were fitted on this same panel, so no test window is a true holdout.
        out_of_sample=False,
        window_start=str(data.get("window_start", "")),
        window_end=str(data.get("window_end", "")),
    )
    return result, None
