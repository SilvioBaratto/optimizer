"""Tool envelope: the ``{ok, data}`` / ``{ok: false, error}`` contract shared by
every ``fund`` tool.

Two hard rules shape this module:

* A tool NEVER raises across its own boundary — a caught exception becomes
  ``{"ok": False, "error": <msg>}`` so a mis-called tool degrades the agent's
  turn instead of crashing the LangGraph run. A database failure also rolls the
  tool's session back (every bound tool shares one) and reaches the model without
  its SQL text or bound parameters.
* A tool never hands a raw DataFrame back to the model — big frames are collapsed
  to a JSON-serialisable shape/coverage summary in-environment
  (``summarize_frame``).

``tool_envelope`` wraps a plain callable into that contract;
``session_scope`` hands a tool a sync ``portopt_db`` session it does not own.
The module imports nothing from ``optimizer`` — that dependency arrives only
inside the individual tool modules that need it.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging
import math
import threading
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec

from fund.periods import (
    DEFAULT_PERIODS_PER_YEAR,
    MAX_PERIODS_PER_YEAR,
    MIN_PERIODS_PER_YEAR,
    periods_per_year,
)

if TYPE_CHECKING:
    from portopt_db.engine import DatabaseManager
    from sqlalchemy.orm import Session

logger = logging.getLogger("fund.tools")

# JSON-serialisable envelope handed back to the agent runtime.
ToolResult = dict[str, Any]

# Floor on the number of return rows shared by every asset in a fit (about one
# trading month of daily bars, the same floor ``estimate_moments`` holds each asset
# to). The common sample starts at the youngest listing's first quote, so a single
# recent ticker can shrink a multi-year panel to a handful of rows; a covariance
# estimated on that is noise that would still come back as ``ok``. Wide universes are
# held to ``2 * n_assets`` instead, see ``thin_sample_error``.
MIN_RETURN_OBSERVATIONS = 20

# An asset is too young for a panel when its history is under this share of the
# panel's median depth. The common sample starts at the youngest asset's first quote,
# so a listing only slightly past the absolute floor would still cut a multi-year
# panel to a few weeks for every other asset.
MIN_DEPTH_SHARE = 0.5

P = ParamSpec("P")


def coerce_date(asof: dt.date | str) -> dt.date:
    """Normalise ``asof`` to a date; a datetime collapses to its date.

    Shared by every tool that takes an ``asof`` upper bound so the date-coercion
    rule (accept date, datetime, or ISO ``YYYY-MM-DD`` string) lives once.

    Args:
        asof: Upper-bound date for a query; accepts a date, datetime, or an
            ISO 8601 ``YYYY-MM-DD`` string.

    Returns:
        The calendar date, with any time component discarded.
    """
    if isinstance(asof, dt.datetime):
        return asof.date()
    if isinstance(asof, dt.date):
        return asof
    return dt.date.fromisoformat(asof)


def depth_floor(depths: Any) -> int:
    """Return rows an asset needs to stay in a panel with these ``depths``.

    Args:
        depths: Each asset's own number of return observations, as a pandas Series.

    Returns:
        The larger of ``MIN_RETURN_OBSERVATIONS`` and ``MIN_DEPTH_SHARE`` of the
        median depth, rounded up. The one rule ``estimate_moments`` drops by,
        ``universe_filter`` screens by and ``thin_sample_error`` refuses by.
    """
    median = float(depths.median()) if len(depths) else 0.0
    return max(MIN_RETURN_OBSERVATIONS, math.ceil(MIN_DEPTH_SHARE * median))


def thin_sample_error(prices: Any, returns: Any, *, action: str) -> str | None:
    """Explain why the common return sample is too thin to ``action``, if it is.

    The required depth is the largest of ``MIN_RETURN_OBSERVATIONS``, ``2 *
    n_assets`` (a wide universe needs proportionally more history to keep the sample
    covariance well conditioned) and ``MIN_DEPTH_SHARE`` of the assets' median own
    history (so one recent listing cannot swap a multi-year sample for a few weeks
    without the caller noticing). The message names the shortest-history asset
    because dropping it is the usual remedy.

    Args:
        prices: The wide price frame the returns were derived from.
        returns: The linear-returns frame after ``prices_to_returns`` (rows before
            the youngest asset's first quote already dropped).
        action: Verb phrase for the message, e.g. ``"optimize"``.

    Returns:
        An error message, or ``None`` when the sample is deep enough.
    """
    n_obs, n_assets = returns.shape
    priced_days = prices.count()
    required = max(
        MIN_RETURN_OBSERVATIONS,
        2 * n_assets,
        depth_floor(priced_days - 1),
    )
    if n_obs >= required:
        return None
    shortest = priced_days.idxmin()
    return (
        f"insufficient common history to {action}: {n_obs} return observations "
        f"shared by all {n_assets} assets, need at least {required}; "
        f"{shortest} has only {int(priced_days.min())} priced days, so drop it "
        "from the universe or wait for more history"
    )


def dropped_by_returns(prices: Any, returns: Any) -> list[str]:
    """Name the ``prices`` columns that ``prices_to_returns`` left out of ``returns``.

    A column with no usable price at all (an FX conversion can blank one) vanishes
    from the returns frame without a trace, so a caller that counts or weights
    assets from ``prices`` would otherwise treat it as still present.

    Args:
        prices: The wide price frame the returns were derived from.
        returns: The linear-returns frame after ``prices_to_returns``.

    Returns:
        The labels missing from ``returns``, in the order ``prices`` holds them.
    """
    kept = set(returns.columns)
    return [str(column) for column in prices.columns if column not in kept]


def per_period_rate(annual_rate: float, periods: int) -> float:
    """Compound an annual rate down to the return frequency.

    A client quotes the risk-free rate per year while the optimiser compares it with
    per-bar returns, so the rate is converted with the same ``periods`` the returns
    are annualised by.

    Args:
        annual_rate: The annual rate as a fraction (``0.04`` is 4%).
        periods: Bars per year, normally from :func:`periods_per_year`.

    Returns:
        The per-bar rate whose ``periods``-fold compounding is ``annual_rate``.

    Raises:
        ValueError: ``annual_rate`` is -100% or lower, which has no real root.
    """
    if annual_rate <= -1.0:
        raise ValueError(f"annual rate {annual_rate:g} is not above -100%")
    return float((1.0 + annual_rate) ** (1.0 / periods) - 1.0)


def annualized_metrics(portfolio: Any, periods: int) -> dict[str, float]:
    """Annualise a skfolio portfolio's per-bar mean, volatility and Sharpe ratio.

    skfolio reports these per observation (daily here), which reads as noise
    ("vol 0.01") next to the annual figures clients and thresholds are quoted in.
    Return scales with ``periods`` and volatility with its square root, and so does
    the Sharpe ratio.

    Args:
        portfolio: A skfolio ``Portfolio`` (anything exposing ``mean``,
            ``standard_deviation`` and ``sharpe_ratio``).
        periods: Bars per year, normally from :func:`periods_per_year` on the
            portfolio's return index.

    Returns:
        ``annualized_mean``, ``annualized_standard_deviation`` and
        ``annualized_sharpe_ratio``, in that order.
    """
    root = math.sqrt(periods)
    return {
        "annualized_mean": float(portfolio.mean) * periods,
        "annualized_standard_deviation": float(portfolio.standard_deviation) * root,
        "annualized_sharpe_ratio": float(portfolio.sharpe_ratio) * root,
    }


def ok(data: Any) -> ToolResult:
    """Return a success envelope wrapping ``data``.

    Args:
        data: The payload; must be JSON-serialisable so the agent runtime can
            consume it without further conversion.

    Returns:
        A mapping ``{"ok": True, "data": data}``.
    """
    return {"ok": True, "data": data}


def err(error: str) -> ToolResult:
    """Return a failure envelope wrapping ``error``.

    Args:
        error: Human-readable description of what went wrong; shown verbatim
            to the agent.

    Returns:
        A mapping ``{"ok": False, "error": error}``.
    """
    return {"ok": False, "error": error}


def _database_failure_message(exc: Exception) -> str | None:
    """A model-safe message for a SQLAlchemy ``exc``, or ``None`` for any other error.

    A SQLAlchemy message embeds the statement and its bound parameters, which would
    hand schema and stored values to the model; the full text stays in the log.
    """
    from sqlalchemy.exc import SQLAlchemyError

    if isinstance(exc, SQLAlchemyError):
        return f"{type(exc).__name__}: database error (details are in the server log)"
    return None


# Sessions with a SAVEPOINT owner, by identity, mapped to how many owners are open.
# Keyed outside the session so any session-like object can be declared.
_SAVEPOINT_OWNERS: dict[int, int] = {}
_SAVEPOINT_OWNERS_LOCK = threading.Lock()


@contextmanager
def savepoint_owner(session: Session) -> Generator[None, None, None]:
    """Declare that the caller owns a SAVEPOINT on ``session`` and will recover it.

    A frozen tool called inside the block carries its own ``tool_envelope``, whose
    default recovery is a rollback of the session's root transaction. That would
    discard everything the run holds uncommitted (its ``agent_runs`` row, every
    decision, the audit row) before the owner could roll its SAVEPOINT back. While
    the block is open the envelope leaves the session alone and the owner, which
    sees the failure as an ``ok: false`` result or an exception, rolls back to its
    SAVEPOINT instead.

    Args:
        session: The session the caller opened a SAVEPOINT on. Owners may nest.

    Yields:
        Nothing; the declaration is lifted on exit even when the body raises.
    """
    key = id(session)
    with _SAVEPOINT_OWNERS_LOCK:
        _SAVEPOINT_OWNERS[key] = _SAVEPOINT_OWNERS.get(key, 0) + 1
    try:
        yield
    finally:
        with _SAVEPOINT_OWNERS_LOCK:
            remaining = _SAVEPOINT_OWNERS[key] - 1
            if remaining:
                _SAVEPOINT_OWNERS[key] = remaining
            else:
                del _SAVEPOINT_OWNERS[key]


def _has_savepoint_owner(session: Session) -> bool:
    with _SAVEPOINT_OWNERS_LOCK:
        owned = id(session) in _SAVEPOINT_OWNERS
    return owned and session.in_nested_transaction()


def _recover_sessions(exc: Exception, arguments: Iterable[Any]) -> None:
    """Roll back every session reachable from a failed tool's ``arguments``.

    A session is an argument itself (the frozen ``fund.tools`` functions) or the
    ``session`` attribute of one (the bound agent closures take a run context).

    The tools of one run share a single session. A failed flush leaves it inactive
    ("rollback required") and a failed statement aborts a PostgreSQL transaction
    while the session still reports itself active; either way every later tool call
    and audit write would fail until someone rolls back. Other errors leave the
    transaction intact, so the work already flushed in it is kept. A session whose
    SAVEPOINT has an owner (``savepoint_owner``) is left to it.
    """
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.orm import Session

    # A tool that ran inside its own SAVEPOINT already handed the session back usable,
    # and a full rollback would also discard the run's own uncommitted rows.
    if getattr(exc, "session_recovered", False):
        return
    sessions: dict[int, Session] = {}
    for argument in arguments:
        candidate = getattr(argument, "session", argument)
        if isinstance(candidate, Session):
            sessions[id(candidate)] = candidate

    for session in sessions.values():
        if _has_savepoint_owner(session):
            continue
        if isinstance(exc, DBAPIError) or not session.is_active:
            try:
                session.rollback()
            except Exception:  # recovery must not break the envelope either
                logger.exception("rollback after a failed tool did not complete")


def tool_envelope(fn: Callable[P, Any]) -> Callable[P, ToolResult]:
    """Wrap ``fn`` so it always returns a ToolResult, never raises.

    * A raised exception is logged and converted to ``err("<Type>: <msg>")``. A
      SQLAlchemy error is reported without its SQL text, and a session it left
      unusable is rolled back, whether passed as an argument or held in the
      ``session`` attribute of one (a run context).
    * A return value that is already an envelope (a mapping with an ``"ok"`` key)
      passes through as a plain ``dict``.
    * Any other return value is wrapped with ``ok``.

    Args:
        fn: The tool callable to wrap.

    Returns:
        A wrapper with the same signature as ``fn`` that always returns
        a ToolResult.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # the envelope must never raise across its boundary
            logger.exception("tool %s failed", fn.__name__)
            _recover_sessions(exc, (*args, *kwargs.values()))
            return err(_database_failure_message(exc) or f"{type(exc).__name__}: {exc}")
        if isinstance(result, Mapping) and "ok" in result:
            return dict(result)
        return ok(result)

    return wrapper


@contextmanager
def session_scope(manager: DatabaseManager) -> Generator[Session, None, None]:
    """Yield a sync session from ``manager``, always closed on exit.

    Thin passthrough over ``DatabaseManager.get_session`` so a tool can open its
    own short-lived session without owning the lifecycle. Tests inject a session
    directly and skip this.

    Args:
        manager: Source of sync SQLAlchemy sessions.

    Yields:
        An open ``Session``; closed and returned to the pool on exit.
    """
    with manager.get_session() as session:
        yield session


def summarize_frame(frame: Any, *, name: str = "frame") -> dict[str, Any]:
    """Collapse a pandas DataFrame to a JSON-serialisable shape/coverage summary.

    Returns row/column counts, column labels, the index span (for a
    DatetimeIndex) and, per column, how many observations it holds and, with a
    DatetimeIndex, the dates of its first and last one — never the frame itself,
    so a large price/return matrix never reaches the model verbatim.

    The per-column figures are deliberately not NaN counts on the shared index:
    the panels are outer-joined, so a monthly series next to a daily one, or two
    exchange calendars, leave holes that are alignment and not missing data. A
    column's own span against ``index_start``/``index_end`` shows a series that
    ends early or starts late.

    Args:
        frame: The DataFrame to summarise.
        name: Label embedded in the returned dict to identify the frame at the
            call site.

    Returns:
        A dict with keys ``name``, ``rows``, ``cols``, ``columns``, optionally
        ``index_start``/``index_end``, and ``series`` mapping each column to its
        ``observations`` (non-null count) plus ``first``/``last`` when the index
        is dates and the column holds any observation.
    """
    n_rows, n_cols = frame.shape
    summary: dict[str, Any] = {
        "name": name,
        "rows": int(n_rows),
        "cols": int(n_cols),
        "columns": [str(c) for c in frame.columns],
    }
    if n_rows == 0 or n_cols == 0:
        return summary

    import pandas as pd

    dated = isinstance(frame.index, pd.DatetimeIndex)
    if dated:
        summary["index_start"] = str(frame.index.min())
        summary["index_end"] = str(frame.index.max())

    series: dict[str, dict[str, Any]] = {}
    for position, label in enumerate(frame.columns):
        column = frame.iloc[:, position]
        entry: dict[str, Any] = {"observations": int(column.notna().sum())}
        first = column.first_valid_index()
        if dated and first is not None:
            entry["first"] = str(first)
            entry["last"] = str(column.last_valid_index())
        series[str(label)] = entry
    summary["series"] = series
    return summary


__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "MAX_PERIODS_PER_YEAR",
    "MIN_DEPTH_SHARE",
    "MIN_PERIODS_PER_YEAR",
    "MIN_RETURN_OBSERVATIONS",
    "ToolResult",
    "annualized_metrics",
    "coerce_date",
    "depth_floor",
    "dropped_by_returns",
    "err",
    "ok",
    "per_period_rate",
    "periods_per_year",
    "savepoint_owner",
    "session_scope",
    "summarize_frame",
    "thin_sample_error",
    "tool_envelope",
]
