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
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec

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

# Daily bars per year: converts between the per-period figures skfolio works in and
# the annual figures a client or an adviser reads.
PERIODS_PER_YEAR = 252

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


def thin_sample_error(prices: Any, returns: Any, *, action: str) -> str | None:
    """Explain why the common return sample is too thin to ``action``, if it is.

    The required depth is ``max(MIN_RETURN_OBSERVATIONS, 2 * n_assets)`` return
    rows, so a wide universe also needs proportionally more history to keep the
    sample covariance well conditioned. The message names the shortest-history
    asset because dropping it is the usual remedy.

    Args:
        prices: The wide price frame the returns were derived from.
        returns: The linear-returns frame after ``prices_to_returns`` (rows before
            the youngest asset's first quote already dropped).
        action: Verb phrase for the message, e.g. ``"optimize"``.

    Returns:
        An error message, or ``None`` when the sample is deep enough.
    """
    n_obs, n_assets = returns.shape
    required = max(MIN_RETURN_OBSERVATIONS, 2 * n_assets)
    if n_obs >= required:
        return None
    priced_days = prices.count()
    shortest = priced_days.idxmin()
    return (
        f"insufficient common history to {action}: {n_obs} return observations "
        f"shared by all {n_assets} assets, need at least {required}; "
        f"{shortest} has only {int(priced_days.min())} priced days, so drop it "
        "from the universe or wait for more history"
    )


def annualized_metrics(portfolio: Any) -> dict[str, float]:
    """Annualise a skfolio portfolio's per-bar mean, volatility and Sharpe ratio.

    skfolio reports these per observation (daily here), which reads as noise
    ("vol 0.01") next to the annual figures clients and thresholds are quoted in.
    Return and volatility scale with ``PERIODS_PER_YEAR`` and ``sqrt`` of it
    respectively, and so does the Sharpe ratio.

    Args:
        portfolio: A skfolio ``Portfolio`` (anything exposing ``mean``,
            ``standard_deviation`` and ``sharpe_ratio``).

    Returns:
        ``annualized_mean``, ``annualized_standard_deviation`` and
        ``annualized_sharpe_ratio``, in that order.
    """
    root = math.sqrt(PERIODS_PER_YEAR)
    return {
        "annualized_mean": float(portfolio.mean) * PERIODS_PER_YEAR,
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


def _recover_sessions(exc: Exception, arguments: Iterable[Any]) -> None:
    """Roll back every session reachable from a failed tool's ``arguments``.

    A session is an argument itself (the frozen ``fund.tools`` functions) or the
    ``session`` attribute of one (the bound agent closures take a run context).

    The tools of one run share a single session. A failed flush leaves it inactive
    ("rollback required") and a failed statement aborts a PostgreSQL transaction
    while the session still reports itself active; either way every later tool call
    and audit write would fail until someone rolls back. Other errors leave the
    transaction intact, so the work already flushed in it is kept.
    """
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.orm import Session

    # A tool that ran inside its own SAVEPOINT already handed the session back usable;
    # a full rollback would also discard the run's own uncommitted rows.
    if getattr(exc, "session_recovered", False):
        return
    sessions: dict[int, Session] = {}
    for argument in arguments:
        candidate = getattr(argument, "session", argument)
        if isinstance(candidate, Session):
            sessions[id(candidate)] = candidate

    for session in sessions.values():
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
    "MIN_RETURN_OBSERVATIONS",
    "PERIODS_PER_YEAR",
    "ToolResult",
    "annualized_metrics",
    "coerce_date",
    "err",
    "ok",
    "session_scope",
    "summarize_frame",
    "thin_sample_error",
    "tool_envelope",
]
