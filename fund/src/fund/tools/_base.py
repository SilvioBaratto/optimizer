"""Tool envelope: the ``{ok, data}`` / ``{ok: false, error}`` contract shared by
every ``fund`` tool.

Two hard rules shape this module:

* A tool NEVER raises across its own boundary — a caught exception becomes
  ``{"ok": False, "error": <msg>}`` so a mis-called tool degrades the agent's
  turn instead of crashing the LangGraph run.
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
from collections.abc import Callable, Generator, Mapping
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


def tool_envelope(fn: Callable[P, Any]) -> Callable[P, ToolResult]:
    """Wrap ``fn`` so it always returns a ToolResult, never raises.

    * A raised exception is logged and converted to ``err("<Type>: <msg>")``.
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
            return err(f"{type(exc).__name__}: {exc}")
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

    Returns row/column counts, column labels, per-column NaN counts, overall
    non-NaN coverage, and (for a DatetimeIndex) the index span — never the frame
    itself, so a large price/return matrix never reaches the model verbatim.

    Args:
        frame: The DataFrame to summarise.
        name: Label embedded in the returned dict to identify the frame at the
            call site.

    Returns:
        A dict with keys ``name``, ``rows``, ``cols``, ``columns``, optionally
        ``index_start``/``index_end``, ``na_counts``, and ``coverage``.
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

    if isinstance(frame.index, pd.DatetimeIndex):
        summary["index_start"] = str(frame.index.min())
        summary["index_end"] = str(frame.index.max())

    na = frame.isna().sum()
    summary["na_counts"] = {str(col): int(na[col]) for col in frame.columns}
    summary["coverage"] = float(1.0 - frame.isna().to_numpy().mean())
    return summary


__all__ = [
    "MIN_RETURN_OBSERVATIONS",
    "ToolResult",
    "coerce_date",
    "err",
    "ok",
    "session_scope",
    "summarize_frame",
    "thin_sample_error",
    "tool_envelope",
]
