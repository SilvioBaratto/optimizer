"""Tool envelope: the ``{ok,data}`` / ``{ok:false,error}`` contract shared by
every Fase-3 ``fund`` tool.

Two hard rules from SPEC Fase 3 shape this module:

* A tool NEVER raises across its own boundary — a caught exception becomes
  ``{"ok": False, "error": <msg>}`` so a mis-called tool degrades the agent's
  turn instead of crashing the LangGraph run.
* A tool never hands a raw DataFrame back to the model — big frames are collapsed
  to a JSON-serialisable shape/coverage summary in-environment
  (:func:`summarize_frame`).

:func:`tool_envelope` wraps a plain callable into that contract;
:func:`session_scope` hands a tool a sync ``portopt_db`` session (D1) it does not
own. The module imports nothing from ``optimizer`` — that dependency arrives only
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

P = ParamSpec("P")


def coerce_date(asof: dt.date | str) -> dt.date:
    """Normalise ``asof`` to a ``date``; a ``datetime`` collapses to its date.

    Shared by every tool that takes an ``asof`` upper bound so the date-coercion
    rule (accept ``date`` / ``datetime`` / ISO ``YYYY-MM-DD`` string) lives once.
    """
    if isinstance(asof, dt.datetime):
        return asof.date()
    if isinstance(asof, dt.date):
        return asof
    return dt.date.fromisoformat(asof)


def ok(data: Any) -> ToolResult:
    """Success envelope carrying ``data`` (must be JSON-serialisable)."""
    return {"ok": True, "data": data}


def err(error: str) -> ToolResult:
    """Failure envelope carrying a human-readable ``error`` message."""
    return {"ok": False, "error": error}


def tool_envelope(fn: Callable[P, Any]) -> Callable[P, ToolResult]:
    """Wrap ``fn`` so it always returns a :data:`ToolResult`, never raises.

    * A raised exception is logged and converted to ``err("<Type>: <msg>")``.
    * A return value that is already an envelope (a mapping with an ``"ok"`` key)
      passes through as a plain ``dict``.
    * Any other return value is wrapped with :func:`ok`.
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
    own short-lived session (D1: sessions are sync and caller-owned). Tests inject
    a session directly and skip this.
    """
    with manager.get_session() as session:
        yield session


def summarize_frame(frame: Any, *, name: str = "frame") -> dict[str, Any]:
    """Collapse a pandas DataFrame to a JSON-serialisable shape/coverage summary.

    Returns row/column counts, column labels, per-column NaN counts, overall
    non-NaN coverage, and (for a DatetimeIndex) the index span — never the frame
    itself, so a large price/return matrix never reaches the model verbatim.
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
    "ToolResult",
    "coerce_date",
    "err",
    "ok",
    "session_scope",
    "summarize_frame",
    "tool_envelope",
]
