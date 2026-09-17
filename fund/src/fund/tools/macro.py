"""T3.5 — ``get_macro_series``: pull macro time-series for the regime/views agents.

Reads FRED observations out of the shared ``portopt_db`` layer via
:class:`~portopt_db.repositories.macro.macro_regime_repository.MacroRegimeRepository`
and hands the agent a JSON-serialisable shape/coverage **summary** — never the
raw observation matrix (SPEC Fase 3). The tool is a pure function of ``(session,
names, asof)``: same seeded data + same args ⇒ identical output.

Contract (via :func:`fund.tools._base.tool_envelope`):

* empty ``names`` ⇒ ``{ok: false, error}``;
* a series with no stored observations (or only null values) as of ``asof`` ⇒
  **flagged** in ``data["missing"]``, never raised;
* every other failure (e.g. a malformed ``asof``) is caught by the envelope and
  returned as ``{ok: false, error}``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
from portopt_db.repositories.macro.macro_regime_repository import (
    MacroRegimeRepository,
)

from fund.tools._base import (
    ToolResult,
    coerce_date,
    err,
    ok,
    summarize_frame,
    tool_envelope,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def load_macro_frame(
    session: Session,
    names: Sequence[str],
    asof: dt.date | str,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a wide macro-series frame for ``names`` as of ``asof``, plus missing.

    Shared loader behind :func:`get_macro_series`: one column per *populated*
    series (in requested order) on a sorted ``DatetimeIndex``; ``missing`` lists
    requested series with no stored observation or only null values as of
    ``asof``. The caller validates non-empty ``names``.

    Args:
        session: A sync ``portopt_db`` session (D1); the loader does not own it.
        names: FRED series ids, in the order the panel columns should follow.
        asof: Inclusive upper bound; observations strictly after it are excluded.

    Returns:
        ``(frame, missing)`` — the wide macro frame and the absent series ids.
    """
    end_date = coerce_date(asof)
    repo = MacroRegimeRepository(session)

    series_by_name: dict[str, pd.Series] = {}
    for name in names:
        rows = repo.get_fred_observations(series_id=name, end_date=end_date)
        values = {row.date: float(row.value) for row in rows if row.value is not None}
        if values:
            series_by_name[name] = pd.Series(values)

    frame = pd.DataFrame(series_by_name)
    if not frame.empty:
        # A DatetimeIndex lets ``summarize_frame`` report the panel's date span.
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
    populated = set(frame.columns)
    missing = [name for name in names if name not in populated]
    return frame, missing


@tool_envelope
def get_macro_series(
    session: Session,
    names: Sequence[str],
    asof: dt.date | str,
) -> ToolResult:
    """Return a macro-series-panel summary for ``names`` as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        names: FRED series ids to resolve, in the order the panel columns should
            follow.
        asof: Inclusive upper bound; observations strictly after it are excluded
            (no look-ahead). Accepts a ``date`` or an ISO ``YYYY-MM-DD`` string.

    Returns:
        ``ok`` with a shape/coverage summary of the wide macro frame, plus
        ``asof``, ``requested`` and ``missing`` (requested series absent from the
        panel). ``err`` on empty ``names``.
    """
    if not names:
        return err("no series requested")

    frame, missing = load_macro_frame(session, names, asof)

    summary = summarize_frame(frame, name="macro")
    summary["asof"] = coerce_date(asof).isoformat()
    summary["requested"] = list(names)
    summary["missing"] = missing
    return ok(summary)


__all__ = ["get_macro_series", "load_macro_frame"]
