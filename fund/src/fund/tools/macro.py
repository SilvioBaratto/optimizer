"""``get_macro_series``: pull macro time-series for the regime/views agents.

Reads FRED observations out of the shared ``portopt_db`` layer via
:class:`~portopt_db.repositories.macro.macro_regime_repository.MacroRegimeRepository`
and hands the agent a JSON-serialisable shape/coverage **summary** — never the
raw observation matrix. The tool is a pure function of ``(session,
names, asof)``: same seeded data + same args ⇒ identical output.

Contract (via :func:`fund.tools._base.tool_envelope`):

* empty ``names`` ⇒ ``{ok: false, error}``;
* a series with no stored observations (or only null values) as of ``asof`` ⇒
  **flagged** in ``data["missing"]``, never raised;
* an observation counts as known on ``asof`` only once its publication lag has
  elapsed (see :func:`load_macro_frame`). The lag applies to live reads
  (``asof`` = today) exactly as to replays;
* every other failure (e.g. a malformed ``asof``) is caught by the envelope and
  returned as ``{ok: false, error}``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import statistics
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

# FRED stores each observation under its period-start date, but the value only
# exists once the statistical agency has published it. ``fred_observations``
# keeps no release date, so availability is approximated by a conservative lag
# (days after the observation date) chosen from the series cadence: the first
# bucket whose upper bound on the median spacing between observations fits.
_CADENCE_LAG_DAYS: tuple[tuple[int, int], ...] = (
    (4, 1),  # daily: next-day release
    (10, 8),  # weekly
    (45, 62),  # monthly: period length plus the slowest releases (PCE, ~2 months)
    (120, 125),  # quarterly: advance estimate lands ~30 days after quarter end
)
_ANNUAL_LAG_DAYS = 400
# A lone observation carries no cadence information; assume the shortest lag
# instead of hiding data on a guess.
_UNKNOWN_CADENCE_LAG_DAYS = 1

# Series whose release behaviour is not captured by their observation spacing.
_SERIES_LAG_DAYS: dict[str, int] = {
    # H.10 index is observed daily but published weekly.
    "DTWEXBGS": 8,
    # Smoothed probability: month t is published after month t+1 closes.
    "RECPROUSM156N": 92,
    # GDP-based indicator waits for the second GDP estimate.
    "JHGDPBRINDX": 185,
    # NBER dates turning points up to ~20 months after the fact, so the flags
    # are ex-post labels rather than information available at decision time.
    "USREC": 640,
    "USRECDM": 640,
}


def _publication_lag_days(series_id: str, dates: Sequence[dt.date]) -> int:
    """Days after its observation date that a value of the series is published.

    Args:
        series_id: FRED series id; ids in ``_SERIES_LAG_DAYS`` use their override.
        dates: Ascending observation dates, used to infer cadence for other ids.

    Returns:
        A conservative lag in days. Revisions after the first release are not
        modelled, so a value may still differ from what was first published.
    """
    override = _SERIES_LAG_DAYS.get(series_id)
    if override is not None:
        return override
    if len(dates) < 2:
        return _UNKNOWN_CADENCE_LAG_DAYS
    spacing = statistics.median(
        (later - earlier).days for earlier, later in itertools.pairwise(dates)
    )
    for max_spacing, lag in _CADENCE_LAG_DAYS:
        if spacing <= max_spacing:
            return lag
    return _ANNUAL_LAG_DAYS


def load_macro_frame(
    session: Session,
    names: Sequence[str],
    asof: dt.date | str,
    *,
    point_in_time: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a wide macro-series frame for ``names`` as of ``asof``, plus missing.

    Shared loader behind :func:`get_macro_series`: one column per *populated*
    series (in requested order) on a sorted ``DatetimeIndex``; ``missing`` lists
    requested series with no stored observation or only null values as of
    ``asof``. The caller validates non-empty ``names``.

    Args:
        session: A sync ``portopt_db`` session; the loader does not own it.
        names: FRED series ids, in the order the panel columns should follow.
        asof: Inclusive upper bound; observations strictly after it are excluded.
        point_in_time: When true (the default) an observation is also excluded
            until its estimated publication lag has elapsed, so a replayed
            decision does not see quarterly, monthly or NBER values that were
            released after ``asof``. The filter is not replay-only: a live read
            (``asof`` = today) is lagged identically, so a monthly series such
            as the OECD CLI can show up to a month later than it was actually
            released and ``USREC``/``USRECDM`` are never visible. That cost buys
            a result that depends only on ``(data, asof)``, never on the wall
            clock. Pass false to read history for a retrospective explanation
            that is not a decision.

    Returns:
        ``(frame, missing)`` — the wide macro frame and the absent series ids.
    """
    end_date = coerce_date(asof)
    repo = MacroRegimeRepository(session)

    series_by_name: dict[str, pd.Series] = {}
    for name in names:
        rows = repo.get_fred_observations(series_id=name, end_date=end_date)
        if point_in_time and rows:
            lag = _publication_lag_days(name, [row.date for row in rows])
            published_by = end_date - dt.timedelta(days=lag)
            rows = [row for row in rows if row.date <= published_by]
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
        session: A sync ``portopt_db`` session; the tool does not own it.
        names: FRED series ids to resolve, in the order the panel columns should
            follow.
        asof: Inclusive upper bound; observations strictly after it, or not yet
            published by it, are excluded (no look-ahead). Live reads are lagged
            too, so the newest monthly and quarterly prints may be absent. Accepts
            a ``date`` or an ISO ``YYYY-MM-DD`` string.

    Returns:
        ``ok`` with a shape/coverage summary of the wide macro frame, plus
        ``asof``, ``requested`` and ``missing`` (requested series absent from the
        panel). ``err`` on empty ``names``.
    """
    if not names:
        return err("no series requested")

    frame, missing = load_macro_frame(session, names, asof, point_in_time=True)

    summary = summarize_frame(frame, name="macro")
    summary["asof"] = coerce_date(asof).isoformat()
    summary["requested"] = list(names)
    summary["missing"] = missing
    return ok(summary)


__all__ = ["get_macro_series", "load_macro_frame"]
