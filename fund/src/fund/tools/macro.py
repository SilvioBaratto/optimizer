"""``get_macro_series``: pull macro time-series for the regime/views agents.

Reads FRED observations out of the shared ``portopt_db`` layer via
:class:`~portopt_db.repositories.macro.macro_regime_repository.MacroRegimeRepository`
and hands the agent a JSON-serialisable shape/coverage **summary** plus a bounded
set of per-series statistics (level, horizon changes, z-score) and the latest observed
values — never the raw observation matrix. The tool is a pure function of
``(session, names, asof)``: same seeded data + same args ⇒ identical output.

Contract (via :func:`fund.tools._base.tool_envelope`):

* empty ``names`` ⇒ ``{ok: false, error}``;
* a series with no stored observations (or only null values) as of ``asof`` ⇒
  **flagged** in ``data["missing"]``, never raised;
* an observation counts as known on ``asof`` only once its publication lag has
  elapsed (see :func:`load_macro_frame`). The lag applies to live reads
  (``asof`` = today) exactly as to replays;
* a blank or non-string series name ⇒ **flagged** in ``data["missing"]`` without
  querying, since the repository reads an empty id as "every series";
* every other failure (e.g. a malformed ``asof``) is caught by the envelope and
  returned as ``{ok: false, error}``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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
from fund.tools._stats import level_stats

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Trailing observations returned per series; the cap keeps the reply bounded however
# long the stored history is, so a regime read gets numbers without a raw matrix.
_DEFAULT_TAIL = 12
_MAX_TAIL = 60

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
    start: dt.date | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a wide macro-series frame for ``names`` as of ``asof``, plus missing.

    Shared loader behind :func:`get_macro_series`: one column per *populated*
    series (in requested order) on a sorted ``DatetimeIndex``; ``missing`` lists
    requested series with no stored observation or only null values as of
    ``asof``, and any blank or non-string name. The caller validates non-empty
    ``names``.

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
        start: Optional inclusive lower bound; earlier observations are neither read
            nor materialised. ``None`` loads each series' whole stored history. The
            publication-lag cadence is inferred from the rows inside the bounds, so
            pair a narrow ``start`` with ``point_in_time=False``.

    Returns:
        ``(frame, missing)`` — the wide macro frame and the absent series ids; a series
        with no observation inside ``[start, asof]`` counts as absent.
    """
    end_date = coerce_date(asof)
    start_date = coerce_date(start) if start is not None else None
    repo = MacroRegimeRepository(session)

    series_by_name: dict[str, pd.Series] = {}
    for name in names:
        # The repository treats a falsy ``series_id`` as "no filter", so a blank
        # name would merge every stored series into one plausible-looking column.
        if not isinstance(name, str) or not name.strip():
            continue
        rows = repo.get_fred_observations(
            series_id=name, start_date=start_date, end_date=end_date
        )
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


def _observed_values(
    frame: pd.DataFrame, tail: int
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """The latest and trailing-``tail`` non-null observations of each series.

    Series with different frequencies or publication lags are read independently,
    so each ``latest`` is that series' own last observation, not the panel's last row.
    """
    latest: dict[str, dict[str, Any]] = {}
    recent: dict[str, list[dict[str, Any]]] = {}
    for name in frame.columns:
        observed = frame[name].dropna().tail(tail)
        points = [
            {"date": index.date().isoformat(), "value": float(value)}
            for index, value in observed.items()
        ]
        recent[str(name)] = points
        latest[str(name)] = dict(points[-1])
    return latest, recent


@tool_envelope
def get_macro_series(
    session: Session,
    names: Sequence[str],
    asof: dt.date | str,
    *,
    tail: int = _DEFAULT_TAIL,
) -> ToolResult:
    """Return a macro-series-panel summary and per-series statistics as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        names: FRED series ids to resolve, in the order the panel columns should
            follow.
        asof: Inclusive upper bound; observations strictly after it, or not yet
            published by it, are excluded (no look-ahead). Live reads are lagged
            too, so the newest monthly and quarterly prints may be absent. Accepts
            a ``date`` or an ISO ``YYYY-MM-DD`` string.
        tail: Trailing observations to return per series; clamped to
            ``[1, 60]``.

    Returns:
        ``ok`` with a shape/coverage summary of the wide macro frame, plus
        ``asof``, ``requested``, ``missing`` (requested series absent from the
        panel), ``stats`` (per populated series its ``last`` level and
        ``last_date``, the level differences ``change_1m``/``change_3m``/
        ``change_12m`` and the ``zscore`` of the last level against the series'
        history to ``asof``), ``latest`` (each series' last observation as
        ``{date, value}``) and ``recent`` (its trailing ``tail`` observations, oldest
        first). ``err`` on empty ``names``.
    """
    if not names:
        return err("no series requested")

    frame, missing = load_macro_frame(session, names, asof, point_in_time=True)

    summary = summarize_frame(frame, name="macro")
    summary["asof"] = coerce_date(asof).isoformat()
    summary["requested"] = list(names)
    summary["missing"] = missing
    summary["stats"] = level_stats(frame)
    summary["latest"], summary["recent"] = _observed_values(
        frame, max(1, min(tail, _MAX_TAIL))
    )
    return ok(summary)


__all__ = ["get_macro_series", "load_macro_frame"]
