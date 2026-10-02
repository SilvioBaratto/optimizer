"""``get_recent_events``: the recent global daily-digest window for the economist.

Reads ``market_journal`` rows out of the shared ``portopt_db`` layer via
:class:`~portopt_db.repositories.market_data.market_journal_repository.MarketJournalRepository`
and hands the agent a compact, bounded summary of the digests in a trailing
window — never a raw matrix. The tool is a pure function of ``(session, asof,
window)``: same seeded data + same args ⇒ identical output.

Contract (via :func:`fund.tools._base.tool_envelope`):

* an empty window ⇒ ``ok`` with ``count: 0`` and ``events: []``;
* a malformed ``asof`` (or any other failure) is caught by the envelope and
  returned as ``{ok: false, error}``.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from portopt_db.repositories.market_data.market_journal_repository import (
    MarketJournalRepository,
)

from fund.tools._base import ToolResult, coerce_date, ok, tool_envelope

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@tool_envelope
def get_recent_events(
    session: Session,
    asof: dt.date | str,
    window: int = 14,
) -> ToolResult:
    """Return the global daily digests in the ``window`` days up to ``asof``.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        asof: Inclusive upper bound; digests after it are excluded (no
            look-ahead). Accepts a ``date`` or an ISO ``YYYY-MM-DD`` string.
        window: Trailing calendar-day span ending at ``asof`` (inclusive); bounds
            how many digest rows the summary can carry.

    Returns:
        ``ok`` with ``asof`` (isoformat), ``window``, ``count``, and ``events`` —
        one compact entry per digest (``as_of``, ``narrative``, ``themes``) in
        ascending date order.
    """
    end = coerce_date(asof)
    start = end - dt.timedelta(days=window)
    rows = MarketJournalRepository(session).get_range(start, end)
    events = [
        {
            "as_of": row.as_of.isoformat(),
            "narrative": row.narrative,
            "themes": (row.news_themes or {}).get("themes", {}),
        }
        for row in rows
    ]
    return ok(
        {
            "asof": end.isoformat(),
            "window": window,
            "count": len(events),
            "events": events,
        }
    )


__all__ = ["get_recent_events"]
