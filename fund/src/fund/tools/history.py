"""``get_portfolio_history``: a portfolio's own recent rebalance activity.

Rolls up the per-portfolio overlay
(:class:`~portopt_db.repositories.orders.portfolio_journal_repository.PortfolioJournalRepository`)
plus the prior runs / paper orders / holdings as of ``asof`` (the ``fund.audit``
repos) into a compact, bounded summary for the PM — never a raw matrix. The tool is a
pure function of ``(session, portfolio_id, asof, lookback)``.

Contract (via :func:`fund.tools._base.tool_envelope`):

* no history ⇒ ``ok`` with empty ``rebalances`` / ``runs`` / ``orders`` and an
  empty ``current_holdings``;
* a malformed ``asof`` / ``portfolio_id`` (or any other failure) is caught by the
  envelope and returned as ``{ok: false, error}``.

Module-top repo imports follow the leaf-tool convention (as in
:mod:`fund.tools.orders`); ``fund.audit`` is part of the agent machinery here, so
pulling it at import time is expected.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING

from portopt_db.repositories.orders.portfolio_journal_repository import (
    PortfolioJournalRepository,
)

from fund.audit import AgentRunRepository, OrderRepository, PositionRepository
from fund.tools._base import ToolResult, coerce_date, ok, tool_envelope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from portopt_db.models import PortfolioJournal
    from sqlalchemy.orm import Session


def _coerce_uuid(portfolio_id: uuid.UUID | str) -> uuid.UUID:
    """Normalise ``portfolio_id`` to a ``UUID`` (accepts a canonical string)."""
    if isinstance(portfolio_id, uuid.UUID):
        return portfolio_id
    return uuid.UUID(portfolio_id)


def _holdings_as_of(
    session: Session,
    pid: uuid.UUID,
    end: dt.date,
    known_overlays: Sequence[PortfolioJournal],
) -> dict[str, float]:
    """Return ``{ticker: weight}`` held after the last rebalance dated ``<= end``.

    ``positions`` is a single snapshot replaced on every approve, so once a later
    bar has been approved it describes the future of a back-dated ``end``. In that
    case the weights are rebuilt from the newest overlay at or before ``end``,
    which stores the same allocation the snapshot was written from.

    Args:
        session: A sync ``portopt_db`` session; not owned here.
        pid: Owning portfolio.
        end: Inclusive as-of date.
        known_overlays: The portfolio's overlays dated ``<= end``, newest first.
    """
    snapshot = PositionRepository(session).get_holdings(pid)
    if all(p.asof <= end for p in snapshot):
        return {p.ticker: float(p.weight) for p in snapshot}
    if not known_overlays:
        return {}
    allocation = known_overlays[0].allocation or {}
    return {ticker: float(weight) for ticker, weight in allocation.items()}


@tool_envelope
def get_portfolio_history(
    session: Session,
    portfolio_id: uuid.UUID | str,
    asof: dt.date | str,
    lookback: int = 90,
) -> ToolResult:
    """Summarise a portfolio's rebalances, runs, orders, and current holdings.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        portfolio_id: Owning portfolio; accepts a ``UUID`` or canonical string.
        asof: Inclusive upper bound; activity after it is excluded (no
            look-ahead). Accepts a ``date`` or an ISO ``YYYY-MM-DD`` string.
        lookback: Trailing calendar-day span ending at ``asof``; runs, orders,
            and overlays dated before ``asof - lookback`` are dropped.

    Returns:
        ``ok`` with ``asof`` (isoformat), ``portfolio_id``, ``lookback``,
        ``current_holdings`` (``{ticker: weight}`` held as of ``asof``), and
        ``rebalances`` / ``runs`` / ``orders`` — each a compact list over the
        lookback window, newest first.
    """
    end = coerce_date(asof)
    since = end - dt.timedelta(days=lookback)
    pid = _coerce_uuid(portfolio_id)

    # Unwindowed: the holdings fallback needs the last rebalance even when it is
    # older than the lookback span.
    known_overlays = [
        row
        for row in PortfolioJournalRepository(session).list_for_portfolio(pid)
        if row.as_of <= end
    ]
    rebalances = [
        {
            "as_of": row.as_of.isoformat(),
            "allocation": row.allocation,
            "narrative": row.narrative,
            "run_id": str(row.run_id) if row.run_id else None,
        }
        for row in known_overlays
        if row.as_of >= since
    ]

    runs = [
        {"asof": run.asof.isoformat(), "status": run.status, "weights": run.weights}
        for run in AgentRunRepository(session).list_runs_for_portfolio(pid, since=since)
        if run.asof <= end
    ]

    orders = [
        {
            "asof": order.asof.isoformat(),
            "fill_date": order.fill_date.isoformat(),
            "weights": order.weights,
        }
        for order in OrderRepository(session).list_for_portfolio(pid)
        if since <= order.asof <= end
    ]

    holdings = _holdings_as_of(session, pid, end, known_overlays)

    return ok(
        {
            "asof": end.isoformat(),
            "portfolio_id": str(pid),
            "lookback": lookback,
            "current_holdings": holdings,
            "rebalances": rebalances,
            "runs": runs,
            "orders": orders,
        }
    )


__all__ = ["get_portfolio_history"]
