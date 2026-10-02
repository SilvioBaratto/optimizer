"""``get_portfolio_history``: a portfolio's own recent rebalance activity.

Rolls up the per-portfolio overlay
(:class:`~portopt_db.repositories.orders.portfolio_journal_repository.PortfolioJournalRepository`)
plus the prior runs / paper orders / current holdings (the ``fund.audit`` repos)
into a compact, bounded summary for the PM — never a raw matrix. The tool is a
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
    from sqlalchemy.orm import Session


def _coerce_uuid(portfolio_id: uuid.UUID | str) -> uuid.UUID:
    """Normalise ``portfolio_id`` to a ``UUID`` (accepts a canonical string)."""
    if isinstance(portfolio_id, uuid.UUID):
        return portfolio_id
    return uuid.UUID(portfolio_id)


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
        ``current_holdings`` (``{ticker: weight}``), and ``rebalances`` / ``runs``
        / ``orders`` — each a compact list over the lookback window, newest first.
    """
    end = coerce_date(asof)
    since = end - dt.timedelta(days=lookback)
    pid = _coerce_uuid(portfolio_id)

    overlays = PortfolioJournalRepository(session).list_for_portfolio(pid, since=since)
    rebalances = [
        {
            "as_of": row.as_of.isoformat(),
            "allocation": row.allocation,
            "narrative": row.narrative,
            "run_id": str(row.run_id) if row.run_id else None,
        }
        for row in overlays
    ]

    runs = [
        {"asof": run.asof.isoformat(), "status": run.status, "weights": run.weights}
        for run in AgentRunRepository(session).list_runs_for_portfolio(pid)
        if run.asof >= since
    ]

    orders = [
        {
            "asof": order.asof.isoformat(),
            "fill_date": order.fill_date.isoformat(),
            "weights": order.weights,
        }
        for order in OrderRepository(session).list_for_portfolio(pid)
        if order.asof >= since
    ]

    holdings = {
        p.ticker: float(p.weight) for p in PositionRepository(session).get_holdings(pid)
    }

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
