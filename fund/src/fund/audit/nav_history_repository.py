"""``NavHistoryRepository`` — fund-side behavior over the ``nav_history`` model.

The ``nav_history`` *model* lives in ``portopt_db`` (shared schema, Alembic-owned);
its *behavior* lives here, mirroring the ``positions`` / ``agent_runs`` /
``paper_orders`` model/repo split. Sits on ``portopt_db.repository``'s
``RepositoryBase`` and opens no session of its own — the caller injects a sync
session and owns the transaction boundary (no ``commit`` here).

Module-level imports stay clear of the agent stack (``deepagents``/``langgraph``)
so the daily NAV-mark scheduler step can lazy-import this repo without dragging
the deep-agent framework into a bare ``import fund.scheduler``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from portopt_db.models import NavHistory
from portopt_db.repository import RepositoryBase
from sqlalchemy import select


class NavHistoryRepository(RepositoryBase):
    """Upsert/read a portfolio's persisted NAV series on an injected session."""

    def upsert_navs(self, rows: list[dict[str, Any]]) -> int:
        """Upsert NAV rows idempotently, returning the count written.

        Rows are deduplicated on ``(portfolio_id, date)`` (last value wins within
        one payload), so re-marking a date overwrites rather than duplicates.
        Rows missing portfolio id, date, or nav are skipped. The caller owns the
        transaction; this issues the upsert without committing.

        Args:
            rows: Mappings with ``portfolio_id``, ``date``, ``nav`` and an
                optional ``benchmark`` key.

        Returns:
            Number of distinct rows passed to the upsert.
        """
        out: dict[tuple[uuid.UUID, dt.date], dict[str, Any]] = {}
        for r in rows:
            pid = r.get("portfolio_id")
            date = r.get("date")
            nav = r.get("nav")
            if pid is None or date is None or nav is None:
                continue
            out[(pid, date)] = {
                "id": uuid.uuid4(),
                "portfolio_id": pid,
                "date": date,
                "nav": nav,
                "benchmark": r.get("benchmark"),
            }
        if not out:
            return 0
        self._upsert(
            NavHistory,
            list(out.values()),
            constraint_name="uq_nav_history_portfolio_date",
            update_columns=["nav", "benchmark", "updated_at"],
        )
        return len(out)

    def get_navs(
        self,
        portfolio_id: uuid.UUID,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> list[NavHistory]:
        """Return a portfolio's NAV series, ordered ascending by date.

        Args:
            portfolio_id: Portfolio to query.
            start: Inclusive lower date bound; ``None`` leaves it open.
            end: Inclusive upper date bound; ``None`` leaves it open.

        Returns:
            Matching :class:`NavHistory` rows ordered by ``date``; empty when none
            match.
        """
        stmt = select(NavHistory).where(NavHistory.portfolio_id == portfolio_id)
        if start is not None:
            stmt = stmt.where(NavHistory.date >= start)
        if end is not None:
            stmt = stmt.where(NavHistory.date <= end)
        stmt = stmt.order_by(NavHistory.date)
        return list(self.session.execute(stmt).scalars().all())


__all__ = ["NavHistoryRepository"]
